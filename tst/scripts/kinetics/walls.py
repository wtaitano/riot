# ========================================================================================
# (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================
# This file was made in part with generative AI.

# Kinetic wall boundary conditions (S0 step 5, claude_sessions/kinetic_bgk/S0_DESIGN.md).
# A closed 1D box (walls at both ends) of the Sod deck:
#   * specular walls: mass and energy conserved to roundoff (linear SL, nu = 1e4),
#   * diffuse walls: mass conserved to roundoff (zero net mass flux), energy not,
#   * diffuse walls at the gas temperature around a uniform gas at rest: the state is
#     an exact discrete equilibrium of the walls and stays unchanged to roundoff,
#   * diffuse walls hotter than a uniform gas (nu = 0): the gas energy increases
#     every step (negative control for the energy check of the specular case).

import logging

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs, read_line

logger = logging.getLogger("riot" + __name__[7:])

input_id = "kinetics/sod"
common = [
    "kinetics/nv1=16",
    "kinetics/nv2=8",
    "kinetics/nv3=8",
    "kinetics/min_vth_over_dv=0",
    "kinetics/sl_order=1",
    "parthenon/output1/dt=0.2",
    "parthenon/output3/dt=-1",
    "parthenon/output2/data_format=%.17e",
    "parthenon/time/tlim=0.4",
]
uniform = [
    "shock_tube/rho_r=1.0",
    "shock_tube/P_r=1.0",
]


def walls(kind, T=1.0):
    args = [f"kinetics/ix1_bc={kind}", f"kinetics/ox1_bc={kind}"]
    if kind == "diffuse":
        args += [f"kinetics/ix1_wall_T={T}", f"kinetics/ox1_wall_T={T}"]
    return args


cases = {
    "kinetics_walls_specular": walls("specular") + ["kinetics/nu0=1.0e4"],
    "kinetics_walls_diffuse": walls("diffuse") + ["kinetics/nu0=1.0e4"],
    "kinetics_walls_diffuse_eq": walls("diffuse") + uniform + ["kinetics/nu0=0.0"],
    "kinetics_walls_diffuse_hot": walls("diffuse", 2.0)
    + uniform
    + ["kinetics/nu0=0.0"],
}


def run(**kwargs):
    clean_outputs(*cases.keys())
    riot.generate(input_id + ".py")
    for pid, args in cases.items():
        riot.run(input_id + ".rin", ["parthenon/job/problem_id=" + pid] + common + args)


def history(pid):
    fname = f"build/src/{pid}.out2.hst"
    with open(fname) as fh:
        header = [line for line in fh if line.startswith("# [1]")][0]
    cols = {}
    for tok in header[2:].split():
        if "=" in tok:
            idx, name = tok.split("=")
            cols[name] = int(idx[1:-1]) - 1
    h = np.loadtxt(fname)
    return h[:, cols["kinetics_sums_0"]], h[:, cols["kinetics_sums_4"]]


def drift(a):
    return np.max(np.abs(a - a[0])) / abs(a[0])


def analyze():
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            logger.warning(msg)
            ok = False

    m, e = history("kinetics_walls_specular")
    logger.debug(f"specular: mass {drift(m):.3e}, energy {drift(e):.3e}")
    check(drift(m) < 1e-11 and drift(e) < 1e-11, "specular walls do not conserve")

    m, e = history("kinetics_walls_diffuse")
    logger.debug(f"diffuse: mass {drift(m):.3e}, energy {drift(e):.3e}")
    check(drift(m) < 1e-11, "diffuse walls do not conserve mass")
    check(drift(e) > 1e-6, "diffuse walls exchange no energy with a non-uniform gas")

    d0 = phdf("build/src/kinetics_walls_diffuse_eq.out1.00000.phdf")
    d1 = phdf("build/src/kinetics_walls_diffuse_eq.out1.final.phdf")
    for var in ("kinetics.rho", "kinetics.temperature"):
        _, a0 = read_line(d0, var)
        _, a1 = read_line(d1, var)
        err = np.max(np.abs(a1 - a0) / np.abs(a0))
        logger.debug(f"diffuse equilibrium: {var} change {err:.3e}")
        check(err < 1e-12, f"diffuse walls at the gas temperature changed {var}")

    m, e = history("kinetics_walls_diffuse_hot")
    logger.debug(
        f"hot diffuse walls: mass {drift(m):.3e}, energy gain {e[-1] / e[0] - 1:.3e}"
    )
    check(drift(m) < 1e-11, "hot diffuse walls do not conserve mass")
    check(np.all(np.diff(e) > 0.0), "hot diffuse walls do not heat the gas every step")
    return ok
