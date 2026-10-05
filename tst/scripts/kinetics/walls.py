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
# Hydro sees the walls as no-penetration (mesh reflecting) walls, as the closure coupling
# requires (S3_DESIGN.md, S3-Q10), and takes the kinetic wall fluxes (S3-Q9/Q14): in
# every coupled case the hydro mass, momentum and energy change by exactly the kinetic
# amounts (to 1e-12 of the totals), also for TT and a 2D box (diffuse x1, specular x2
# walls, 2 ranks). Control: uncoupled, the hot walls heat the kinetic gas but not hydro.
# Energy is not checked on the two Sod cases: when the (smeared) kinetic shock reaches
# the wall before the hydro shock, the hydro wall cell cools to T ~ 0.04 and hydro gains
# energy in one or two steps (4e-4 specular, 6e-3 diffuse); without the wall flux hydro
# conserves to 3e-15. Mass and momentum still match. Without LoMaC the hydro and kinetic
# states at the wall are different gases (S3_DESIGN.md, step 2 log).

import logging

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs, read_history, read_line

logger = logging.getLogger("riot" + __name__[7:])

input_id = "kinetics/sod"
common = [
    "kinetics/nv1=16",
    "kinetics/nv2=8",
    "kinetics/nv3=8",
    "kinetics/min_vth_over_dv=0",
    "kinetics/sl_order=1",
    "parthenon/mesh/ix1_bc=reflecting",
    "parthenon/mesh/ox1_bc=reflecting",
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
    # The T = 2 walls put 5e-6 of the mass on the (coarse) velocity-box edge by cycle
    # 100; this case checks only mass and the sign of the energy exchange.
    "kinetics_walls_diffuse_hot": walls("diffuse", 2.0)
    + uniform
    + ["kinetics/nu0=0.0", "kinetics/edge_mass_abort=1e-4"],
}
tt = ["kinetics/representation=tt", "kinetics/tt_eps=1e-14"]
cases["kinetics_walls_diffuse_hot_tt"] = cases["kinetics_walls_diffuse_hot"] + tt
cases["kinetics_walls_diffuse_hot_off"] = cases["kinetics_walls_diffuse_hot"] + [
    "kinetics/closure_coupling=false"
]
box2d = [
    "parthenon/mesh/nx1=16",
    "parthenon/mesh/nx2=16",
    "parthenon/meshblock/nx1=8",
    "parthenon/meshblock/nx2=8",
    "kinetics/nv1=12",
    "kinetics/nv2=12",
    "kinetics/nv3=12",
    "kinetics/min_vth_over_dv=0",
    "kinetics/sl_order=1",
    "parthenon/mesh/ix1_bc=reflecting",
    "parthenon/mesh/ox1_bc=reflecting",
    "parthenon/mesh/ix2_bc=reflecting",
    "parthenon/mesh/ox2_bc=reflecting",
    "kinetics/ix1_bc=diffuse",
    "kinetics/ox1_bc=diffuse",
    "kinetics/ix1_wall_T=2.0",
    "kinetics/ox1_wall_T=2.0",
    "kinetics/ix2_bc=specular",
    "kinetics/ox2_bc=specular",
    "kinetics/edge_mass_abort=1e-3",
    "parthenon/output1/dt=-1",
    "parthenon/output3/dt=-1",
    "parthenon/output2/data_format=%.17e",
    "parthenon/time/tlim=0.05",
]
cases2d = {
    "kinetics_walls_2d_dense": box2d,
    "kinetics_walls_2d_tt": box2d + tt + ["kinetics/tt_rank_max=24"],
}


def run(**kwargs):
    clean_outputs(*cases.keys(), *cases2d.keys())
    riot.generate(input_id + ".py")
    for pid, args in cases.items():
        riot.run(input_id + ".rin", ["parthenon/job/problem_id=" + pid] + common + args)
    riot.generate("kinetics/blast2d.py")
    for pid, args in cases2d.items():
        riot.mpirun(
            2, "kinetics/blast2d.rin", ["parthenon/job/problem_id=" + pid] + args
        )


def exchange(pid):
    """Largest |kinetic - hydro| change of mass, momentum, energy over the run, relative
    to the totals, and the kinetic energy change."""
    h, cols = read_history(pid)
    s = lambda n: h[:, cols[f"kinetics_sums_{n}"]]
    m0, e0 = s(0)[0], s(4)[0]
    err = np.max(np.abs((s(0) - s(0)[0]) - (s(7) - s(7)[0]))) / m0
    for a in range(3):
        err = max(err, np.max(np.abs(s(1 + a) - s(8 + a))) / m0)
    err_e = np.max(np.abs((s(4) - e0) - (s(11) - s(11)[0]))) / e0
    return err, err_e, s(4)[-1] / e0 - 1.0


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

    # Hydro wall fluxes = kinetic wall fluxes
    sod = ("kinetics_walls_specular", "kinetics_walls_diffuse")
    for pid in [p for p in cases if not p.endswith("_off")] + list(cases2d):
        err, err_e, gain = exchange(pid)
        logger.debug(
            f"{pid}: kinetic - hydro mass/momentum {err:.3e}, energy {err_e:.3e}, "
            f"kinetic energy gain {gain:.3e}"
        )
        check(
            err < 1e-12, f"{pid}: hydro and kinetic wall exchange differ by {err:.3e}"
        )
        if pid not in sod:
            check(
                err_e < 1e-12, f"{pid}: hydro and kinetic energy differ by {err_e:.3e}"
            )
    for pid in ("kinetics_walls_diffuse_hot_tt", "kinetics_walls_2d_dense"):
        check(exchange(pid)[2] > 1e-2, f"{pid}: the walls do not heat the gas")
    err_e = exchange("kinetics_walls_diffuse_hot_off")[1]
    logger.debug(f"uncoupled control: energy exchange difference {err_e:.3e}")
    check(err_e > 1e-2, "uncoupled control: hydro followed the kinetic wall exchange")
    return ok
