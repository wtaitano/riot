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

# Regression test R2 (claude_sessions/kinetic_bgk/S0_DESIGN.md): Sod shock tube at small
# Knudsen number. As nu grows the kinetic moments approach the Euler solution; at fixed
# nu they converge with the mesh. Linear (conservative, default) semi-Lagrangian
# streaming is used: quadratic + minmax limiter loses about 0.3% of mass and energy here,
# which displaces the shock (see the step-6 notes in S0_DESIGN.md). Pass criteria:
#   * the L1 density error against the exact Euler solution decreases as nu increases
#     (1e2, 1e3, 1e4) at fixed mesh,
#   * at nu = 1e5 it decreases with the mesh (128, 256, 512 cells),
#   * mass and energy are conserved to 5e-10 at nu = 1e5. The drift comes from the
#     equilibrium-solve tolerance (eq_tol = 1e-13): at 512 cells it is 1.1e-10 with
#     sl_dirk2 and 5e-11 with strang, and 3e-12 with eq_tol = 1e-15. The
#     non-conservative limiter this guards against gives 3e-3.
#   * the shock position is within 2 cells of the exact one at 512 cells.
# The exact solver is checked against RIOT's hydro solution on the same mesh.

import logging

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs, read_line, sod_exact

logger = logging.getLogger("riot" + __name__[7:])

input_id = "kinetics/sod"
common = [
    "kinetics/sl_order=1",
    "kinetics/nv1=24",
    "kinetics/nv2=12",
    "kinetics/nv3=12",
    "kinetics/min_vth_over_dv=0",
    "parthenon/output1/dt=0.2",
    "parthenon/output3/dt=-1",
    "parthenon/output2/data_format=%.17e",
]
nu_sweep = [1.0e2, 1.0e3, 1.0e4]
nx_sweep = [128, 256, 512]


def pid(nu, nx):
    return f"kinetics_r2_nu{nu:.0e}_n{nx}".replace("+", "")


def cases():
    for nu in nu_sweep:
        yield nu, 256
    for nx in nx_sweep:
        yield 1.0e5, nx


def run(**kwargs):
    clean_outputs(*[pid(*c) for c in cases()])
    riot.generate(input_id + ".py")
    for nu, nx in cases():
        riot.run(
            input_id + ".rin",
            [
                "parthenon/job/problem_id=" + pid(nu, nx),
                f"kinetics/nu0={nu}",
                f"parthenon/mesh/nx1={nx}",
                f"parthenon/meshblock/nx1={nx // 4}",
            ]
            + common,
        )


def l1_density(problem_id, var="kinetics.rho"):
    d = phdf(f"build/src/{problem_id}.out1.final.phdf")
    x, rho = read_line(d, var)
    exact = sod_exact(x, d.Time)[0]
    return np.mean(np.abs(rho - exact)), x, rho, exact


def shock_position(x, rho, post):
    mid = 0.5 * (post + 0.125)
    return x[np.where((x > 0.75) & (rho < mid))[0][0]]


def analyze():
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            logger.warning(msg)
            ok = False

    # Instrument check: RIOT hydro converges to the exact solution.
    eh = [l1_density(pid(1.0e5, nx), "c.c.bulk.rho")[0] for nx in nx_sweep]
    logger.debug(f"hydro L1 vs exact: {eh}")
    check(eh[-1] < 0.6 * eh[0], f"hydro does not converge to the exact solution: {eh}")

    ek = [l1_density(pid(nu, 256))[0] for nu in nu_sweep]
    logger.debug(f"kinetic L1 vs exact, nu = {nu_sweep}: {ek}")
    check(all(ek[i + 1] < ek[i] for i in range(len(ek) - 1)), f"no Kn trend: {ek}")

    ex = [l1_density(pid(1.0e5, nx))[0] for nx in nx_sweep]
    logger.debug(f"kinetic L1 vs exact, nx = {nx_sweep}: {ex}")
    check(all(ex[i + 1] < ex[i] for i in range(len(ex) - 1)), f"no mesh trend: {ex}")

    _, x, rho, exact = l1_density(pid(1.0e5, nx_sweep[-1]))
    post = exact[(x > 0.7) & (x < 0.85)].mean()
    dx = x[1] - x[0]
    xs, xe = shock_position(x, rho, post), shock_position(x, exact, post)
    logger.debug(f"shock at {xs:.4f}, exact {xe:.4f}")
    check(abs(xs - xe) <= 2.0 * dx, f"shock at {xs:.4f}, exact {xe:.4f}")

    fname = f"build/src/{pid(1.0e5, nx_sweep[-1])}.out2.hst"
    with open(fname) as fh:
        header = [line for line in fh if line.startswith("# [1]")][0]
    cols = {}
    for tok in header[2:].split():
        if "=" in tok:
            idx, name = tok.split("=")
            cols[name] = int(idx[1:-1]) - 1
    h = np.loadtxt(fname)
    for name in ("kinetics_sums_0", "kinetics_sums_4"):
        a = h[:, cols[name]]
        drift = np.max(np.abs(a - a[0])) / a[0]
        logger.debug(f"{name} drift {drift:.3e}")
        check(drift < 5.0e-10, f"{name} drift {drift:.3e}")
    return ok
