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

# Regression test R3 (claude_sessions/kinetic_bgk/S0_DESIGN.md): Sod shock tube in the
# free-molecular limit (nu = 0). The discrete-velocity solution is exact,
# f(x, v, t) = f0(x - v_x t, v), with f0 the discrete equilibria of the two Sod states.
# A jump in f limits every scheme to sub-first-order convergence in L1 (about
# 1/2 for linear, 2/3 for quadratic on a smooth-in-v solution); pass criteria:
#   * the density, velocity and temperature errors decrease with resolution,
#   * the quadratic scheme beats the linear one at every resolution,
#   * the errors at the finest resolution are below fixed bounds (set from the first
#     verified run, with margin).
# Instrument check: the comparator reproduces the initial dump to roundoff.

import logging

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import (
    clean_outputs,
    free_streaming_moments,
    read_line,
    velocity_nodes,
)

logger = logging.getLogger("riot" + __name__[7:])

input_id = "kinetics/sod"
resolutions = [64, 128, 256]
nv = (32, 16, 16)
vbox = 8.0
# Finest-resolution L1 bounds (verified run: rho 8.0e-3, u 1.6e-2, T 5.6e-3 quadratic;
# rho 1.44e-2, u 2.95e-2, T 1.09e-2 linear).
bounds = {1: (1.6e-2, 3.3e-2, 1.2e-2), 2: (9.0e-3, 1.8e-2, 6.3e-3)}


def pid(order, nx):
    return f"kinetics_r3_o{order}_n{nx}"


def run(**kwargs):
    clean_outputs(*[pid(o, n) for o in (1, 2) for n in resolutions])
    riot.generate(input_id + ".py")
    for order in (1, 2):
        for nx in resolutions:
            riot.run(
                input_id + ".rin",
                [
                    "parthenon/job/problem_id=" + pid(order, nx),
                    "kinetics/nu0=0.0",
                    f"kinetics/sl_order={order}",
                    f"parthenon/mesh/nx1={nx}",
                    f"parthenon/meshblock/nx1={nx // 4}",
                    f"kinetics/nv1={nv[0]}",
                    f"kinetics/nv2={nv[1]}",
                    f"kinetics/nv3={nv[2]}",
                    "kinetics/min_vth_over_dv=0",
                    "parthenon/output1/dt=0.2",
                    "parthenon/output3/dt=-1",
                ],
            )


def state(xf):
    left = xf < 0.5
    rho = np.where(left, 1.0, 0.125)
    theta = np.where(left, 1.0, 0.8)  # k_B / m = 1: theta = P / rho
    return rho, np.zeros((xf.size, 3)), theta


def errors(problem_id, which="final"):
    d = phdf(f"build/src/{problem_id}.out1.{which}.phdf")
    grids = [velocity_nodes(n, -vbox, vbox) for n in nv]
    x, rho = read_line(d, "kinetics.rho")
    _, u = read_line(d, "kinetics.velocity", 0)
    _, T = read_line(d, "kinetics.temperature")
    n, ux, th = free_streaming_moments(
        x, d.Time, state, [g[0] for g in grids], [g[1] for g in grids]
    )
    return (
        np.mean(np.abs(rho - n)),
        np.mean(np.abs(u - ux)),
        np.mean(np.abs(T - th)),
    )


def analyze():
    ok = True
    e0 = max(errors(pid(2, resolutions[0]), "00000"))
    if e0 > 1.0e-12:
        logger.warning(f"comparator does not reproduce the initial state: {e0:.3e}")
        ok = False
    errs = {o: [errors(pid(o, n)) for n in resolutions] for o in (1, 2)}
    for order in (1, 2):
        for k, name in enumerate(("rho", "u", "T")):
            series = [e[k] for e in errs[order]]
            logger.debug(f"order {order} {name}: L1 errors {series}")
            if not all(series[i + 1] < series[i] for i in range(len(series) - 1)):
                logger.warning(f"order {order} {name}: errors do not decrease {series}")
                ok = False
            if series[-1] > bounds[order][k]:
                logger.warning(
                    f"order {order} {name}: finest error {series[-1]:.3e} "
                    f"above {bounds[order][k]:.3e}"
                )
                ok = False
    for i, nx in enumerate(resolutions):
        if not all(errs[2][i][k] < errs[1][i][k] for k in range(3)):
            logger.warning(f"nx = {nx}: quadratic not better than linear")
            ok = False
    return ok
