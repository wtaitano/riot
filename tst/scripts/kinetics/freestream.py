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

# Regression test R3b (claude_sessions/kinetic_bgk/S0_DESIGN.md): smooth free streaming
# (nu = 0) of an entropy wave on a periodic domain. The discrete-velocity solution is
# exact, f(x, v, t) = M_d[state0(x - v_x t)](v), so the kinetic density error measures
# the semi-Lagrangian scheme alone. Pass criteria:
#   * observed spatial order >= 0.85 (linear; first order is approached from below) and >= 1.9 (quadratic, unlimited),
#   * mass and energy conserved to 1e-12 (unlimited SL is in flux form),
#   * merged and unmerged Strang half steps agree to well below the scheme error.
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

input_id = "kinetics/freestream"
resolutions = [32, 64, 128]
nv = (16, 8, 8)
vbox = 6.0
min_order = {1: 0.85, 2: 1.9}


def pid(order, nx, merge=True):
    return f"kinetics_r3b_o{order}_n{nx}" + ("" if merge else "_unmerged")


def cases():
    for order in (1, 2):
        for nx in resolutions:
            yield order, nx, True
    yield 2, resolutions[1], False


def run(**kwargs):
    clean_outputs(*[pid(*c) for c in cases()])
    riot.generate(input_id + ".py")
    for order, nx, merge in cases():
        riot.run(
            input_id + ".rin",
            [
                "parthenon/job/problem_id=" + pid(order, nx, merge),
                f"parthenon/mesh/nx1={nx}",
                f"parthenon/meshblock/nx1={nx // 4}",
                f"kinetics/sl_order={order}",
                "kinetics/merge_half_steps=" + ("true" if merge else "false"),
                f"kinetics/nv1={nv[0]}",
                f"kinetics/nv2={nv[1]}",
                f"kinetics/nv3={nv[2]}",
            ],
        )


def density_error(problem_id, final=True):
    d0 = phdf(f"build/src/{problem_id}.out1.00000.phdf")
    d = phdf(f"build/src/{problem_id}.out1.{'final' if final else '00000'}.phdf")
    x, r0 = read_line(d0, "kinetics.rho")
    _, T0 = read_line(d0, "kinetics.temperature")
    # Initial state rho = 1 + A sin(2 pi x), uniform p and u = (0.5, 0, 0).
    s = np.sin(2 * np.pi * x)
    A = np.sum((r0 - 1.0) * s) / np.sum(s * s)
    p0 = np.mean(r0 * T0)
    grids = [velocity_nodes(n, -vbox, vbox) for n in nv]

    def state(xf):
        rho = 1.0 + A * np.sin(2 * np.pi * xf)
        u = np.zeros((xf.size, 3))
        u[:, 0] = 0.5
        return rho, u, p0 / rho

    _, r = read_line(d, "kinetics.rho")
    n, _, _ = free_streaming_moments(
        x, d.Time, state, [g[0] for g in grids], [g[1] for g in grids]
    )
    return np.mean(np.abs(r - n)), r


def conservation(problem_id):
    fname = f"build/src/{problem_id}.out2.hst"
    with open(fname) as fh:
        header = [line for line in fh if line.startswith("# [1]")][0]
    cols = {}
    for tok in header[2:].split():
        if "=" in tok:
            idx, name = tok.split("=")
            cols[name] = int(idx[1:-1]) - 1
    h = np.loadtxt(fname)
    m = h[:, cols["kinetics_sums_0"]]
    e = h[:, cols["kinetics_sums_4"]]
    return max(np.max(np.abs(m - m[0])) / m[0], np.max(np.abs(e - e[0])) / e[0])


def analyze():
    ok = True
    # Instrument check
    e0, _ = density_error(pid(2, resolutions[0]), final=False)
    if e0 > 1.0e-12:
        logger.warning(f"comparator does not reproduce the initial state: {e0:.3e}")
        ok = False
    for order in (1, 2):
        errs = [density_error(pid(order, nx))[0] for nx in resolutions]
        rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
        logger.debug(f"order {order}: L1 errors {errs}, observed orders {rates}")
        if min(rates) < min_order[order]:
            logger.warning(f"order {order}: observed orders {rates}")
            ok = False
    for c in cases():
        drift = conservation(pid(*c))
        logger.debug(f"{pid(*c)}: invariant drift {drift:.3e}")
        if drift > 1.0e-12:
            logger.warning(f"{pid(*c)}: invariant drift {drift:.3e}")
            ok = False
    # Merged vs unmerged half steps: same order of accuracy, nearly identical result.
    err_m, rm = density_error(pid(2, resolutions[1]))
    err_u, ru = density_error(pid(2, resolutions[1], merge=False))
    diff = np.mean(np.abs(rm - ru))
    logger.debug(f"merged {err_m:.3e}, unmerged {err_u:.3e}, difference {diff:.3e}")
    if not (diff < 0.5 * err_m and abs(err_u - err_m) < 0.5 * err_m):
        logger.warning(
            f"merged/unmerged: errors {err_m:.3e}, {err_u:.3e}, diff {diff:.3e}"
        )
        ok = False
    return ok
