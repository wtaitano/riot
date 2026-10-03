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

# Regression test R10: time order of the streaming + collision integrator in the stiff
# regime (claude_sessions/kinetic_bgk/OPEN_QUESTIONS.md, OQ1). Smooth entropy wave of the
# freestream deck, quadratic unlimited SL, nu = 1e5, dx and dt refined together
# (dt = 0.64 dx, forced, so the substep count is fixed per integrator) and nu h >> 1 on
# every mesh. Self-convergence of the kinetic density with 2:1 restriction. Pass criteria:
#   * sl_dirk2 (default): observed order >= 1.8 (measured 1.98, 1.99),
#   * strang: observed order <= 1.3 (measured 1.06, 0.98), so the comparator tells a
#     first-order scheme from a second-order one,
#   * mass and energy are conserved to 1e-10 (measured up to 7e-12: the equilibrium-solve
#     tolerance eq_tol = 1e-13 accumulated over the relaxations, as in R2).
# The same sweep is repeated with representation = tt (tt_eps = 1e-14, S1 step 5): the
# same order criteria and conservation tolerance, and the TT density must equal the dense
# density to 1e-9 on every mesh (measured <= 3.4e-11), so TT rounding does not change the
# order of either integrator.

import logging

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs, read_line

logger = logging.getLogger("riot" + __name__[7:])

input_id = "kinetics/freestream"
resolutions = [32, 64, 128, 256]
integrators = ["sl_dirk2", "strang"]
nu = 1.0e5
min_order_dirk = 1.8
max_order_strang = 1.3


reps = ["dense", "tt"]
tol_tt = 1.0e-9


def pid(integrator, nx, rep="dense"):
    return f"kinetics_r10_{integrator}_n{nx}" + ("" if rep == "dense" else "_tt")


def run(**kwargs):
    clean_outputs(
        *[pid(i, nx, r) for i in integrators for nx in resolutions for r in reps]
    )
    riot.generate(input_id + ".py")
    for rep in reps:
        for integrator in integrators:
            for nx in resolutions:
                riot.run(
                    input_id + ".rin",
                    [
                        "parthenon/job/problem_id=" + pid(integrator, nx, rep),
                        f"kinetics/representation={rep}",
                        "kinetics/tt_eps=1e-14",
                        f"kinetics/integrator={integrator}",
                        f"kinetics/nu0={nu}",
                        "kinetics/nv1=16",
                        "kinetics/nv2=8",
                        "kinetics/nv3=8",
                        "kinetics/min_vth_over_dv=0",
                        "kinetics/edge_mass_warn=1",
                        f"parthenon/mesh/nx1={nx}",
                        f"parthenon/meshblock/nx1={nx // 4}",
                        f"parthenon/time/dt_force={0.64 / nx}",
                    ],
                )


def history(problem_id):
    fname = f"build/src/{problem_id}.out2.hst"
    with open(fname) as fh:
        header = [line for line in fh if line.startswith("# [1]")][0]
    cols = {}
    for tok in header[2:].split():
        if "=" in tok:
            idx, name = tok.split("=")
            cols[name] = int(idx[1:-1]) - 1
    return np.loadtxt(fname), cols


def densities(integrator, rep):
    return [
        read_line(
            phdf(f"build/src/{pid(integrator, nx, rep)}.out1.final.phdf"),
            "kinetics.rho",
        )[1]
        for nx in resolutions
    ]


def orders(integrator, rep="dense"):
    rho = densities(integrator, rep)
    diffs = [
        np.mean(np.abs(rho[i] - 0.5 * (rho[i + 1][0::2] + rho[i + 1][1::2])))
        for i in range(len(rho) - 1)
    ]
    return [np.log2(diffs[i] / diffs[i + 1]) for i in range(len(diffs) - 1)], diffs


def analyze():
    ok = True
    for rep in reps:
        for integrator in integrators:
            rates, diffs = orders(integrator, rep)
            logger.debug(f"{rep} {integrator}: differences {diffs}, orders {rates}")
            if integrator == "sl_dirk2" and min(rates) < min_order_dirk:
                logger.warning(
                    f"{rep} sl_dirk2: observed orders {rates} below {min_order_dirk}"
                )
                ok = False
            if integrator == "strang" and max(rates) > max_order_strang:
                logger.warning(
                    f"{rep} strang: observed orders {rates} above {max_order_strang}; "
                    "the order comparator does not resolve the stiff regime"
                )
                ok = False
            for nx in resolutions:
                h, cols = history(pid(integrator, nx, rep))
                nuh = nu * h[0, 1] / h[1:, cols["kinetics_substeps"]].max()
                if nuh < 10.0:
                    logger.warning(
                        f"{pid(integrator, nx, rep)}: nu h = {nuh:.3g} is not stiff"
                    )
                    ok = False
                for name in ("kinetics_sums_0", "kinetics_sums_4"):
                    a = h[:, cols[name]]
                    drift = np.max(np.abs(a - a[0])) / a[0]
                    if drift > 1.0e-10:
                        logger.warning(
                            f"{pid(integrator, nx, rep)}: {name} drift {drift:.3e}"
                        )
                        ok = False
    for integrator in integrators:
        for nx, d, t in zip(
            resolutions, densities(integrator, "dense"), densities(integrator, "tt")
        ):
            e = np.max(np.abs(d - t)) / np.max(np.abs(d))
            logger.debug(f"{integrator} n{nx}: tt vs dense density {e:.3e}")
            if e > tol_tt:
                logger.warning(
                    f"{integrator} n{nx}: tt density differs from dense by {e:.3e}"
                )
                ok = False
    return ok
