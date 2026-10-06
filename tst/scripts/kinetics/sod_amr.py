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

# Sod shock tube with adaptive refinement (claude_sessions/kinetic_bgk/S4_DESIGN.md,
# S4-Q8 i). Kinetics Sod deck, nu = 1e3, closure coupling, Nv 16^3, t = 0.15: a root
# mesh of 64 cells with up to two extra levels, refined by the hydro density gradient
# and the kinetic non-equilibrium criterion, against a uniform mesh of 256 cells (the
# finest AMR level) and a uniform mesh of 64 cells (the root level).
# Pass:
#   * the AMR kinetic density is closer to the uniform-fine one than the uniform-coarse
#     run, by a factor >= 5 in L1 (measured 16: 7.6e-4 vs 1.2e-2);
#   * AMR uses on average at most 0.75 of the uniform-fine cells (measured 0.52);
#   * a remesh step changes the hydro mass and energy no more than an ordinary step (the
#     outflow boundaries pass mass every step; measured 5e-14 vs 3e-12);
#   * TT AMR (tt_eps = 1e-10) follows dense AMR: same block layout, L1 difference of the
#     kinetic density below 1e-6 (measured 2e-9).

import logging

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs, read_history

logger = logging.getLogger("riot" + __name__[7:])

common = [
    "kinetics/nv1=16",
    "kinetics/nv2=16",
    "kinetics/nv3=16",
    "kinetics/min_vth_over_dv=0",
    "kinetics/nu0=1.0e3",
    "parthenon/time/tlim=0.15",
    "parthenon/output1/dt=0.15",
    "parthenon/output3/dt=-1",
    "parthenon/output2/dt=-1",
    "parthenon/output2/dn=1",
    "parthenon/output2/data_format=%.17e",
] + [f"kinetics/v{d}{e}" for d in (1, 2, 3) for e in ("min=-8", "max=8")]
amr = [
    "parthenon/mesh/nx1=64",
    "parthenon/meshblock/nx1=8",
    "parthenon/mesh/refinement=adaptive",
    "parthenon/mesh/numlevel=3",
    "parthenon/mesh/derefine_count=5",
    "parthenon/refinement0/method=derivative_order_1",
    "parthenon/refinement0/field=c.c.bulk.rho",
    "parthenon/refinement0/refine_tol=0.05",
    "parthenon/refinement0/derefine_tol=0.01",
    "kinetics/amr_noneq_refine=1e-2",
    "kinetics/amr_noneq_derefine=1e-3",
]
cases = {
    "fine": ["parthenon/mesh/nx1=256", "parthenon/meshblock/nx1=16"],
    "coarse": ["parthenon/mesh/nx1=64", "parthenon/meshblock/nx1=16"],
    "amr": amr,
    "amr_tt": amr
    + [
        "kinetics/representation=tt",
        "kinetics/tt_eps=1e-10",
        "kinetics/tt_rank_max=16",
    ],
}


def pid(case):
    return f"kinetics_sod_amr_{case}"


def run(**kwargs):
    clean_outputs(*[pid(c) for c in cases])
    riot.generate("kinetics/sod.py")
    for case, args in cases.items():
        riot.run(
            "kinetics/sod.rin",
            ["parthenon/job/problem_id=" + pid(case)] + common + args,
        )


def cells(case):
    d = phdf(f"build/src/{pid(case)}.out1.final.phdf")
    x = np.asarray(d.x).ravel()
    xf = np.asarray(d.xf)
    dx = (xf[:, 1:] - xf[:, :-1]).ravel()
    r = np.asarray(d.Get("kinetics.rho", flatten=False)).reshape(-1)
    o = np.argsort(x)
    return x[o], r[o], dx[o], d


def l1_vs(case, ref):
    """L1 difference of the kinetic density, with ref averaged onto case's cells."""
    xr, rr, dr, _ = cells(ref)
    x, r, dx, _ = cells(case)
    refc = np.array(
        [
            np.sum(rr[m] * dr[m]) / np.sum(dr[m])
            for m in (
                (xr >= xc - 0.5 * h) & (xr < xc + 0.5 * h) for xc, h in zip(x, dx)
            )
        ]
    )
    return np.sum(np.abs(r - refc) * dx)


def analyze():
    ok = True
    e_amr = l1_vs("amr", "fine")
    e_coarse = l1_vs("coarse", "fine")
    logger.info(f"L1 vs uniform fine: AMR {e_amr:.3e}, uniform coarse {e_coarse:.3e}")
    if not e_amr * 5.0 < e_coarse:
        logger.warning(f"AMR not closer to fine: {e_amr:.3e} vs {e_coarse:.3e}")
        ok = False
    h, c = read_history(pid("amr"))
    levels = sorted(k for k in c if k.startswith("kinetics_blocks_per_level"))
    nx_block = 8
    mean_cells = np.mean(sum(h[:, c[k]] for k in levels)) * nx_block
    frac = mean_cells / 256.0
    logger.info(f"AMR mean cells {mean_cells:.1f} = {frac:.2f} of uniform fine")
    if frac > 0.75:
        logger.warning(f"AMR uses {frac:.2f} of the uniform-fine cells")
        ok = False
    # Outflow boundaries: the hydro sums change by the boundary flux every step. A
    # remesh itself must not change them (restriction and prolongation are
    # conservative).
    remesh = np.diff(h[:, c["nbtotal"]]) != 0
    for n in ("kinetics_sums_7", "kinetics_sums_11"):
        dm = np.abs(np.diff(h[:, c[n]])) / np.abs(h[0, c[n]])
        at, off = np.max(dm[remesh]), np.max(dm[~remesh])
        logger.info(f"AMR {n}: step change at remeshes {at:.1e}, otherwise {off:.1e}")
        if not (remesh.any() and at < 10.0 * off + 1.0e-13):
            logger.warning(f"AMR {n}: remesh changes the hydro sum by {at:.1e}")
            ok = False
    _, _, _, da = cells("amr")
    _, _, _, dt = cells("amr_tt")
    if not np.array_equal(np.asarray(da.level), np.asarray(dt.level)):
        logger.warning("TT and dense AMR block layouts differ")
        ok = False
    else:
        e_tt = l1_vs("amr_tt", "amr")
        logger.info(f"TT AMR vs dense AMR: L1 {e_tt:.2e}")
        if e_tt > 1.0e-6:
            logger.warning(f"TT AMR vs dense AMR: L1 {e_tt:.2e}")
            ok = False
    return ok
