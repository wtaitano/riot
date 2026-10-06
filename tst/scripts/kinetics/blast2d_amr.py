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

# 2D blast with adaptive refinement, tensor-train f (claude_sessions/kinetic_bgk/
# S4_DESIGN.md, S4-Q8 ii, Q18). blast2d deck on [-1, 1]^2 (the pulse stays local),
# closure coupling, Nv 12^3, tt_eps 1e-8, 2 MPI ranks, t = 0.06: root 8^2 with up to two
# extra levels (hydro density gradient OR kinetic non-equilibrium) vs uniform 32^2 (the
# finest AMR level) and uniform 8^2.
# Pass (S4-Q18 a, the gate): relative L1 of the kinetic density of AMR vs uniform fine
# <= 1e-2 (measured 5.0e-5; uniform coarse 3.0e-2). Reported, not gated (S4-Q18 b):
# mean AMR cell fraction (measured 0.56) and the wall-time ratio.

import logging
import time

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs, read_history

logger = logging.getLogger("riot" + __name__[7:])

nranks = 2
common = [
    "parthenon/mesh/x1min=-1",
    "parthenon/mesh/x1max=1",
    "parthenon/mesh/x2min=-1",
    "parthenon/mesh/x2max=1",
    "parthenon/output3/dt=-1",
    "parthenon/output2/dt=-1",
    "parthenon/output2/dn=1",
    "parthenon/time/tlim=0.06",
    "parthenon/output1/dt=0.06",
    "kinetics/representation=tt",
    "kinetics/tt_eps=1e-8",
    "kinetics/tt_rank_max=24",
    "kinetics/tt_diag=exact",
    "kinetics/min_vth_over_dv=0",
] + [
    f"kinetics/{a}"
    for d in (1, 2, 3)
    for a in (f"nv{d}=12", f"v{d}min=-9.5", f"v{d}max=9.5")
]
coarse_mesh = [
    "parthenon/mesh/nx1=8",
    "parthenon/mesh/nx2=8",
    "parthenon/meshblock/nx1=4",
    "parthenon/meshblock/nx2=4",
]
cases = {
    "fine": [
        "parthenon/mesh/nx1=32",
        "parthenon/mesh/nx2=32",
        "parthenon/meshblock/nx1=8",
        "parthenon/meshblock/nx2=8",
    ],
    "amr": coarse_mesh
    + [
        "parthenon/mesh/refinement=adaptive",
        "parthenon/mesh/numlevel=3",
        "parthenon/mesh/derefine_count=5",
        "parthenon/refinement0/method=derivative_order_1",
        "parthenon/refinement0/field=c.c.bulk.rho",
        "parthenon/refinement0/refine_tol=0.05",
        "parthenon/refinement0/derefine_tol=0.01",
        "kinetics/amr_noneq_refine=1e-2",
        "kinetics/amr_noneq_derefine=1e-3",
    ],
    "coarse": coarse_mesh,
}
walltime = {}


def pid(case):
    return f"kinetics_blast2d_amr_{case}"


def run(**kwargs):
    clean_outputs(*[pid(c) for c in cases])
    riot.generate("kinetics/blast2d.py")
    for case, args in cases.items():
        t0 = time.time()
        riot.mpirun(
            nranks,
            "kinetics/blast2d.rin",
            ["parthenon/job/problem_id=" + pid(case)] + common + args,
        )
        walltime[case] = time.time() - t0


def field(case):
    d = phdf(f"build/src/{pid(case)}.out1.final.phdf")
    x, y, nb = np.asarray(d.x), np.asarray(d.y), d.NumBlocks
    r = np.asarray(d.Get("kinetics.rho", flatten=False)).reshape(
        nb, y.shape[1], x.shape[1]
    )
    dx = np.asarray(d.xf)[:, 1:] - np.asarray(d.xf)[:, :-1]
    dy = np.asarray(d.yf)[:, 1:] - np.asarray(d.yf)[:, :-1]
    return x, y, r, dx, dy, nb


def rel_l1(case, ref="fine"):
    """Relative L1 of the kinetic density, ref averaged onto case's cells."""
    xf, yf, rf, _, _, nbf = field(ref)
    pts = np.array(
        [
            (xf[b, i], yf[b, j], rf[b, j, i])
            for b in range(nbf)
            for j in range(yf.shape[1])
            for i in range(xf.shape[1])
        ]
    )
    x, y, r, dx, dy, nb = field(case)
    err = norm = 0.0
    for b in range(nb):
        for j in range(y.shape[1]):
            for i in range(x.shape[1]):
                m = (np.abs(pts[:, 0] - x[b, i]) < 0.5 * dx[b, i]) & (
                    np.abs(pts[:, 1] - y[b, j]) < 0.5 * dy[b, j]
                )
                q = pts[m, 2].mean()
                err += abs(r[b, j, i] - q) * dx[b, i] * dy[b, j]
                norm += abs(q) * dx[b, i] * dy[b, j]
    return err / norm


def analyze():
    ok = True
    e_amr, e_coarse = rel_l1("amr"), rel_l1("coarse")
    h, c = read_history(pid("amr"))
    levels = sorted(k for k in c if k.startswith("kinetics_blocks_per_level"))
    frac = np.mean(sum(h[:, c[k]] for k in levels)) * 16 / 1024.0
    ratio = walltime.get("amr", np.nan) / walltime.get("fine", np.nan)
    logger.info(
        f"rel L1 vs uniform fine: AMR {e_amr:.2e}, uniform coarse {e_coarse:.2e}; "
        f"AMR mean cell fraction {frac:.2f}, wall-time ratio AMR / fine {ratio:.2f}"
    )
    if not e_amr <= 1.0e-2:
        logger.warning(f"AMR rel L1 {e_amr:.2e} > 1e-2")
        ok = False
    if not h[:, c[levels[-1]]].max() > 0:
        logger.warning("AMR never reached the finest level")
        ok = False
    return ok
