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

# Regression test T-R6 (claude_sessions/kinetic_bgk/S1_DESIGN.md, step 6): multi-D
# tensor-train streaming by x1/x2/x3 sweeps (S1-Q5) agrees with the dense single-pass
# tensor-product SL. Deck inputs/kinetics/blast2d.py: a pressure pulse drifting
# diagonally, periodic, nu0 = 1e2, sl_dirk2.
#   blast2d   24 x 24 cells, Nv 20 x 20 x 14, t = 0.1, tt_eps = 1e-14, rank cap 24
#             (at 1e-12 the accumulated rounding error is 1.4e-10, see tt_accumulation)
#   blast3d   8^3 cells, Nv 14^3, drift (0.3, 0.2, 0.1), t = 0.04, tt_eps = 1e-12
# Pass: every derived moment of every dump within 1e-10 of dense (zero fields scaled by
# sound speed and pressure), history mass / energy / entropy within 1e-10, no rank-cap
# hits, all SVDs converged. Negative control: blast2d at tt_eps = 1e-8 must differ from
# dense by more than 1e-8 (the comparator sees TT error in 2D).

import glob
import logging

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs, read_history

logger = logging.getLogger("riot" + __name__[7:])

input_id = "kinetics/blast2d"
tol = 1.0e-10
three_d = [
    "parthenon/mesh/nx1=8",
    "parthenon/mesh/nx2=8",
    "parthenon/mesh/nx3=8",
    "parthenon/meshblock/nx1=8",
    "parthenon/meshblock/nx2=8",
    "parthenon/meshblock/nx3=8",
    "kinetics/nv1=14",
    "kinetics/nv2=14",
    "kinetics/nv3=14",
    "kinetics/v3min=-9.0",
    "kinetics/v3max=9.0",
    "parthenon/time/tlim=0.04",
    "parthenon/output1/dt=0.02",
    "region0/c_c_bulk_velocity=0.3,0.2,0.1",
    "region1/c_c_bulk_velocity=0.3,0.2,0.1",
]
cases = {
    "blast2d": ["kinetics/tt_rank_max=24", "kinetics/tt_eps=1.0e-14"],
    "blast3d": three_d + ["kinetics/tt_eps=1.0e-12"],
}


def pid(case, rep, tag=""):
    return f"kinetics_tr6_{case}_{rep}{tag}"


def run(**kwargs):
    ids = [pid(c, r) for c in cases for r in ("dense", "tt")]
    clean_outputs(*ids, pid("blast2d", "tt", "_loose"))
    riot.generate(input_id + ".py")
    for case, args in cases.items():
        for rep in ("dense", "tt"):
            riot.run(
                input_id + ".rin",
                [
                    "parthenon/job/problem_id=" + pid(case, rep),
                    "kinetics/representation=" + rep,
                ]
                + args,
            )
    riot.run(
        input_id + ".rin",
        [
            "parthenon/job/problem_id=" + pid("blast2d", "tt", "_loose"),
            "kinetics/representation=tt",
            "kinetics/tt_eps=1.0e-8",
        ],
    )


def moment_diff(d, t):
    worst = 0.0
    dumps = sorted(glob.glob(f"build/src/{d}.out1.*.phdf"))
    if len(dumps) < 2:
        return np.inf
    for fd in dumps:
        a, b = phdf(fd), phdf(fd.replace(d, t))
        p = np.asarray(a.Get("kinetics.pressure"))
        rho = np.asarray(a.Get("kinetics.rho"))
        cs = np.max(np.sqrt(p / rho))
        refs = {
            "kinetics.rho": np.max(rho),
            "kinetics.velocity": cs,
            "kinetics.temperature": np.max(np.asarray(a.Get("kinetics.temperature"))),
            "kinetics.pressure": np.max(p),
            "kinetics.stress": np.max(p),
            "kinetics.heat_flux": np.max(p) * cs,
        }
        for n, r in refs.items():
            x, y = np.asarray(a.Get(n)), np.asarray(b.Get(n))
            worst = max(worst, np.max(np.abs(x - y)) / r)
    return worst


def analyze():
    ok = True
    for case in cases:
        d, t = pid(case, "dense"), pid(case, "tt")
        e = moment_diff(d, t)
        hd, cd = read_history(d)
        ht, ct = read_history(t)
        eh = np.inf
        if hd.shape[0] == ht.shape[0]:
            eh = max(
                np.max(np.abs(hd[:, cd[n]] - ht[:, ct[n]]) / np.abs(hd[:, cd[n]]))
                for n in ("kinetics_sums_0", "kinetics_sums_4", "kinetics_sums_5")
            )
        caps = ht[:, ct["kinetics_tt_round_1"]].sum()
        svd = ht[:, ct["kinetics_tt_round_2"]].sum()
        logger.info(
            f"{case}: moments {e:.2e}, history {eh:.2e}, max rank "
            f"{ht[:, ct['kinetics_tt_max_rank']].max():.0f}, cap hits {caps:.0f}"
        )
        if e > tol or eh > tol or caps > 0 or svd > 0:
            logger.warning(
                f"{case}: tt differs from dense (moments {e:.2e}, history {eh:.2e}, "
                f"cap hits {caps}, svd failures {svd})"
            )
            ok = False
    e = moment_diff(pid("blast2d", "dense"), pid("blast2d", "tt", "_loose"))
    logger.info(f"negative control (tt_eps 1e-8): {e:.2e}")
    if not e > 1.0e-8:
        logger.warning(f"negative control: tt_eps = 1e-8 differs only by {e:.2e}")
        ok = False
    return ok
