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

# Adaptive mesh refinement with kinetics (claude_sessions/kinetic_bgk/S4_DESIGN.md, step
# 5). A smooth entropy wave streams through a periodic 1D mesh (freestream deck, 64
# cells, 8-cell blocks, two levels); the hydro density-gradient criterion
# (parthenon/refinement0) refines and derefines blocks as the wave moves. Each case runs
# dense and TT (tt_eps = 1e-14) on the same remeshing sequence:
#   constant   uncoupled, nu = 0, injection
#   linear     uncoupled, nu = 0, amr_prolong = linear, sl_order 2 (new fine blocks
#              reconstructed with slopes)
#   coupled    closure coupling, nu = 10
#   lomac      closure coupling + kinetics/lomac
# and noneq: the kinetic criterion alone (Sod deck, nu = 1e2, refine on
# ||f - M|| / ||f|| > 1e-2), dense only.
# Pass:
#   * every case refines and derefines (nbtotal rises and falls), and dense and TT have
#     the same block layout in every dump;
#   * TT within 1e-10 of dense in every derived moment of every dump (measured 3.6e-12
#     to 2.3e-11), no rank-cap hits or failed SVDs;
#   * hydro mass and energy conserved to 1e-12 through the remeshes (measured 8.7e-16);
#     with LoMaC the kinetic sums equal the hydro sums to 1e-12 at every step (2.8e-15);
#   * noneq: the kinetic criterion refines (the finest level appears) and the history
#     reports max ||f - M|| / ||f|| above the threshold (measured 0.11, 10 finest blocks).
# Instrument check: TT at tt_eps = 1e-8 (constant case) is farther from dense than at
# 1e-14 by >= 100.

import glob
import logging

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs, read_history

logger = logging.getLogger("riot" + __name__[7:])

tol = 1.0e-10
moment_vars = [
    "kinetics.rho",
    "kinetics.velocity",
    "kinetics.temperature",
    "kinetics.pressure",
    "kinetics.stress",
    "kinetics.heat_flux",
]
amr = [
    "parthenon/mesh/refinement=adaptive",
    "parthenon/mesh/numlevel=2",
    "parthenon/mesh/derefine_count=2",
    "parthenon/refinement0/method=derivative_order_1",
    "parthenon/refinement0/field=c.c.bulk.rho",
    "kinetics/min_vth_over_dv=0",
    "kinetics/tt_diag=exact",
    "kinetics/tt_rank_max=8",
    "kinetics/nv1=16",
    "kinetics/nv2=8",
    "kinetics/nv3=8",
    "parthenon/output2/dt=-1",
    "parthenon/output2/dn=1",
    "parthenon/output2/data_format=%.17e",
    "parthenon/output1/variables=" + ",".join(moment_vars),
]
wave = amr + [
    "parthenon/mesh/nx1=64",
    "parthenon/meshblock/nx1=8",
    "parthenon/refinement0/refine_tol=0.012",
    "parthenon/refinement0/derefine_tol=0.008",
    "kinetics/amr_noneq_refine=1e30",
    "kinetics/amr_noneq_derefine=1e30",
    "kinetics/sl_limiter=none",
]
uncoupled = ["kinetics/closure_coupling=false", "kinetics/nu0=0.0"]
coupled = ["kinetics/closure_coupling=true", "kinetics/nu0=10.0"]
cases = {
    "constant": wave + uncoupled + ["kinetics/sl_order=1"],
    "linear": wave + uncoupled + ["kinetics/sl_order=2", "kinetics/amr_prolong=linear"],
    "coupled": wave + coupled + ["kinetics/sl_order=1"],
    "lomac": wave + coupled + ["kinetics/sl_order=1", "kinetics/lomac=true"],
}
noneq = (
    amr
    + [
        "parthenon/mesh/nx1=64",
        "parthenon/meshblock/nx1=8",
        "parthenon/mesh/numlevel=3",
        "parthenon/refinement0/refine_tol=1e30",
        "parthenon/refinement0/derefine_tol=1e30",
        "kinetics/amr_noneq_refine=1e-2",
        "kinetics/amr_noneq_derefine=1e-3",
        "kinetics/nu0=1.0e2",
        "parthenon/time/tlim=0.05",
        "parthenon/output1/dt=0.025",
        "parthenon/output3/dt=-1",
    ]
    + [
        f"kinetics/v{d}{e}=8.0" if e == "max" else f"kinetics/v{d}{e}=-8.0"
        for d in (1, 2, 3)
        for e in ("min", "max")
    ]
)


def pid(case, rep, tag=""):
    return f"kinetics_tamr_{case}_{rep}{tag}"


def run(**kwargs):
    ids = [pid(c, r) for c in cases for r in ("dense", "tt")]
    ids += [pid("constant", "tt", "_loose"), pid("noneq", "dense")]
    clean_outputs(*ids)
    riot.generate("kinetics/freestream.py")
    riot.generate("kinetics/sod.py")
    for case, args in cases.items():
        for rep in ("dense", "tt"):
            riot.run(
                "kinetics/freestream.rin",
                [
                    "parthenon/job/problem_id=" + pid(case, rep),
                    "kinetics/representation=" + rep,
                    "kinetics/tt_eps=1e-14",
                ]
                + args,
            )
    riot.run(
        "kinetics/freestream.rin",
        [
            "parthenon/job/problem_id=" + pid("constant", "tt", "_loose"),
            "kinetics/representation=tt",
            "kinetics/tt_eps=1e-8",
        ]
        + cases["constant"],
    )
    riot.run(
        "kinetics/sod.rin",
        ["parthenon/job/problem_id=" + pid("noneq", "dense")] + noneq,
    )


def moment_diff(d, t):
    """Largest scaled moment difference over all dumps; inf if the layouts differ."""
    worst = 0.0
    dumps = sorted(glob.glob(f"build/src/{d}.out1.*.phdf"))
    if len(dumps) < 2:
        return np.inf
    for fd in dumps:
        a, b = phdf(fd), phdf(fd.replace(d, t))
        if not np.array_equal(np.asarray(a.level), np.asarray(b.level)):
            return np.inf
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


def drift(h, c):
    return np.max(np.abs(h[:, c] - h[0, c])) / np.abs(h[0, c])


def analyze():
    ok = True
    for case in cases:
        d, t = pid(case, "dense"), pid(case, "tt")
        hd, cd = read_history(d)
        ht, ct = read_history(t)
        nb = hd[:, cd["nbtotal"]]
        rises = int(np.sum(np.diff(nb) > 0))
        falls = int(np.sum(np.diff(nb) < 0))
        e = moment_diff(d, t)
        caps = sum(
            ht[:, ct[f"kinetics_tt_{k}_{a}"]].sum()
            for k in ("round", "amr_round")
            for a in (1, 2)
        )
        hyd = max(
            drift(h, c[n])
            for h, c in ((hd, cd), (ht, ct))
            for n in ("kinetics_sums_7", "kinetics_sums_11")
        )
        logger.info(
            f"{case}: remesh +{rises}/-{falls}, TT vs dense {e:.2e}, cap/svd {caps:.0f}, "
            f"hydro drift {hyd:.2e}"
        )
        if rises == 0 or falls == 0:
            logger.warning(f"{case}: mesh did not both refine and derefine")
            ok = False
        if e > tol or caps > 0 or hyd > 1.0e-12:
            logger.warning(
                f"{case}: TT vs dense {e:.2e}, cap/svd {caps}, hydro drift {hyd:.2e}"
            )
            ok = False
    for rep in ("dense", "tt"):
        h, c = read_history(pid("lomac", rep))
        gap = max(
            np.max(
                np.abs(h[:, c[f"kinetics_sums_{k}"]] - h[:, c[f"kinetics_sums_{y}"]])
            )
            / np.max(np.abs(h[:, c[f"kinetics_sums_{y}"]]))
            for k, y in ((0, 7), (1, 8), (4, 11))
        )
        logger.info(f"lomac {rep}: kinetic vs hydro sums {gap:.2e}")
        if gap > 1.0e-12:
            logger.warning(f"lomac {rep}: kinetic vs hydro sums {gap:.2e}")
            ok = False
    e_tight = moment_diff(pid("constant", "dense"), pid("constant", "tt"))
    e_loose = moment_diff(pid("constant", "dense"), pid("constant", "tt", "_loose"))
    logger.info(f"eps convergence: 1e-8 -> {e_loose:.2e}, 1e-14 -> {e_tight:.2e}")
    if not e_loose > 100.0 * e_tight:
        logger.warning(f"no convergence in tt_eps: {e_loose:.2e} vs {e_tight:.2e}")
        ok = False
    h, c = read_history(pid("noneq", "dense"))
    finest = c["kinetics_blocks_per_level_2"]
    qmax = h[:, c["kinetics_noneq_max"]].max()
    logger.info(
        f"noneq: max ||f - M|| / ||f|| {qmax:.2e}, finest blocks {h[-1, finest]}"
    )
    if not (h[:, finest].max() > 0 and qmax > 1.0e-2):
        logger.warning(f"noneq: kinetic criterion did not refine (max {qmax:.2e})")
        ok = False
    return ok
