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

# Regression test T-R8 (claude_sessions/kinetic_bgk/S1_DESIGN.md, step 8): nonlinear
# diagnostics of a tensor-train f by cross approximation (tt_diag = cross,
# src/kinetics/tt_cross.hpp) agree with decompression (tt_diag = exact). The same TT run
# is made twice, once per tt_diag; the diagnostics do not feed back, so f is identical
# and only the history columns differ. Cases: Sod nu0 = 1e4 and 1e1 (128 cells, Nv 32^3,
# tt_eps 1e-8), Couette (diffuse moving walls), blast2d (2D, 24 x 24).
# Pass:
#   * f-derived history columns (mass, energy, negative mass) are bitwise equal;
#   * entropy (cross estimate of the regularized f ln f, delta = 1e-12 max f) within
#     1e-9 relative of the exact entropy at every dump;
#   * cross min f is a sampled upper bound: exact <= cross <= exact + 1e-9 max f (min f
#     is rounding noise here, |min f| ~ 1e-11 max f; max f from the final restart);
#   * every cross converged, no rank-cap hit; on Sod (Nv 32^3) the phi evaluations per
#     cell are below the number of velocity nodes (cheaper than decompression). At the
#     small Nv of couette (6,912) and blast2d (5,600) the cross is not cheaper (measured
#     0.8x and 1.25x the node count); only logged.
# Negative control: a cross with tt_cross_rank_max = 1 on Sod nu0 = 1e1 must miss the
# entropy by more than 1e-6 and report cap hits.

import logging

import numpy as np

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs, read_history, tt_decompress

logger = logging.getLogger("riot" + __name__[7:])

ent_tol = 1.0e-9
minf_tol = 1.0e-9
sod = ["parthenon/time/tlim=0.1", "parthenon/output2/data_format=%.17e"]
cases = {
    "sod_nu4": ("kinetics/sod", sod + ["kinetics/nu0=1.0e4"]),
    "sod_nu1": ("kinetics/sod", sod + ["kinetics/nu0=1.0e1"]),
    "couette": (
        "kinetics/couette",
        [
            "parthenon/time/tlim=1.0",
            "parthenon/output2/data_format=%.17e",
            "parthenon/output3/file_type=rst",
            "parthenon/output3/dt=1.0",
            "parthenon/output3/write_xdmf=false",
            "parthenon/output3/hdf5_compression_level=0",
        ],
    ),
    "blast2d": (
        "kinetics/blast2d",
        ["parthenon/time/tlim=0.05", "parthenon/output2/data_format=%.17e"],
    ),
}


def pid(case, diag):
    return f"kinetics_tr8_{case}_{diag}"


def run(**kwargs):
    ids = [pid(c, d) for c in cases for d in ("exact", "cross")] + [
        pid("sod_nu1", "cap1")
    ]
    clean_outputs(*ids)
    for deck in sorted({d for d, _ in cases.values()}):
        riot.generate(deck + ".py")
    for case, (deck, args) in cases.items():
        for diag in ("exact", "cross"):
            riot.run(
                deck + ".rin",
                [
                    "parthenon/job/problem_id=" + pid(case, diag),
                    "kinetics/representation=tt",
                    "kinetics/tt_diag=" + diag,
                ]
                + args,
            )
    deck, args = cases["sod_nu1"]
    riot.run(
        deck + ".rin",
        [
            "parthenon/job/problem_id=" + pid("sod_nu1", "cap1"),
            "kinetics/representation=tt",
            "kinetics/tt_diag=cross",
            "kinetics/tt_cross_rank_max=1",
        ]
        + args,
    )


def max_f_nodes(case):
    """Largest f over the final restart file (from the cores) and the node count."""
    import h5py

    fname = f"build/src/{pid(case, 'exact')}.out3.final.rhdf"
    with h5py.File(fname, "r") as h:
        layout = h["Params"].attrs["kinetics/f_layout"]
    nv = tuple(int(x) for x in layout[1:4])
    f, _ = tt_decompress(fname, nv, int(layout[4]))
    return np.max(f), np.prod(nv)


def analyze():
    ok = True
    for case in cases:
        he, ce = read_history(pid(case, "exact"))
        hc, cc = read_history(pid(case, "cross"))
        if he.shape[0] != hc.shape[0]:
            logger.warning(f"{case}: history shapes differ {he.shape} {hc.shape}")
            ok = False
            continue
        for n in ("kinetics_sums_0", "kinetics_sums_4", "kinetics_sums_6"):
            if not np.array_equal(he[:, ce[n]], hc[:, cc[n]]):
                logger.warning(f"{case}: {n} not bitwise equal (f differs?)")
                ok = False
        x, y = he[:, ce["kinetics_sums_5"]], hc[:, cc["kinetics_sums_5"]]
        e_ent = np.max(np.abs(x - y) / np.abs(x))
        fm, nnodes = max_f_nodes(case)
        m_ex, m_cr = he[:, ce["kinetics_min_f"]], hc[:, cc["kinetics_min_f"]]
        gap = np.max(m_cr - m_ex) / fm
        below = np.any(m_cr < m_ex - 1.0e-15 * fm)
        ncell = hc[0, cc["kinetics_tt_ranks_0"]]
        evals = np.max(hc[:, cc["kinetics_tt_cross_0"]]) / ncell
        caps = np.sum(hc[:, cc["kinetics_tt_cross_2"]])
        nconv = np.sum(hc[:, cc["kinetics_tt_cross_3"]])
        rank = np.max(hc[:, cc["kinetics_tt_cross_1"]] / ncell)
        logger.info(
            f"{case}: entropy rel {e_ent:.2e}, min f gap {gap:.2e} max f, evals/cell "
            f"{evals:.0f} of {nnodes}, mean g rank "
            f"{rank:.1f}, cap hits {caps:.0f}, not converged {nconv:.0f}"
        )
        if e_ent > ent_tol or gap > minf_tol or below or caps > 0 or nconv > 0:
            logger.warning(f"{case}: cross diagnostics out of tolerance")
            ok = False
        if case.startswith("sod") and evals >= nnodes:
            logger.warning(f"{case}: cross costs more than decompression ({evals:.0f})")
            ok = False

    he, ce = read_history(pid("sod_nu1", "exact"))
    hc, cc = read_history(pid("sod_nu1", "cap1"))
    x, y = he[:, ce["kinetics_sums_5"]], hc[:, cc["kinetics_sums_5"]]
    e_neg = np.max(np.abs(x - y) / np.abs(x))
    caps = np.sum(hc[:, cc["kinetics_tt_cross_2"]])
    logger.info(
        f"negative control (rank cap 1): entropy rel {e_neg:.2e}, caps {caps:.0f}"
    )
    if not (e_neg > 1.0e-6 and caps > 0):
        logger.warning("negative control: a rank-1 cross was not detected")
        ok = False
    return ok
