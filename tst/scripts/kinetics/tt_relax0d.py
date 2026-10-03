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

# Regression test T-R3 (claude_sessions/kinetic_bgk/S1_DESIGN.md, step 3): BGK
# relaxation in the tensor-train representation agrees with the dense oracle. The 0D
# relaxation deck runs with kinetics/streaming = false (collisions only, same substeps)
# for init = bimaxwellian and two_maxwellian, integrators sl_dirk2 and strang, in both
# representations. Pass:
#   * dense with streaming off equals dense with streaming on to 1e-13 in every history
#     column used below (streaming of uniform periodic data is the identity to
#     roundoff), so the oracle is the one checked by kinetics/relax0d;
#   * tt vs dense, every history row: mass, energy and entropy to 1e-12 (relative),
#     momentum to 1e-12 of the mass; the anisotropy P_xx - P_yy of every dump to 1e-12
#     of the pressure;
#   * tt ranks: max rank 2 (f stays in span{f_0, M}, rank 2 for both starts), no cap
#     hits, every SVD converged, discarded norm per hydro step <= 1e-12;
#   * TT rounding noise is small: min f >= -1e-12 max f and |negative mass| <= 1e-12 mass.
# Negative control: tt with tt_rank_max = 2 and tt_eps = 0.5 (rounding to rank 1 is
# allowed) must fail the anisotropy comparison.

import glob
import logging

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs, read_history

logger = logging.getLogger("riot" + __name__[7:])

input_id = "kinetics/relax0d"
inits = ("bimaxwellian", "two_maxwellian")
integrators = ("sl_dirk2", "strang")
tol = 1.0e-12


def pid(rep, init, integ, tag=""):
    return f"kinetics_tr3_{rep}_{init}_{integ}{tag}"


def all_ids():
    ids = [pid(r, i, g) for r in ("dense", "tt") for i in inits for g in integrators]
    ids += [pid("dense", i, g, "_stream") for i in inits for g in integrators]
    return ids + [pid("tt", inits[0], integrators[0], "_ctrl")]


def run_one(problem_id, rep, init, integ, streaming, extra=()):
    riot.run(
        input_id + ".rin",
        [
            "parthenon/job/problem_id=" + problem_id,
            "kinetics/representation=" + rep,
            "kinetics/init=" + init,
            "kinetics/integrator=" + integ,
            "kinetics/streaming=" + ("true" if streaming else "false"),
            "parthenon/output1/variables=kinetics.rho,kinetics.stress",
        ]
        + list(extra),
    )


def run(**kwargs):
    clean_outputs(*all_ids())
    riot.generate(input_id + ".py")
    for init in inits:
        for integ in integrators:
            run_one(pid("dense", init, integ, "_stream"), "dense", init, integ, True)
            for rep in ("dense", "tt"):
                run_one(pid(rep, init, integ), rep, init, integ, False)
    run_one(
        pid("tt", inits[0], integrators[0], "_ctrl"),
        "tt",
        inits[0],
        integrators[0],
        False,
        ["kinetics/tt_rank_max=2", "kinetics/tt_eps=0.5"],
    )


def anisotropy(problem_id):
    """P_xx - P_yy per cell in every dump, divided by the pressure."""
    out = []
    for f in sorted(glob.glob(f"build/src/{problem_id}.out1.*.phdf")):
        s = np.asarray(phdf(f).Get("kinetics.stress", flatten=False))
        s = np.moveaxis(s, 1, -1).reshape(-1, 6)
        out.append((s[:, 0] - s[:, 1]) / np.mean(s[:, :3]))
    return np.array(out)


def history_diff(a, b):
    """Largest relative difference of mass, momentum, energy and entropy over all rows."""
    ha, ca = read_history(a)
    hb, cb = read_history(b)
    if ha.shape[0] != hb.shape[0]:
        return np.inf
    mass = np.abs(ha[:, ca["kinetics_sums_0"]])
    e = 0.0
    for n in ("kinetics_sums_0", "kinetics_sums_4", "kinetics_sums_5"):
        x, y = ha[:, ca[n]], hb[:, cb[n]]
        e = max(e, np.max(np.abs(x - y) / np.abs(x)))
    for n in ("kinetics_sums_1", "kinetics_sums_2", "kinetics_sums_3"):
        e = max(e, np.max(np.abs(ha[:, ca[n]] - hb[:, cb[n]]) / mass))
    return e


def analyze():
    ok = True
    for init in inits:
        for integ in integrators:
            d, t = pid("dense", init, integ), pid("tt", init, integ)
            e = history_diff(pid("dense", init, integ, "_stream"), d)
            if e > 1.0e-13:
                logger.warning(f"{d}: streaming on/off differ by {e:.3e}")
                ok = False
            e = history_diff(d, t)
            logger.info(f"{t}: history rel diff vs dense {e:.3e}")
            if e > tol:
                logger.warning(f"{t}: history differs from dense by {e:.3e}")
                ok = False
            ad, at = anisotropy(d), anisotropy(t)
            if ad.shape != at.shape or ad.shape[0] < 3:
                logger.warning(f"{t}: dumps missing ({ad.shape} vs {at.shape})")
                ok = False
                continue
            e = np.max(np.abs(ad - at))
            if e > tol:
                logger.warning(f"{t}: anisotropy differs from dense by {e:.3e}")
                ok = False
            h, c = read_history(t)
            if np.max(h[:, c["kinetics_tt_max_rank"]]) != 2:
                logger.warning(
                    f"{t}: max rank {np.max(h[:, c['kinetics_tt_max_rank']])}"
                )
                ok = False
            if np.any(h[:, c["kinetics_tt_round_1"]]) or np.any(
                h[:, c["kinetics_tt_round_2"]]
            ):
                logger.warning(f"{t}: rank-cap hits or non-converged SVDs")
                ok = False
            ncell = h[0, c["kinetics_tt_ranks_0"]]
            if np.max(h[:, c["kinetics_tt_round_0"]]) / ncell > tol:
                logger.warning(
                    f"{t}: discarded norm {np.max(h[:, c['kinetics_tt_round_0']]):.3e}"
                )
                ok = False
            mass = h[0, c["kinetics_sums_0"]]
            minf = np.min(h[:, c["kinetics_min_f"]])
            negm = np.min(h[:, c["kinetics_sums_6"]])
            fmax = np.max(_fmax(t))
            logger.info(
                f"{t}: min f / max f {minf / fmax:.3e}, neg mass {negm / mass:.3e}"
            )
            if minf < -1.0e-12 * fmax or abs(negm) > 1.0e-12 * mass:
                logger.warning(
                    f"{t}: min f {minf:.3e} (max f {fmax:.3e}), neg mass {negm:.3e}"
                )
                ok = False

    # Negative control: rank-1 rounding must break the anisotropy history.
    ctrl = pid("tt", inits[0], integrators[0], "_ctrl")
    ad, ac = anisotropy(pid("dense", inits[0], integrators[0])), anisotropy(ctrl)
    if ad.shape == ac.shape and np.max(np.abs(ad - ac)) <= tol:
        logger.warning("negative control passed the anisotropy comparison")
        ok = False
    return ok


def _fmax(problem_id):
    """Scale of max f per cell, n / (2 pi theta_min)^(3/2) with the smallest per-axis
    temperature of the initial state (the peak of a Maxwellian of that temperature)."""
    d = phdf(f"build/src/{problem_id}.out1.00000.phdf")
    s = np.asarray(d.Get("kinetics.stress", flatten=False))
    s = np.moveaxis(s, 1, -1).reshape(-1, 6)
    rho = np.asarray(d.Get("kinetics.rho", flatten=False)).ravel()
    m = 1.380649e-16  # particle mass [g]: the deck has k_B / m = 1
    th = np.min(s[:, :3], axis=1) / rho
    return rho / m / (2.0 * np.pi * th) ** 1.5
