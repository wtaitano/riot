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

# Regression test T-R4 (claude_sessions/kinetic_bgk/S1_DESIGN.md, step 4): 1D streaming
# and the kinetic boundary conditions in the tensor-train representation agree with the
# dense oracle. Cases (sl_dirk2, sl_order 1 unless noted):
#   sod_nu4     kinetics Sod deck, nu0 = 1e4, outflow, 64 cells, Nv 16^3, t = 0.05
#   sod_nu0     same, nu0 = 0 (free streaming)
#   wave_o2     smooth entropy wave (kinetics freestream deck), periodic, sl_order = 2
#               unlimited, nu0 = 1e2 (the Sod shock with unlimited quadratic SL drives one
#               cell to an equilibrium fallback in dense S0 at every resolution tried)
#   specular    closed box of the Sod deck, specular walls, nu0 = 1e4
#   diffuse     closed box, diffuse walls at T = 1.2 and 0.8, nu0 = 1
#   couette     kinetics Couette deck (diffuse moving walls, sl_order 2), t = 1
# Pass, per case: every derived moment field of every dump within tol of dense (zero
# fields scaled by the sound speed and pressure); history mass, energy and entropy within
# tol; no rank-cap hits, every SVD converged. tt_eps = 1e-14, tol = 1e-10.
# Convergence in eps: sod_nu4 at tt_eps = 1e-10 must be farther from dense than at 1e-14
# by at least a factor 100 (the comparator sees the TT error and it shrinks with eps).

import glob
import logging

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs, read_history

logger = logging.getLogger("riot" + __name__[7:])

tol = 1.0e-10
eps = 1.0e-14
moment_vars = [
    "kinetics.rho",
    "kinetics.velocity",
    "kinetics.temperature",
    "kinetics.pressure",
    "kinetics.stress",
    "kinetics.heat_flux",
]
small = [
    "parthenon/mesh/nx1=64",
    "parthenon/meshblock/nx1=16",
    "kinetics/nv1=16",
    "kinetics/nv2=16",
    "kinetics/nv3=16",
    "kinetics/min_vth_over_dv=0",
    "parthenon/output1/dt=0.025",
    "parthenon/output3/dt=-1",
    "parthenon/output2/data_format=%.17e",
]
sod = small + ["parthenon/time/tlim=0.05"]
box = small + [
    "parthenon/time/tlim=0.2",
    "parthenon/output1/dt=0.1",
    "kinetics/nv1=16",
    "kinetics/nv2=8",
    "kinetics/nv3=8",
]
cases = {
    "sod_nu4": ("kinetics/sod", sod + ["kinetics/nu0=1.0e4"]),
    "sod_nu0": ("kinetics/sod", sod + ["kinetics/nu0=0.0"]),
    "wave_o2": (
        "kinetics/freestream",
        [
            "kinetics/nu0=1.0e2",
            "kinetics/sl_limiter=none",
            "parthenon/time/tlim=0.25",
            "parthenon/output1/dt=0.125",
            "parthenon/output2/data_format=%.17e",
            "parthenon/output1/variables=" + ",".join(moment_vars),
        ],
    ),
    "specular": (
        "kinetics/sod",
        box
        + [
            "kinetics/ix1_bc=specular",
            "kinetics/ox1_bc=specular",
            "kinetics/nu0=1.0e4",
        ],
    ),
    "diffuse": (
        "kinetics/sod",
        box
        + [
            "kinetics/ix1_bc=diffuse",
            "kinetics/ox1_bc=diffuse",
            "kinetics/ix1_wall_T=1.2",
            "kinetics/ox1_wall_T=0.8",
            "kinetics/nu0=1.0",
        ],
    ),
    "couette": (
        "kinetics/couette",
        [
            "parthenon/time/tlim=1.0",
            "parthenon/output1/dt=0.5",
            "parthenon/output2/data_format=%.17e",
            "parthenon/output1/variables=" + ",".join(moment_vars),
        ],
    ),
}


def pid(case, rep, tag=""):
    return f"kinetics_tr4_{case}_{rep}{tag}"


def run(**kwargs):
    ids = [pid(c, r) for c in cases for r in ("dense", "tt")] + [
        pid("sod_nu4", "tt", "_loose")
    ]
    clean_outputs(*ids)
    for deck in sorted({d for d, _ in cases.values()}):
        riot.generate(deck + ".py")
    for case, (deck, args) in cases.items():
        for rep in ("dense", "tt"):
            riot.run(
                deck + ".rin",
                [
                    "parthenon/job/problem_id=" + pid(case, rep),
                    "kinetics/representation=" + rep,
                    "kinetics/tt_diag=exact",
                    f"kinetics/tt_eps={eps}",
                ]
                + args,
            )
    deck, args = cases["sod_nu4"]
    riot.run(
        deck + ".rin",
        [
            "parthenon/job/problem_id=" + pid("sod_nu4", "tt", "_loose"),
            "kinetics/representation=tt",
            "kinetics/tt_eps=1.0e-10",
        ]
        + args,
    )


def moment_diff(d, t):
    """Largest scaled difference of the derived moments over all dumps of the dense run."""
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
        caps = (
            ht[:, ct["kinetics_tt_round_1"]].sum()
            + ht[:, ct["kinetics_tt_round_2"]].sum()
        )
        logger.info(
            f"{case}: moments {e:.2e}, history {eh:.2e}, max rank "
            f"{ht[:, ct['kinetics_tt_max_rank']].max():.0f}, cap/svd {caps:.0f}"
        )
        if e > tol or eh > tol or caps > 0:
            logger.warning(
                f"{case}: tt differs from dense (moments {e:.2e}, history {eh:.2e}, cap/svd {caps})"
            )
            ok = False
    e_tight = moment_diff(pid("sod_nu4", "dense"), pid("sod_nu4", "tt"))
    e_loose = moment_diff(pid("sod_nu4", "dense"), pid("sod_nu4", "tt", "_loose"))
    logger.info(f"eps convergence: 1e-10 -> {e_loose:.2e}, 1e-14 -> {e_tight:.2e}")
    if not e_loose > 100.0 * e_tight:
        logger.warning(f"no convergence in tt_eps: {e_loose:.2e} vs {e_tight:.2e}")
        ok = False
    return ok
