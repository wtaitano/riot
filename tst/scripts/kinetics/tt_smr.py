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

# Tensor-train f on statically refined meshes (claude_sessions/kinetic_bgk/
# S4_DESIGN.md, step 2): fine ghosts copy the parent TT (injection), coarse ghosts take
# the rounded TT average of the children (kinetics_amr.cpp, RestrictTT). Cases, 1D,
# middle half of the mesh refined one level, sl_order 1:
#   wave      smooth entropy wave (freestream deck), periodic, nu0 = 0, uncoupled
#   sod       Sod deck, outflow, nu0 = 1e2, closure coupling
#   sod_lomac same with kinetics/lomac
# Pass, per case: every derived moment field of every dump within tol = 1e-10 of the
# dense run on the same mesh (tt_eps = 1e-14); history mass and energy within tol; no
# rank-cap hits or failed SVDs in streaming, relaxation or restriction; the restriction
# ran (its discarded-norm history column is nonzero). Instrument check: sod at
# tt_eps = 1e-8 is farther from dense than at 1e-14 by a factor >= 100.

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
smr = [
    "parthenon/mesh/refinement=static",
    "parthenon/mesh/numlevel=2",
    "parthenon/static_refinement0/x1min=0.25",
    "parthenon/static_refinement0/x1max=0.75",
    "parthenon/static_refinement0/level=1",
    "parthenon/mesh/nx1=64",
    "parthenon/meshblock/nx1=8",
    "kinetics/nv1=16",
    "kinetics/nv2=8",
    "kinetics/nv3=8",
    "kinetics/sl_order=1",
    "kinetics/min_vth_over_dv=0",
    "parthenon/output2/data_format=%.17e",
    "parthenon/output1/variables=" + ",".join(moment_vars),
]
sod = smr + [
    "parthenon/time/tlim=0.05",
    "parthenon/output1/dt=0.025",
    "kinetics/nu0=1.0e2",
    "kinetics/v1min=-8.0",
    "kinetics/v1max=8.0",
    "kinetics/v2min=-8.0",
    "kinetics/v2max=8.0",
    "kinetics/v3min=-8.0",
    "kinetics/v3max=8.0",
]
cases = {
    "wave": (
        "kinetics/freestream",
        smr
        + [
            "kinetics/closure_coupling=false",
            "kinetics/nu0=0.0",
            "kinetics/sl_limiter=none",
            "parthenon/time/tlim=0.25",
            "parthenon/output1/dt=0.125",
        ],
    ),
    "sod": ("kinetics/sod", sod),
    "sod_lomac": ("kinetics/sod", sod + ["kinetics/lomac=true"]),
}


def pid(case, rep, tag=""):
    return f"kinetics_tsmr_{case}_{rep}{tag}"


def run(**kwargs):
    ids = [pid(c, r) for c in cases for r in ("dense", "tt")] + [
        pid("sod", "tt", "_loose")
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
    deck, args = cases["sod"]
    riot.run(
        deck + ".rin",
        [
            "parthenon/job/problem_id=" + pid("sod", "tt", "_loose"),
            "kinetics/representation=tt",
            "kinetics/tt_diag=exact",
            "kinetics/tt_eps=1.0e-8",
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


def analyze():
    ok = True
    d = phdf(f"build/src/{pid('sod', 'dense')}.out1.00000.phdf")
    if len(set(np.asarray(d.level).tolist())) != 2:
        logger.warning(f"SMR mesh is not two-level: levels {d.level}")
        ok = False
    for case in cases:
        d, t = pid(case, "dense"), pid(case, "tt")
        e = moment_diff(d, t)
        hd, cd = read_history(d)
        ht, ct = read_history(t)
        eh = np.inf
        if hd.shape[0] == ht.shape[0]:
            eh = max(
                np.max(np.abs(hd[:, cd[n]] - ht[:, ct[n]]) / np.abs(hd[:, cd[n]]))
                for n in ("kinetics_sums_0", "kinetics_sums_4")
            )
        caps = sum(
            ht[:, ct[f"kinetics_tt_{k}_{a}"]].sum()
            for k in ("round", "amr_round")
            for a in (1, 2)
        )
        restricted = ht[:, ct["kinetics_tt_amr_round_0"]].sum()
        logger.info(
            f"{case}: moments {e:.2e}, history {eh:.2e}, max rank "
            f"{ht[:, ct['kinetics_tt_max_rank']].max():.0f}, cap/svd {caps:.0f}, "
            f"restriction discarded {restricted:.2e}"
        )
        if e > tol or eh > tol or caps > 0 or not restricted > 0.0:
            logger.warning(
                f"{case}: tt differs from dense (moments {e:.2e}, history {eh:.2e}, "
                f"cap/svd {caps}, restriction discarded {restricted:.2e})"
            )
            ok = False
    e_tight = moment_diff(pid("sod", "dense"), pid("sod", "tt"))
    e_loose = moment_diff(pid("sod", "dense"), pid("sod", "tt", "_loose"))
    logger.info(f"eps convergence: 1e-8 -> {e_loose:.2e}, 1e-14 -> {e_tight:.2e}")
    if not e_loose > 100.0 * e_tight:
        logger.warning(f"no convergence in tt_eps: {e_loose:.2e} vs {e_tight:.2e}")
        ok = False
    return ok
