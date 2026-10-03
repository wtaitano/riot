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

# Regression test T-R2 (claude_sessions/kinetic_bgk/S1_DESIGN.md, step 2): the
# tensor-train representation is initialized exactly. The kinetics Sod deck is run for
# zero cycles (initialization and the first outputs only) with representation = dense
# and = tt, for init = equilibrium (TT rank 1) and two_maxwellian (rank 2). Pass:
#   * the decompressed TT restart f equals the dense f node by node to 1e-13
#     (relative to max f), and every cell has the expected ranks;
#   * the derived moment fields agree to 1e-12 (zero fields scaled by the sound speed
#     and pressure of the left state);
#   * history: kinetic mass, energy, entropy and min f agree to 1e-12, and the TT rank
#     columns report the expected ranks;
#   * two_maxwellian has the hydro moments: kinetic mass and energy equal the hydro ones
#     to 1e-12.
# Negative control: the dense equilibrium f against the TT two_maxwellian f must differ.

import logging

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import (
    clean_outputs,
    read_history,
    tt_decompress,
)

logger = logging.getLogger("riot" + __name__[7:])

input_id = "kinetics/sod"
nv = (32, 32, 32)
rcap = 16
inits = {"equilibrium": 1, "two_maxwellian": 2}
moment_vars = [
    "kinetics.rho",
    "kinetics.velocity",
    "kinetics.temperature",
    "kinetics.pressure",
    "kinetics.stress",
    "kinetics.heat_flux",
]


def pid(rep, init):
    return f"kinetics_tr2_{rep}_{init}"


def run(**kwargs):
    clean_outputs(*[pid(r, i) for r in ("dense", "tt") for i in inits])
    riot.generate(input_id + ".py")
    for rep in ("dense", "tt"):
        for init in inits:
            riot.run(
                input_id + ".rin",
                [
                    "parthenon/job/problem_id=" + pid(rep, init),
                    "kinetics/representation=" + rep,
                    "kinetics/init=" + init,
                    f"kinetics/tt_rank_max={rcap}",
                    "parthenon/time/nlim=0",
                    "parthenon/output2/data_format=%.17e",
                ],
            )


def restart(p):
    return f"build/src/{p}.out3.00000.rhdf"


def dump(p):
    return f"build/src/{p}.out1.00000.phdf"


def dense_f(p):
    import h5py

    with h5py.File(restart(p), "r") as h:
        return np.array(h["kinetics.f"])


def analyze():
    ok = True
    for init, rank in inits.items():
        fd = dense_f(pid("dense", init))
        ft, ranks = tt_decompress(restart(pid("tt", init)), nv, rcap)
        err = np.max(np.abs(fd - ft)) / np.max(np.abs(fd))
        logger.info(f"{init}: f max rel diff {err:.3e}, ranks {sorted(set(ranks))}")
        if err > 1.0e-13:
            logger.warning(
                f"{init}: decompressed TT f differs from dense f ({err:.3e})"
            )
            ok = False
        if set(ranks) != {(rank, rank)}:
            logger.warning(f"{init}: ranks {sorted(set(ranks))}, expected {rank}")
            ok = False

        a, b = phdf(dump(pid("dense", init))), phdf(dump(pid("tt", init)))
        p = np.asarray(a.Get("kinetics.pressure"))
        cs = np.max(np.sqrt(p / np.asarray(a.Get("kinetics.rho"))))
        ref = {
            "kinetics.velocity": cs,
            "kinetics.heat_flux": np.max(p) * cs,
            "kinetics.stress": np.max(p),
        }
        for n in moment_vars:
            x, y = np.asarray(a.Get(n)), np.asarray(b.Get(n))
            e = np.max(np.abs(x - y)) / max(np.max(np.abs(x)), ref.get(n, 0.0))
            if e > 1.0e-12:
                logger.warning(f"{init}: {n} differs, rel {e:.3e}")
                ok = False

        hd, cd = read_history(pid("dense", init))
        ht, ct = read_history(pid("tt", init))
        for n in [
            "kinetics_sums_0",
            "kinetics_sums_4",
            "kinetics_sums_5",
            "kinetics_min_f",
        ]:
            x, y = hd[0, cd[n]], ht[0, ct[n]]
            if abs(x - y) > 1.0e-12 * abs(x):
                logger.warning(f"{init}: history {n} dense {x:.17e} tt {y:.17e}")
                ok = False
        ncell = ht[0, ct["kinetics_tt_ranks_0"]]
        for c in ("kinetics_tt_ranks_1", "kinetics_tt_ranks_2"):
            if ht[0, ct[c]] != rank * ncell:
                logger.warning(f"{init}: {c} = {ht[0, ct[c]]}, expected {rank * ncell}")
                ok = False
        if (
            ht[0, ct["kinetics_tt_max_rank"]] != rank
            or ht[0, ct["kinetics_tt_ranks_3"]]
        ):
            logger.warning(f"{init}: max rank / cap-hit columns wrong")
            ok = False
        for kin, hyd in (
            ("kinetics_sums_0", "kinetics_sums_7"),
            ("kinetics_sums_4", "kinetics_sums_11"),
        ):
            x, y = ht[0, ct[kin]], ht[0, ct[hyd]]
            if abs(x - y) > 1.0e-12 * abs(y):
                logger.warning(f"{init}: {kin} {x:.17e} vs hydro {hyd} {y:.17e}")
                ok = False

    # Negative control: the comparator must see a different distribution.
    fd = dense_f(pid("dense", "equilibrium"))
    ft, _ = tt_decompress(restart(pid("tt", "two_maxwellian")), nv, rcap)
    if np.max(np.abs(fd - ft)) / np.max(np.abs(fd)) <= 1.0e-13:
        logger.warning(
            "negative control: equilibrium and two_maxwellian f do not differ"
        )
        ok = False
    return ok
