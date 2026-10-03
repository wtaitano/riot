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

# Regression test T-R5 (claude_sessions/kinetic_bgk/S1_DESIGN.md, step 5, S1-Q12):
# accumulation of the TT rounding error with the number of roundings. Smooth entropy wave
# (kinetics freestream deck, periodic, sl_order 2 unlimited, nu0 = 1e2, sl_dirk2), fixed
# final time, forced time steps of 1/16, 1/64 and 1/256 of it, tt_eps in {1e-8, 1e-12}.
# Each substep does 3 roundings per stage (2 for streaming, 1 for relaxation) and 2
# stages. Error = largest difference of the derived moments from the dense run with the
# same steps (scaled by the moment's max, heat flux by the pressure). Pass:
#   * bound: error <= C N eps with N the number of roundings and C = 1 (measured 0.17-0.81:
#     the per-rounding errors are not all aligned);
#   * the TT error is real: error >= 1e-3 N eps at every point, and the error at
#     eps = 1e-8 exceeds the error at 1e-12 by at least 1e3 for every step count;
#   * the reported discarded norms (kinetics_tt_round_0) add up to within a factor 10 of
#     the measured error at the finest step count (they bound it per step, and the sum
#     over steps tracks the accumulated error).

import logging

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs, read_history

logger = logging.getLogger("riot" + __name__[7:])

input_id = "kinetics/freestream"
tlim = 0.25
steps = [16, 64, 256]
epss = [1.0e-8, 1.0e-12]
moment_vars = [
    "kinetics.rho",
    "kinetics.pressure",
    "kinetics.stress",
    "kinetics.heat_flux",
]
common = [
    "kinetics/nu0=1.0e2",
    "kinetics/sl_limiter=none",
    "kinetics/nv1=16",
    "kinetics/nv2=12",
    "kinetics/nv3=12",
    "kinetics/min_vth_over_dv=0",
    "parthenon/mesh/nx1=32",
    "parthenon/meshblock/nx1=8",
    f"parthenon/time/tlim={tlim}",
    f"parthenon/output1/dt={tlim}",
    "parthenon/output1/variables=" + ",".join(moment_vars),
    "parthenon/output2/data_format=%.17e",
]


def pid(rep, nsteps, eps=None):
    tag = "" if eps is None else f"_e{-int(round(np.log10(eps)))}"
    return f"kinetics_tr5_{rep}_s{nsteps}{tag}"


def run(**kwargs):
    ids = [pid("dense", s) for s in steps] + [
        pid("tt", s, e) for s in steps for e in epss
    ]
    clean_outputs(*ids)
    riot.generate(input_id + ".py")
    for s in steps:
        force = [f"parthenon/time/dt_force={tlim / s}"]
        riot.run(
            input_id + ".rin",
            ["parthenon/job/problem_id=" + pid("dense", s)] + common + force,
        )
        for e in epss:
            riot.run(
                input_id + ".rin",
                [
                    "parthenon/job/problem_id=" + pid("tt", s, e),
                    "kinetics/representation=tt",
                    f"kinetics/tt_eps={e}",
                ]
                + common
                + force,
            )


def error(d, t):
    a = phdf(f"build/src/{d}.out1.final.phdf")
    b = phdf(f"build/src/{t}.out1.final.phdf")
    p = np.max(np.asarray(a.Get("kinetics.pressure")))
    e = 0.0
    for n in moment_vars:
        x, y = np.asarray(a.Get(n)), np.asarray(b.Get(n))
        sc = p if n == "kinetics.heat_flux" else np.max(np.abs(x))
        e = max(e, np.max(np.abs(x - y)) / sc)
    return e


def analyze():
    ok = True
    err = {}
    for s in steps:
        for e in epss:
            t = pid("tt", s, e)
            h, c = read_history(t)
            nsub = h[1:, c["kinetics_substeps"]].max()
            nround = s * nsub * 2 * 3
            err[s, e] = error(pid("dense", s), t)
            ratio = err[s, e] / (nround * e)
            disc = h[:, c["kinetics_tt_round_0"]].sum() / h[0, c["kinetics_tt_ranks_0"]]
            logger.info(
                f"eps {e:.0e}, {s} steps, {nround:.0f} roundings: error {err[s, e]:.3e}, "
                f"error / (N eps) {ratio:.3f}, summed discarded {disc:.3e}"
            )
            if ratio > 1.0 or ratio < 1.0e-3:
                logger.warning(f"{t}: error / (N eps) = {ratio:.3e} outside [1e-3, 1]")
                ok = False
            if s == steps[-1] and not (0.1 * err[s, e] <= disc <= 10.0 * err[s, e]):
                logger.warning(
                    f"{t}: summed discarded {disc:.3e} vs error {err[s, e]:.3e}"
                )
                ok = False
    for s in steps:
        if not err[s, epss[0]] > 1.0e3 * err[s, epss[1]]:
            logger.warning(f"{s} steps: error does not shrink with eps")
            ok = False
    return ok
