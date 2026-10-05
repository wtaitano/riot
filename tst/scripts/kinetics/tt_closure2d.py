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

# S2 regression (claude_sessions/kinetic_bgk/S2_DESIGN.md, S2-Q15): 2D tensor-train
# streaming with the closure coupling on, on 2 MPI ranks. blast2d deck, 16^2 cells, cubic
# velocity box +-9.5 with 16^3 nodes (equal spacings, see the closure-coupling note in
# kinetics.rst), nu = 1e2, forced dt, tt_rank_max 24. Pass criteria, all on the final
# dump, max difference relative to each field's max (stress and heat flux relative to the
# max stress):
#   * TT at tt_eps = 1e-14 matches dense to 1e-10 in the kinetic rho, u, T, stress, heat
#     flux and the coupled hydro rho, u, T (measured 1.2e-12);
#   * control: TT at tt_eps = 1e-8 differs from dense by more than 1e-9 (measured 6.4e-7);
#   * instrument: the comparator sees the evolution (first vs final dense dump > 1e-2).

import logging

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs

logger = logging.getLogger("riot" + __name__[7:])

input_id = "kinetics/blast2d"
nranks = 2
fields = [
    "kinetics.rho",
    "kinetics.velocity",
    "kinetics.temperature",
    "kinetics.stress",
    "kinetics.heat_flux",
    "c.c.bulk.rho",
    "c.c.bulk.velocity",
    "c.c.bulk.temperature",
]
common = [
    "parthenon/mesh/nx1=16",
    "parthenon/mesh/nx2=16",
    "parthenon/meshblock/nx1=8",
    "parthenon/meshblock/nx2=8",
    "parthenon/output2/dt=-1",
    "parthenon/output3/dt=-1",
    "parthenon/time/dt_force=4e-3",
    "parthenon/output1/variables=" + ",".join(fields),
    "kinetics/tt_rank_max=24",
    "kinetics/tt_diag=exact",
] + [f"kinetics/nv{d}=16" for d in (1, 2, 3)]
common += [f"kinetics/v{d}min=-9.5" for d in (1, 2, 3)]
common += [f"kinetics/v{d}max=9.5" for d in (1, 2, 3)]
cases = {
    "dense": ["kinetics/representation=dense"],
    "tt14": ["kinetics/representation=tt", "kinetics/tt_eps=1e-14"],
    "tt8": ["kinetics/representation=tt", "kinetics/tt_eps=1e-8"],
}


def pid(case):
    return f"kinetics_s2_closure2d_{case}"


def run(**kwargs):
    clean_outputs(*[pid(c) for c in cases])
    riot.generate(input_id + ".py")
    for case, extra in cases.items():
        riot.mpirun(
            nranks,
            input_id + ".rin",
            ["parthenon/job/problem_id=" + pid(case)] + common + extra,
        )


def max_rel_diff(fa, fb):
    a, b = phdf(fa), phdf(fb)
    vel = np.max(np.abs(np.asarray(a.Get("kinetics.velocity"))))
    pscale = np.max(np.abs(np.asarray(a.Get("kinetics.stress"))))
    worst = 0.0
    for v in fields:
        x, y = np.asarray(a.Get(v)), np.asarray(b.Get(v))
        if v in ("kinetics.stress", "kinetics.heat_flux"):
            s = pscale
        elif v.endswith("velocity"):
            s = vel
        else:
            s = np.max(np.abs(x))
        d = np.max(np.abs(x - y)) / s
        logger.debug(f"{v}: {d:.3e}")
        worst = max(worst, d)
    return worst


def analyze():
    ok = True
    dense = f"build/src/{pid('dense')}.out1.final.phdf"
    evol = max_rel_diff(f"build/src/{pid('dense')}.out1.00000.phdf", dense)
    logger.debug(f"instrument: first vs final dense dump {evol:.3e}")
    if evol < 1.0e-2:
        logger.warning(f"comparator sees no evolution: {evol:.3e}")
        ok = False
    d14 = max_rel_diff(dense, f"build/src/{pid('tt14')}.out1.final.phdf")
    d8 = max_rel_diff(dense, f"build/src/{pid('tt8')}.out1.final.phdf")
    logger.debug(f"TT vs dense: eps 1e-14 {d14:.3e}, eps 1e-8 {d8:.3e}")
    if d14 > 1.0e-10:
        logger.warning(f"TT (eps 1e-14) vs dense {d14:.3e} > 1e-10")
        ok = False
    if d8 < 1.0e-9:
        logger.warning(f"control: TT (eps 1e-8) vs dense only {d8:.3e}")
        ok = False
    return ok
