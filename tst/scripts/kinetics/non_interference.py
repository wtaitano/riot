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

# Regression test R8 (claude_sessions/kinetic_bgk/S0_DESIGN.md): enabling the kinetics
# package must leave the hydro solution bitwise unchanged. Runs the kinetics Sod deck
# with physics/kinetics on and off and compares every hydro field in every dump. As a
# check that the comparator can fail, the first and last dumps of one run must differ.

import logging
import os

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs, compare_bitwise

logger = logging.getLogger("riot" + __name__[7:])

input_id = "kinetics/sod"
nlim = 40
hydro_vars = [
    "c.c.bulk.rho",
    "c.c.bulk.velocity",
    "c.c.bulk.pressure",
    "c.c.bulk.temperature",
    "c.c.bulk.momentum",
    "c.c.bulk.total_material_energy",
    "c.c.mat.rho_0",
]
dumps = ["00000", "00001", "00002", "final"]


def problem_id(kinetics):
    return "kinetics_r8_" + ("on" if kinetics else "off")


def run(**kwargs):
    clean_outputs(problem_id(True), problem_id(False))
    riot.generate(input_id + ".py")
    for kinetics in (True, False):
        riot.run(
            input_id + ".rin",
            [
                "parthenon/job/problem_id=" + problem_id(kinetics),
                "physics/kinetics=" + ("true" if kinetics else "false"),
                "parthenon/output1/variables="
                + ",".join(v for v in hydro_vars if not v.startswith("c.c.mat")),
                "parthenon/output3/dt=-1",
                f"parthenon/time/nlim={nlim}",
            ],
        )


def dump(kinetics, n):
    return f"build/src/{problem_id(kinetics)}.out1.{n}.phdf"


def analyze():
    names = [v for v in hydro_vars if not v.startswith("c.c.mat")]
    ok = True
    for n in dumps:
        if not (os.path.exists(dump(True, n)) and os.path.exists(dump(False, n))):
            logger.warning(f"missing dump {n}")
            ok = False
            continue
        differing = compare_bitwise(dump(True, n), dump(False, n), logger, names)
        if differing:
            logger.warning(f"dump {n}: hydro changed with kinetics on: {differing}")
            ok = False
    # Negative control: the comparator must see the evolution of the solution.
    if not compare_bitwise(dump(True, dumps[0]), dump(True, dumps[-1]), logger, names):
        logger.warning("comparator found no change between first and last dump")
        ok = False
    return ok
