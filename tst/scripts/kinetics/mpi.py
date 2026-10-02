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

# Regression test R5 (claude_sessions/kinetic_bgk/S0_DESIGN.md): the Sod deck on 1 and on
# 4 MPI ranks gives bitwise identical output arrays (every variable of every dump, and
# kinetics.f in the restart file). History sums are MPI reductions, whose summation
# order depends on the rank count, so they are only required to agree to 1e-13.

import logging

import numpy as np

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs, compare_bitwise

logger = logging.getLogger("riot" + __name__[7:])

input_id = "kinetics/sod"
ranks = (1, 4)
dumps = ["00000", "00001", "00002", "final"]


def pid(n):
    return f"kinetics_r5_np{n}"


def run(**kwargs):
    clean_outputs(*[pid(n) for n in ranks])
    riot.generate(input_id + ".py")
    for n in ranks:
        riot.mpirun(
            n,
            input_id + ".rin",
            [
                "parthenon/job/problem_id=" + pid(n),
                "kinetics/nv1=16",
                "kinetics/nv2=8",
                "kinetics/nv3=8",
                "kinetics/min_vth_over_dv=0",
                "parthenon/time/nlim=30",
                "parthenon/output1/dt=-1",
                "parthenon/output1/dn=10",
                "parthenon/output3/dt=-1",
                "parthenon/output3/dn=30",
                "parthenon/output2/data_format=%.17e",
            ],
        )


def analyze():
    ok = True
    a, b = pid(ranks[0]), pid(ranks[1])
    for d in dumps:
        bad = compare_bitwise(
            f"build/src/{a}.out1.{d}.phdf", f"build/src/{b}.out1.{d}.phdf", logger
        )
        if bad:
            logger.warning(
                f"dump {d}: {bad} differ between {ranks[0]} and {ranks[1]} ranks"
            )
            ok = False
    bad = compare_bitwise(
        f"build/src/{a}.out3.final.rhdf", f"build/src/{b}.out3.final.rhdf", logger
    )
    if bad:
        logger.warning(f"restart: {bad} differ")
        ok = False
    ha = np.loadtxt(f"build/src/{a}.out2.hst")
    hb = np.loadtxt(f"build/src/{b}.out2.hst")
    if ha.shape != hb.shape:
        logger.warning(
            f"history shapes {ha.shape} vs {hb.shape} (is mpiexec the right MPI?)"
        )
        return False
    rel = np.max(np.abs(ha - hb) / np.maximum(np.abs(ha), 1.0e-300))
    logger.debug(f"history max relative difference {rel:.2e}")
    if rel > 1.0e-13:
        logger.warning(f"history differs by {rel:.2e}")
        ok = False
    # Negative control: the comparator sees the evolution between the first and last dump.
    if not compare_bitwise(
        f"build/src/{a}.out1.00000.phdf",
        f"build/src/{a}.out1.final.phdf",
        logger,
        ["kinetics.rho"],
    ):
        logger.warning("comparator found no change between first and last dump")
        ok = False
    return ok
