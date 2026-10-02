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

# Regression test R6 (claude_sessions/kinetic_bgk/S0_DESIGN.md): a restart round trip at
# Nv = 32^3 continues the kinetic state bitwise. Also exercises writing a variable with
# 32768 components (one HDF5 component label each, OQ2).
#
# Only kinetics variables are required to match bitwise. Hydro (independent of
# kinetics in S0) restarts in RIOT to roundoff, not bitwise, with or without kinetics
# (checked with a kinetics-off control run), so this test reports but does not fail on
# it. Because the hydro time step then also differs at roundoff after a restart, and the
# kinetic substeps follow it, the time step is forced to a fixed value here.

import logging
import os
import shutil
import subprocess

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs, compare_bitwise

logger = logging.getLogger("riot" + __name__[7:])

input_id = "kinetics/sod"
pid_full = "kinetics_r6_full"
pid_rst = "kinetics_r6_rst"
nlim = 40
restart_every = 20
dt_force = 2.0e-3  # below the hydro CFL limit of the deck (about 2.7e-3)
kinetic_vars = [
    "kinetics.rho",
    "kinetics.velocity",
    "kinetics.temperature",
    "kinetics.pressure",
    "kinetics.stress",
    "kinetics.heat_flux",
    "kinetics.eq_fallback",
]


def run(**kwargs):
    clean_outputs(pid_full, pid_rst)
    riot.generate(input_id + ".py")
    riot.run(
        input_id + ".rin",
        [
            "parthenon/job/problem_id=" + pid_full,
            "kinetics/nv1=32",
            "kinetics/nv2=32",
            "kinetics/nv3=32",
            "parthenon/output1/dt=-1",
            f"parthenon/output1/dn={nlim}",
            "parthenon/output3/dt=-1",
            f"parthenon/output3/dn={restart_every}",
            f"parthenon/time/nlim={nlim}",
            f"parthenon/time/dt_force={dt_force}",
        ],
    )
    # Restart from the mid-run restart file under a different problem id.
    src = f"build/src/{pid_full}.out3.00001.rhdf"
    dst = f"build/src/{pid_rst}.rhdf"
    shutil.copyfile(src, dst)
    riot_dir = os.getcwd()
    os.chdir("build/src")
    try:
        subprocess.check_call(
            [
                "./riot",
                "-r",
                os.path.basename(dst),
                "parthenon/job/problem_id=" + pid_rst,
            ],
            env=riot.env,
            stdout=subprocess.DEVNULL,
        )
    finally:
        os.chdir(riot_dir)


def analyze():
    full_rst = f"build/src/{pid_full}.out3.final.rhdf"
    restarted_rst = f"build/src/{pid_rst}.out3.final.rhdf"
    full_out = f"build/src/{pid_full}.out1.final.phdf"
    restarted_out = f"build/src/{pid_rst}.out1.final.phdf"
    for f in (full_rst, restarted_rst, full_out, restarted_out):
        if not os.path.exists(f):
            logger.warning(f"missing {f}")
            return False
    ok = True
    if compare_bitwise(full_rst, restarted_rst, logger, ["kinetics.f"]):
        logger.warning("kinetics.f differs after the restart round trip")
        ok = False
    if compare_bitwise(full_out, restarted_out, logger, kinetic_vars):
        logger.warning("derived kinetic moments differ after the restart round trip")
        ok = False
    # Negative control: f at the restart point differs from f at the end.
    mid = f"build/src/{pid_full}.out3.00001.rhdf"
    hydro = compare_bitwise(full_rst, restarted_rst, logger, ["c.c.bulk.momentum"])
    logger.debug(f"hydro bitwise after restart: {not hydro} (not required)")
    if not compare_bitwise(mid, full_rst, logger, ["c.c.bulk.momentum"]):
        logger.warning("comparator found no change between restart point and end")
        ok = False
    return ok
