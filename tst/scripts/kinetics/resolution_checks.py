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

# Regression test R9 (claude_sessions/kinetic_bgk/S0_DESIGN.md): the velocity-space
# resolution diagnostics fire on under-resolved decks (negative controls) and stay quiet
# on a resolved one.
#   * box [-3, 3]^3: mass on the box edge above kinetics/edge_mass_abort -> abort,
#   * 4^3 nodes on [-8, 8]^3: every equilibrium solve falls back -> abort,
#   * box [-6, 6]^3, 24^3: edge mass between warn and abort -> warning only,
#   * box [-8, 8]^3, 32^3: no warning.

import logging
import os
import subprocess

import scripts.utils.riot as riot

logger = logging.getLogger("riot" + __name__[7:])

input_id = "kinetics/sod"


def box(L, n):
    args = []
    for d in (1, 2, 3):
        args += [
            f"kinetics/v{d}min={-L}",
            f"kinetics/v{d}max={L}",
            f"kinetics/nv{d}={n}",
        ]
    return args


# name: (velocity grid arguments, expect the run to abort, expected message)
cases = {
    "narrow_box": (box(3.0, 12), True, "mass on the velocity-box edge above"),
    "coarse_grid": (box(8.0, 4), True, "fell back to the sampled Maxwellian"),
    "edge_warning": (box(6.0, 24), False, "WARNING: velocity box edge mass"),
    "resolved": (box(8.0, 32), False, None),
}
results = {}


def run(**kwargs):
    riot.generate(input_id + ".py")
    for name, (grid, _, _) in cases.items():
        log = f"build/src/kinetics_r9_{name}.log"
        args = [
            "parthenon/job/problem_id=kinetics_r9_" + name,
            "parthenon/time/nlim=0",
            "parthenon/output1/dt=-1",
            "parthenon/output3/dt=-1",
        ] + grid
        # The aborting cases are expected to fail; capture their output instead of
        # letting the runner raise.
        cmd = ["./riot", "-i", "../../../inputs/" + input_id + ".rin"] + args
        cwd = os.getcwd()
        os.chdir("build/src")
        try:
            with open(os.path.basename(log), "w") as fh:
                rc = subprocess.call(
                    cmd, stdout=fh, stderr=subprocess.STDOUT, env=riot.env
                )
        finally:
            os.chdir(cwd)
        results[name] = rc


def analyze():
    ok = True
    for name, (_, aborts, message) in cases.items():
        with open(f"build/src/kinetics_r9_{name}.log") as fh:
            text = fh.read()
        rc = results.get(name)
        logger.debug(f"{name}: return code {rc}")
        if aborts != (rc != 0):
            logger.warning(
                f"{name}: return code {rc}, expected {'abort' if aborts else 'success'}"
            )
            ok = False
        if message is not None and message not in text:
            logger.warning(f"{name}: expected message '{message}' not found")
            ok = False
        if name == "resolved" and "WARNING: velocity" in text:
            logger.warning("resolved deck produced a velocity-resolution warning")
            ok = False
    return ok
