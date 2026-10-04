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

# Regression test T-R7 (claude_sessions/kinetic_bgk/S1_DESIGN.md, step 7): restart and
# MPI in the tensor-train representation.
#   * Restart round trip (Sod deck, Nv 16 x 8 x 8, forced dt, 40 cycles, restart at 20):
#     kinetics.f_tt in the final restart file and every derived kinetic moment of the
#     final dump are bitwise identical to the uninterrupted run.
#   * Layout guard: restarting that file with a smaller tt_rank_max, or as dense, must
#     abort with the f-layout message (Parthenon reads components by position, so a
#     different layout would otherwise be read silently wrong); with a larger
#     tt_rank_max Parthenon's HDF5 read of f already fails ("not within extent").
#   * MPI: the Sod deck (1D) and the blast2d deck (2D, 4 blocks) on 1 and 4 ranks give
#     bitwise identical dumps and restart files; history within 1e-13 (MPI reductions).
# Negative control: f_tt at the restart point differs from the end.

import logging
import os
import shutil
import subprocess

import numpy as np

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs, compare_bitwise

logger = logging.getLogger("riot" + __name__[7:])

layout_msg = "restart file has f layout"

tt = ["kinetics/representation=tt", "kinetics/tt_eps=1e-12"]
small_v = [
    "kinetics/nv1=16",
    "kinetics/nv2=8",
    "kinetics/nv3=8",
    "kinetics/min_vth_over_dv=0",
]
pid_full, pid_rst = "kinetics_tr7_full", "kinetics_tr7_rst"
# Restarts with a different f layout: smaller layouts reach the package check (message);
# a larger one fails earlier in Parthenon's HDF5 read of f ("not within extent").
bad = {
    "kinetics_tr7_bad_rank": (["kinetics/tt_rank_max=8"], layout_msg),
    "kinetics_tr7_bad_dense": (["kinetics/representation=dense"], layout_msg),
    "kinetics_tr7_bad_grow": (["kinetics/tt_rank_max=24"], "not within extent"),
}
mpi_cases = {
    "sod": (
        "kinetics/sod",
        small_v
        + [
            "parthenon/time/nlim=30",
            "parthenon/output1/dt=-1",
            "parthenon/output1/dn=10",
            "parthenon/output3/dt=-1",
            "parthenon/output3/dn=30",
        ],
    ),
    "blast2d": (
        "kinetics/blast2d",
        [
            "kinetics/nv1=12",
            "kinetics/nv2=12",
            "kinetics/nv3=10",
            "kinetics/edge_mass_abort=1",
            "kinetics/edge_mass_warn=1",
            "parthenon/time/nlim=6",
            "parthenon/output1/dt=-1",
            "parthenon/output1/dn=3",
            "parthenon/output3/dt=-1",
            "parthenon/output3/dn=6",
        ],
    ),
}
ranks = (1, 4)
moment_vars = [
    "kinetics.rho",
    "kinetics.velocity",
    "kinetics.temperature",
    "kinetics.pressure",
    "kinetics.stress",
    "kinetics.heat_flux",
]


def mpi_pid(case, n):
    return f"kinetics_tr7_{case}_np{n}"


def restart(problem_id, restart_file, extra=()):
    """Run ./riot -r from build/src; returns (exit code, combined output)."""
    riot_dir = os.getcwd()
    os.chdir("build/src")
    try:
        p = subprocess.run(
            ["./riot", "-r", restart_file, "parthenon/job/problem_id=" + problem_id]
            + list(extra),
            env=riot.env,
            capture_output=True,
            text=True,
        )
    finally:
        os.chdir(riot_dir)
    return p.returncode, p.stdout + p.stderr


def run(**kwargs):
    ids = (
        [pid_full, pid_rst]
        + list(bad)
        + [mpi_pid(c, n) for c in mpi_cases for n in ranks]
    )
    clean_outputs(*ids)
    for deck in ("kinetics/sod", "kinetics/blast2d"):
        riot.generate(deck + ".py")
    riot.run(
        "kinetics/sod.rin",
        ["parthenon/job/problem_id=" + pid_full]
        + tt
        + small_v
        + [
            "parthenon/time/nlim=40",
            "parthenon/time/dt_force=2.0e-3",
            "parthenon/output1/dt=-1",
            "parthenon/output1/dn=40",
            "parthenon/output3/dt=-1",
            "parthenon/output3/dn=20",
        ],
    )
    shutil.copyfile(
        f"build/src/{pid_full}.out3.00001.rhdf", f"build/src/{pid_rst}.rhdf"
    )
    code, _ = restart(pid_rst, f"{pid_rst}.rhdf")
    if code != 0:
        raise RuntimeError(f"restart run failed with code {code}")
    for p, (extra, _) in bad.items():
        code, out = restart(p, f"{pid_rst}.rhdf", extra)
        with open(f"build/src/{p}.log", "w") as fh:
            fh.write(f"exit {code}\n" + out)
    for case, (deck, args) in mpi_cases.items():
        for n in ranks:
            riot.mpirun(
                n,
                deck + ".rin",
                ["parthenon/job/problem_id=" + mpi_pid(case, n)]
                + tt
                + args
                + ["parthenon/output2/data_format=%.17e"],
            )


def analyze():
    ok = True
    full_rst = f"build/src/{pid_full}.out3.final.rhdf"
    rst_rst = f"build/src/{pid_rst}.out3.final.rhdf"
    if compare_bitwise(full_rst, rst_rst, logger, ["kinetics.f_tt"]):
        logger.warning("kinetics.f_tt differs after the restart round trip")
        ok = False
    if compare_bitwise(
        f"build/src/{pid_full}.out1.final.phdf",
        f"build/src/{pid_rst}.out1.final.phdf",
        logger,
        moment_vars,
    ):
        logger.warning("derived moments differ after the restart round trip")
        ok = False
    if not compare_bitwise(
        f"build/src/{pid_full}.out3.00001.rhdf", full_rst, logger, ["kinetics.f_tt"]
    ):
        logger.warning("negative control: f_tt did not change between restart and end")
        ok = False
    for p, (_, expected) in bad.items():
        text = open(f"build/src/{p}.log").read()
        code = int(text.split("\n", 1)[0].split()[1])
        if code == 0 or expected not in text:
            logger.warning(f"{p}: restart with a different f layout was not refused")
            ok = False
    for case in mpi_cases:
        a, b = mpi_pid(case, ranks[0]), mpi_pid(case, ranks[1])
        files = sorted(
            os.path.basename(f)[len(a) :]
            for f in os.listdir("build/src")
            if f.startswith(a + ".out1.") and f.endswith(".phdf")
        ) + [".out3.final.rhdf"]
        for suffix in files:
            diff = compare_bitwise(
                f"build/src/{a}{suffix}", f"build/src/{b}{suffix}", logger
            )
            if diff:
                logger.warning(f"{case}{suffix}: {diff} differ between 1 and 4 ranks")
                ok = False
        ha = np.loadtxt(f"build/src/{a}.out2.hst")
        hb = np.loadtxt(f"build/src/{b}.out2.hst")
        if ha.shape != hb.shape:
            logger.warning(f"{case}: history shapes {ha.shape} vs {hb.shape}")
            ok = False
            continue
        rel = np.max(np.abs(ha - hb) / np.maximum(np.abs(ha), 1.0e-300))
        logger.info(f"{case}: {len(files)} files bitwise, history rel diff {rel:.2e}")
        if rel > 1.0e-13:
            logger.warning(f"{case}: history differs by {rel:.2e}")
            ok = False
    return ok
