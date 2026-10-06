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

# Restart and MPI on refined meshes (claude_sessions/kinetic_bgk/S4_DESIGN.md, S4-Q17).
#   * SMR restart: Sod deck on a two-level 1D mesh, closure coupling, forced dt (below
#     the fine-level hydro limit), 40 cycles, restart at 20; dense and TT. f in the final
#     restart file is bitwise identical to the uninterrupted run (hydro restarts to
#     roundoff only, as on uniform meshes). Control: f changes between the restart
#     point and the end.
#   * MPI: 1 vs 2 ranks give bitwise identical dumps and final restart files for
#     (a) the TT Sod on the SMR mesh and (b) the coupled TT entropy wave with adaptive
#     refinement (blocks move between ranks at each remesh); history within 1e-13.
#     The per-step rounding tallies (kinetics_tt_round_*, kinetics_tt_amr_round_*) are
#     not remeshed, so on the cycle of a remesh they miss the new and moved blocks and
#     are excluded from the comparison on those rows.
# AMR restarts are not tested (S4-Q17).

import logging
import os
import shutil
import subprocess

import numpy as np

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs, compare_bitwise, read_history

logger = logging.getLogger("riot" + __name__[7:])

small_v = [
    "kinetics/nv1=16",
    "kinetics/nv2=8",
    "kinetics/nv3=8",
    "kinetics/min_vth_over_dv=0",
] + [f"kinetics/v{d}{e}" for d in (1, 2, 3) for e in ("min=-8", "max=8")]
smr = [
    "parthenon/mesh/refinement=static",
    "parthenon/mesh/numlevel=2",
    "parthenon/static_refinement0/x1min=0.25",
    "parthenon/static_refinement0/x1max=0.75",
    "parthenon/static_refinement0/level=1",
    "parthenon/meshblock/nx1=16",
]
restart_args = (
    small_v
    + smr
    + [
        "parthenon/time/nlim=40",
        "parthenon/time/dt_force=1.0e-3",
        "parthenon/output1/dt=-1",
        "parthenon/output1/dn=40",
        "parthenon/output3/dt=-1",
        "parthenon/output3/dn=20",
    ]
)
reps = {"dense": [], "tt": ["kinetics/representation=tt", "kinetics/tt_eps=1e-12"]}
mpi_cases = {
    "smr_sod": (
        "kinetics/sod",
        small_v
        + smr
        + reps["tt"]
        + [
            "parthenon/time/nlim=30",
            "parthenon/output1/dt=-1",
            "parthenon/output1/dn=10",
            "parthenon/output3/dt=-1",
            "parthenon/output3/dn=30",
        ],
    ),
    "amr_wave": (
        "kinetics/freestream",
        [
            "kinetics/nv1=16",
            "kinetics/nv2=8",
            "kinetics/nv3=8",
            "kinetics/min_vth_over_dv=0",
            "parthenon/mesh/refinement=adaptive",
            "parthenon/mesh/numlevel=2",
            "parthenon/mesh/derefine_count=2",
            "parthenon/mesh/nx1=64",
            "parthenon/meshblock/nx1=8",
            "parthenon/refinement0/method=derivative_order_1",
            "parthenon/refinement0/field=c.c.bulk.rho",
            "parthenon/refinement0/refine_tol=0.012",
            "parthenon/refinement0/derefine_tol=0.008",
            "kinetics/amr_noneq_refine=1e30",
            "kinetics/amr_noneq_derefine=1e30",
            "kinetics/nu0=10.0",
            "kinetics/closure_coupling=true",
            "kinetics/tt_rank_max=8",
            "parthenon/output2/dn=1",
            "parthenon/output2/dt=-1",
            "parthenon/output3/file_type=rst",
            "parthenon/output3/write_xdmf=false",
            "parthenon/output3/hdf5_compression_level=0",
            "parthenon/output3/dt=-1",
            "parthenon/output3/dn=124",
        ]
        + reps["tt"],
    ),
}
ranks = (1, 2)


def pid(rep, tag):
    return f"kinetics_amrr_{rep}_{tag}"


def mpi_pid(case, n):
    return f"kinetics_amrr_{case}_np{n}"


def restart(problem_id, restart_file):
    riot_dir = os.getcwd()
    os.chdir("build/src")
    try:
        p = subprocess.run(
            ["./riot", "-r", restart_file, "parthenon/job/problem_id=" + problem_id],
            env=riot.env,
            capture_output=True,
            text=True,
        )
    finally:
        os.chdir(riot_dir)
    return p.returncode


def run(**kwargs):
    ids = [pid(r, t) for r in reps for t in ("full", "rst")]
    ids += [mpi_pid(c, n) for c in mpi_cases for n in ranks]
    clean_outputs(*ids)
    for deck in ("kinetics/sod", "kinetics/freestream"):
        riot.generate(deck + ".py")
    for rep, extra in reps.items():
        riot.run(
            "kinetics/sod.rin",
            ["parthenon/job/problem_id=" + pid(rep, "full")] + restart_args + extra,
        )
        shutil.copyfile(
            f"build/src/{pid(rep, 'full')}.out3.00001.rhdf",
            f"build/src/{pid(rep, 'rst')}.rhdf",
        )
        if restart(pid(rep, "rst"), f"{pid(rep, 'rst')}.rhdf") != 0:
            raise RuntimeError(f"{rep}: restart run failed")
    for case, (deck, args) in mpi_cases.items():
        for n in ranks:
            riot.mpirun(
                n,
                deck + ".rin",
                ["parthenon/job/problem_id=" + mpi_pid(case, n)]
                + args
                + ["parthenon/output2/data_format=%.17e"],
            )


def analyze():
    ok = True
    for rep in reps:
        var = ["kinetics.f_tt" if rep == "tt" else "kinetics.f"]
        full = f"build/src/{pid(rep, 'full')}.out3.final.rhdf"
        rst = f"build/src/{pid(rep, 'rst')}.out3.final.rhdf"
        if compare_bitwise(full, rst, logger, var):
            logger.warning(f"{rep}: f differs after the SMR restart round trip")
            ok = False
        if not compare_bitwise(
            f"build/src/{pid(rep, 'full')}.out3.00001.rhdf", full, logger, var
        ):
            logger.warning(f"{rep}: control: f did not change after the restart point")
            ok = False
        logger.info(f"{rep}: SMR restart f bitwise")
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
                logger.warning(f"{case}{suffix}: {diff} differ between 1 and 2 ranks")
                ok = False
        ha, ca = read_history(a)
        hb, _ = read_history(b)
        if ha.shape != hb.shape:
            logger.warning(f"{case}: history shapes {ha.shape} vs {hb.shape}")
            ok = False
            continue
        rel = np.abs(ha - hb) / np.maximum(np.abs(ha), 1.0e-16)
        remeshed = np.r_[False, np.diff(ha[:, ca["nbtotal"]]) != 0]
        tallies = [i for n, i in ca.items() if "tt_round" in n or "tt_amr_round" in n]
        rel[np.ix_(remeshed, tallies)] = 0.0
        worst = np.max(rel)
        logger.info(
            f"{case}: {len(files)} files bitwise, history rel diff {worst:.2e} "
            f"({int(remeshed.sum())} remesh rows)"
        )
        if worst > 1.0e-13:
            logger.warning(f"{case}: history differs by {worst:.2e}")
            ok = False
    return ok
