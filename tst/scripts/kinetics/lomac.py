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

# LoMaC: kinetic moments enslaved to hydro (claude_sessions/kinetic_bgk/S3_DESIGN.md,
# S3-Q13). With kinetics/lomac the kinetic density, momentum and energy of every cell
# equal the hydro ones after every step. Pass criteria:
#   * exact moments: periodic entropy wave (freestream deck, nu = 30, 16 cells), dense
#     and TT: |kinetic - hydro| of the summed mass, momentum, energy at every step and of
#     rho, u, T in every cell of every dump < 1e-12 (measured 3e-15 / 9e-15). TT
#     (tt_eps 1e-12) matches dense to 1e-9 (measured 1.2e-10). Control: without LoMaC the per-cell gap is
#     > 1e-3 (measured 2.2e-2).
#   * gap: Sod at nu = 0 and 1e2, nx = 64, 128, 256. The L1 difference between the
#     LoMaC kinetic and the standalone (uncoupled) kinetic rho, u, T decreases with the
#     mesh at a rate >= 0.25 (measured 0.32-0.93): the correction is a consistent
#     O(dx) perturbation of the kinetic solution.
#   * walls: closed Sod box with specular and diffuse walls at the gas temperature
#     (mesh reflecting): kinetic and hydro mass, momentum and energy agree at every step
#     (< 1e-12; measured 1.7e-15), including the Sod boxes whose energy check walls.py
#     skips without LoMaC.
#   * TT rank bound: Sod, Nv 16^3, tt_rank_max 6 (hit 2,016 times): the rank stays
#     <= 6, no cell is skipped for rank capacity (the last rounding of a step uses
#     cap - 2), no cell is skipped at all.

import logging

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs, read_history, read_line

logger = logging.getLogger("riot" + __name__[7:])

moments = "c.c.bulk.rho,c.c.bulk.velocity,c.c.bulk.temperature," + (
    "kinetics.rho,kinetics.velocity,kinetics.temperature"
)
tt = ["kinetics/representation=tt", "kinetics/tt_eps=1e-12", "kinetics/tt_rank_max=16"]
wave = [
    "kinetics/nu0=30",
    "problem/amp=0.3",
    "kinetics/nv1=16",
    "kinetics/nv2=16",
    "kinetics/nv3=16",
    "kinetics/min_vth_over_dv=0",
    "parthenon/mesh/nx1=16",
    "parthenon/meshblock/nx1=4",
    "parthenon/time/dt_force=2e-3",
    "parthenon/output2/data_format=%.17e",
    "parthenon/output3/dt=-1",
    "parthenon/output1/variables=" + moments,
]
waves = {
    "kinetics_lomac_wave_dense": wave + ["kinetics/lomac=true"],
    "kinetics_lomac_wave_tt": wave + ["kinetics/lomac=true"] + tt,
    "kinetics_lomac_wave_off": wave,
}
sod = [
    "kinetics/nv1=24",
    "kinetics/nv2=24",
    "kinetics/nv3=24",
    "kinetics/min_vth_over_dv=0",
    "parthenon/output1/dt=0.2",
    "parthenon/output2/dt=-1",
    "parthenon/output3/dt=-1",
    "parthenon/output1/variables=kinetics.rho,kinetics.velocity,kinetics.temperature",
    "parthenon/time/tlim=0.2",
]
gap_nus = [0.0, 1.0e2]
gap_nxs = [64, 128, 256]
box = [
    "kinetics/nv1=16",
    "kinetics/nv2=8",
    "kinetics/nv3=8",
    "kinetics/min_vth_over_dv=0",
    "kinetics/sl_order=1",
    "kinetics/nu0=1.0e4",
    "kinetics/lomac=true",
    "parthenon/mesh/ix1_bc=reflecting",
    "parthenon/mesh/ox1_bc=reflecting",
    "parthenon/output1/dt=-1",
    "parthenon/output3/dt=-1",
    "parthenon/output2/data_format=%.17e",
    "parthenon/time/tlim=0.4",
]
boxes = {
    "kinetics_lomac_box_specular": box
    + ["kinetics/ix1_bc=specular", "kinetics/ox1_bc=specular"],
    "kinetics_lomac_box_diffuse": box
    + [
        "kinetics/ix1_bc=diffuse",
        "kinetics/ox1_bc=diffuse",
        "kinetics/ix1_wall_T=1.0",
        "kinetics/ox1_wall_T=1.0",
    ],
}
rank = [
    "kinetics/nv1=16",
    "kinetics/nv2=16",
    "kinetics/nv3=16",
    "kinetics/min_vth_over_dv=0",
    "parthenon/mesh/nx1=64",
    "parthenon/meshblock/nx1=16",
    "parthenon/output1/dt=-1",
    "parthenon/output3/dt=-1",
    "parthenon/output2/dt=-1",
    "parthenon/output2/dn=1",
    "parthenon/output2/data_format=%.17e",
    "parthenon/time/tlim=0.1",
    "kinetics/nu0=1e2",
    "kinetics/representation=tt",
    "kinetics/tt_eps=1e-12",
    "kinetics/tt_rank_max=6",
    "kinetics/lomac=true",
]
pid_rank = "kinetics_lomac_rank"


def pid_gap(nu, nx, lomac):
    tag = "lomac" if lomac else "std"
    return f"kinetics_lomac_gap_{tag}_nu{nu:.0e}_n{nx}".replace("+", "")


def gap_ids():
    return [pid_gap(nu, nx, lm) for nu in gap_nus for nx in gap_nxs for lm in (0, 1)]


def run(**kwargs):
    clean_outputs(*waves, *boxes, *gap_ids(), pid_rank)
    riot.generate("kinetics/freestream.py")
    for pid, args in waves.items():
        riot.run("kinetics/freestream.rin", ["parthenon/job/problem_id=" + pid] + args)
    riot.generate("kinetics/sod.py")
    for nu in gap_nus:
        for nx in gap_nxs:
            for lm in (False, True):
                riot.run(
                    "kinetics/sod.rin",
                    [
                        "parthenon/job/problem_id=" + pid_gap(nu, nx, lm),
                        f"kinetics/nu0={nu}",
                        f"parthenon/mesh/nx1={nx}",
                        f"parthenon/meshblock/nx1={nx // 4}",
                        (
                            "kinetics/lomac=true"
                            if lm
                            else "kinetics/closure_coupling=false"
                        ),
                    ]
                    + sod,
                )
    for pid, args in boxes.items():
        riot.run("kinetics/sod.rin", ["parthenon/job/problem_id=" + pid] + args)
    riot.run("kinetics/sod.rin", ["parthenon/job/problem_id=" + pid_rank] + rank)


def history_gaps(pid):
    """Largest |kinetic - hydro| of the summed mass, momentum, energy over the run."""
    h, cols = read_history(pid)
    s = lambda n: h[:, cols[f"kinetics_sums_{n}"]]
    m0, e0 = s(7)[0], s(11)[0]
    err = np.max(np.abs(s(0) - s(7))) / m0
    for a in range(3):
        err = max(err, np.max(np.abs(s(1 + a) - s(8 + a))) / m0)
    return max(err, np.max(np.abs(s(4) - s(11))) / e0)


def cell_gap(pid):
    """Largest per-cell |kinetic - hydro| of rho, u, T over all dumps."""
    worst = 0.0
    for which in ("00000", "00001", "final"):
        d = phdf(f"build/src/{pid}.out1.{which}.phdf")
        _, T = read_line(d, "c.c.bulk.temperature")
        cs = np.sqrt(np.max(T))
        for hv, kv, comp, scale in (
            ("c.c.bulk.rho", "kinetics.rho", None, None),
            ("c.c.bulk.velocity", "kinetics.velocity", 0, cs),
            ("c.c.bulk.temperature", "kinetics.temperature", None, None),
        ):
            _, a = read_line(d, hv, comp)
            _, b = read_line(d, kv, comp)
            ref = scale if scale is not None else np.max(np.abs(a))
            worst = max(worst, np.max(np.abs(a - b)) / ref)
    return worst


def analyze():
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            logger.warning(msg)
            ok = False

    # Exact moments every step
    for pid in ("kinetics_lomac_wave_dense", "kinetics_lomac_wave_tt"):
        eh, ec = history_gaps(pid), cell_gap(pid)
        logger.debug(f"{pid}: history gap {eh:.3e}, per-cell gap {ec:.3e}")
        check(
            eh < 1e-12 and ec < 1e-12, f"{pid}: kinetic != hydro ({eh:.2e}, {ec:.2e})"
        )
        h, cols = read_history(pid)
        check(h[:, cols["kinetics_lomac_0"]].max() == 0, f"{pid}: cells skipped")
    ec = cell_gap("kinetics_lomac_wave_off")
    logger.debug(f"control without LoMaC: per-cell gap {ec:.3e}")
    check(ec > 1e-3, f"control without LoMaC: per-cell gap {ec:.2e}")
    d_tt = 0.0
    for which in ("00001", "final"):
        a = phdf(f"build/src/kinetics_lomac_wave_dense.out1.{which}.phdf")
        b = phdf(f"build/src/kinetics_lomac_wave_tt.out1.{which}.phdf")
        for v in ("kinetics.rho", "kinetics.temperature"):
            d_tt = max(d_tt, np.max(np.abs(read_line(a, v)[1] - read_line(b, v)[1])))
    logger.debug(f"TT vs dense: {d_tt:.3e}")
    check(d_tt < 1e-9, f"TT vs dense {d_tt:.2e}")

    # LoMaC kinetic vs standalone kinetic converges with the mesh
    for nu in gap_nus:
        e = []
        for nx in gap_nxs:
            a = phdf(f"build/src/{pid_gap(nu, nx, False)}.out1.final.phdf")
            b = phdf(f"build/src/{pid_gap(nu, nx, True)}.out1.final.phdf")
            e.append(
                [
                    np.mean(np.abs(read_line(a, v, c)[1] - read_line(b, v, c)[1]))
                    for v, c in (
                        ("kinetics.rho", None),
                        ("kinetics.velocity", 0),
                        ("kinetics.temperature", None),
                    )
                ]
            )
        e = np.array(e)
        rates = np.log2(e[:-1] / e[1:])
        logger.debug(f"nu = {nu:.0e}: LoMaC - standalone {e.tolist()}, rates {rates}")
        check(np.min(rates) >= 0.25, f"nu = {nu:.0e}: rates {rates.tolist()}")

    # Walls
    for pid in boxes:
        eh = history_gaps(pid)
        logger.debug(f"{pid}: history gap {eh:.3e}")
        check(eh < 1e-12, f"{pid}: kinetic != hydro ({eh:.2e})")

    # TT rank bound with the cap - 2 rule
    h, cols = read_history(pid_rank)
    rmax = h[:, cols["kinetics_tt_max_rank"]].max()
    caps = h[:, cols["kinetics_tt_round_1"]].sum()
    skipped = h[:, cols["kinetics_lomac_0"]].max()
    logger.debug(f"rank bound: max rank {rmax}, cap hits {caps}, skipped {skipped}")
    check(rmax <= 6, f"rank {rmax} above tt_rank_max")
    check(caps > 0, "rank cap never hit: the test does not exercise the cap - 2 rule")
    check(skipped == 0, f"{skipped} cells skipped the correction")
    return ok
