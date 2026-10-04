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

# Closure coupling (claude_sessions/kinetic_bgk/CLOSURE_DESIGN.md, section 4): hydro fluxes
# take the kinetic stress and heat flux. The Sod cases use a 24^3 velocity grid (equal
# spacings: on an anisotropic grid the discrete equilibrium has P_xx != p at the 3e-3
# level, which would swamp the continuum-limit closure); the conservation case uses
# 16 x 12 x 12, since conservation and TT-vs-dense do not depend on it. Pass criteria:
#   * gap: Sod at nu = 0 and 1e2, nx = 64, 128, 256. The L1 gap between the hydro and the
#     kinetic rho, u, T decreases with the mesh at a rate >= 0.4 (measured 0.51-0.75;
#     first-order SL across discontinuities). Control: uncoupled at nu = 0 the gap does not
#     shrink (rate < 0.15; measured -0.08-0). Instrument: gap at t = 0 below 1e-12.
#   * continuum: Sod at nu = 1e4 and 1e5, nx = 64. Max |Pi_xx|, max |q_x| and the L1
#     change of hydro rho from the coupling each scale like 1 / nu (ratio 7-13 for a
#     factor 10; measured 9.8-10.0).
#   * free-molecular: at nu = 0 the coupled hydro density converges to the exact
#     discrete-velocity solution (errors decrease with the mesh).
#   * conservation + TT: periodic entropy wave (freestream deck, nu = 30, amp 0.3,
#     16 cells, forced dt). Hydro mass, momentum, energy drift < 1e-13 (dense and TT); TT
#     at tt_eps = 1e-14 matches dense in hydro rho to 1e-10 (measured 2e-12) while the
#     coupling itself changes hydro rho by more than 1e-2 (measured 8e-2).

import logging

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import (
    clean_outputs,
    free_streaming_moments,
    read_history,
    read_line,
    velocity_nodes,
)

logger = logging.getLogger("riot" + __name__[7:])

nv = (24, 24, 24)
vbox = 8.0
nvs = [f"kinetics/nv{d + 1}={n}" for d, n in enumerate(nv)]
sod_common = nvs + [
    "kinetics/min_vth_over_dv=0",
    "parthenon/output1/dt=0.2",
    "parthenon/output2/dt=-1",
    "parthenon/output3/dt=-1",
]
gap_nus = [0.0, 1.0e2]
gap_nxs = [64, 128, 256]
cont_nus = [1.0e4, 1.0e5]
tt_args = [
    "kinetics/representation=tt",
    "kinetics/tt_eps=1e-14",
    "kinetics/tt_rank_max=16",
]


def pid_gap(nu, nx, closure=True):
    return f"kinetics_cc_gap_nu{nu:.0e}_n{nx}".replace("+", "") + (
        "" if closure else "_off"
    )


def pid_cont(nu, closure=True):
    return f"kinetics_cc_cont_nu{nu:.0e}".replace("+", "") + ("" if closure else "_off")


def pid_cons(case):
    return f"kinetics_cc_cons_{case}"


def gap_cases():
    for nu in gap_nus:
        for nx in gap_nxs:
            yield nu, nx, True
    for nx in gap_nxs:
        yield 0.0, nx, False


def run(**kwargs):
    clean_outputs(
        *[pid_gap(*c) for c in gap_cases()],
        *[pid_cont(nu, c) for nu in cont_nus for c in (True, False)],
        *[pid_cons(c) for c in ("dense", "tt", "off")],
    )
    riot.generate("kinetics/sod.py")
    for nu, nx, closure in gap_cases():
        riot.run(
            "kinetics/sod.rin",
            [
                "parthenon/job/problem_id=" + pid_gap(nu, nx, closure),
                f"kinetics/nu0={nu}",
                f"kinetics/closure_coupling={'true' if closure else 'false'}",
                f"parthenon/mesh/nx1={nx}",
                f"parthenon/meshblock/nx1={nx // 4}",
            ]
            + sod_common,
        )
    for nu in cont_nus:
        for closure in (True, False):
            riot.run(
                "kinetics/sod.rin",
                [
                    "parthenon/job/problem_id=" + pid_cont(nu, closure),
                    f"kinetics/nu0={nu}",
                    f"kinetics/closure_coupling={'true' if closure else 'false'}",
                    "parthenon/mesh/nx1=64",
                    "parthenon/meshblock/nx1=16",
                    "parthenon/time/dt_force=2e-3",
                ]
                + sod_common,
            )
    riot.generate("kinetics/freestream.py")
    cons = [
        "kinetics/nu0=30",
        "problem/amp=0.3",
        "kinetics/nv1=16",
        "kinetics/nv2=12",
        "kinetics/nv3=12",
        "kinetics/min_vth_over_dv=0",
        "parthenon/mesh/nx1=16",
        "parthenon/meshblock/nx1=4",
        "parthenon/time/dt_force=2e-3",
        "parthenon/output1/variables=c.c.bulk.rho,kinetics.rho",
    ]
    for case, extra in (
        ("dense", []),
        ("tt", tt_args),
        ("off", ["kinetics/closure_coupling=false"]),
    ):
        riot.run(
            "kinetics/freestream.rin",
            ["parthenon/job/problem_id=" + pid_cons(case)] + cons + extra,
        )


def dump(problem_id, which="final"):
    return phdf(f"build/src/{problem_id}.out1.{which}.phdf")


def gaps(problem_id, which="final"):
    d = dump(problem_id, which)
    out = []
    for hv, kv, comp in (
        ("c.c.bulk.rho", "kinetics.rho", None),
        ("c.c.bulk.velocity", "kinetics.velocity", 0),
        ("c.c.bulk.temperature", "kinetics.temperature", None),
    ):
        _, h = read_line(d, hv, comp)
        _, k = read_line(d, kv, comp)
        out.append(np.mean(np.abs(h - k)))
    return np.array(out)


def sod_state(xf):
    left = xf < 0.5
    return (
        np.where(left, 1.0, 0.125),
        np.zeros((xf.size, 3)),
        np.where(left, 1.0, 0.8),
    )


def analyze():
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            logger.warning(msg)
            ok = False

    # Gap between hydro and kinetic moments
    g0 = np.max(gaps(pid_gap(0.0, gap_nxs[0]), "00000"))
    check(g0 < 1.0e-12, f"gap at t = 0 is {g0:.3e}")
    for nu in gap_nus:
        g = np.array([gaps(pid_gap(nu, nx)) for nx in gap_nxs])
        rates = np.log2(g[:-1] / g[1:])
        logger.debug(f"nu = {nu:.0e}: gaps {g.tolist()}, rates {rates.tolist()}")
        check(np.min(rates) >= 0.4, f"nu = {nu:.0e}: gap rates {rates.tolist()}")
    g = np.array([gaps(pid_gap(0.0, nx, False)) for nx in gap_nxs])
    rates = np.log2(g[:-1] / g[1:])
    logger.debug(f"uncoupled nu = 0: gaps {g.tolist()}, rates {rates.tolist()}")
    check(np.max(rates) < 0.15, f"uncoupled gap converges: rates {rates.tolist()}")

    # Continuum limit: closure and its effect on hydro scale like 1 / nu
    res = []
    for nu in cont_nus:
        d, o = dump(pid_cont(nu)), dump(pid_cont(nu, False))
        _, pxx = read_line(d, "kinetics.stress", 0)
        _, p = read_line(d, "kinetics.pressure")
        _, qx = read_line(d, "kinetics.heat_flux", 0)
        _, h1 = read_line(d, "c.c.bulk.rho")
        _, h0 = read_line(o, "c.c.bulk.rho")
        res.append(
            np.array(
                [np.max(np.abs(pxx - p)), np.max(np.abs(qx)), np.mean(np.abs(h1 - h0))]
            )
        )
    ratio = res[0] / res[1]
    logger.debug(f"continuum (Pi_xx, q_x, hydro change) ratios: {ratio.tolist()}")
    check(np.all((ratio > 7.0) & (ratio < 13.0)), f"continuum ratios {ratio.tolist()}")

    # Free-molecular: coupled hydro density vs the exact discrete-velocity solution
    grids = [velocity_nodes(n, -vbox, vbox) for n in nv]
    errs = []
    for nx in gap_nxs:
        d = dump(pid_gap(0.0, nx))
        x, rho = read_line(d, "c.c.bulk.rho")
        n, _, _ = free_streaming_moments(
            x, d.Time, sod_state, [g[0] for g in grids], [g[1] for g in grids]
        )
        errs.append(np.mean(np.abs(rho - n)))
    logger.debug(f"free-molecular hydro rho L1 errors {errs}")
    check(
        all(errs[i + 1] < errs[i] for i in range(len(errs) - 1)),
        f"free-molecular hydro errors do not decrease: {errs}",
    )

    # Conservation (flux form) and TT vs dense
    for case in ("dense", "tt"):
        h, cols = read_history(pid_cons(case))
        for n in (7, 8, 11):
            a = h[:, cols[f"kinetics_sums_{n}"]]
            drift = np.max(np.abs(a - a[0])) / np.abs(a[0])
            logger.debug(f"{case}: hydro invariant {n} drift {drift:.3e}")
            check(drift < 1.0e-13, f"{case}: hydro invariant {n} drift {drift:.3e}")
    rho = {
        c: np.asarray(dump(pid_cons(c)).Get("c.c.bulk.rho"))
        for c in ("dense", "tt", "off")
    }
    d_tt = np.max(np.abs(rho["tt"] - rho["dense"]))
    d_off = np.max(np.abs(rho["off"] - rho["dense"]))
    logger.debug(f"hydro rho: |tt - dense| {d_tt:.3e}, |on - off| {d_off:.3e}")
    check(d_tt < 1.0e-10, f"hydro rho |tt - dense| {d_tt:.3e}")
    check(d_off > 1.0e-2, f"hydro rho |on - off| {d_off:.3e}: coupling has no effect")
    return ok
