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

# Static mesh refinement with kinetics (claude_sessions/kinetic_bgk/S4_DESIGN.md, S4-Q8
# iii, S4-Q27). Smooth free streaming (nu = 0) of an entropy wave on a periodic 1D mesh
# whose middle quarter is refined one level, so the wave crosses two fine-coarse
# interfaces. Fine ghosts copy the parent cell (injection, S4-Q7). Pass criteria:
#   * observed order of the SMR L1 density error against the exact discrete-velocity
#     solution >= 0.9 (nx = 64, 128, 256, sl_order = 1; measured 0.92, 0.96), and the SMR
#     error below the uniform root-level error at every nx;
#   * with kinetics/amr_prolong = linear and sl_order = 2 (unlimited) the order is >= 1.8
#     (measured 2.01, 2.01; injection gives 1.1);
#   * the kinetic mass leak at the fine-coarse faces (S4-Q10: semi-Lagrangian is not
#     flux-corrected) is O(dx): it decreases with nx at a rate >= 0.9 (measured 0.97, 0.98);
#   * coupled run (nu = 10, closure_coupling) on the SMR mesh: hydro mass and energy
#     conserved to 1e-12 (flux correction), and with kinetics/lomac the summed kinetic
#     moments equal the hydro ones to 1e-12 at every step (measured 3.7e-15; hydro drift
#     8.7e-16).
# Instrument check: the comparator reproduces the initial dump to roundoff.

import logging

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import (
    clean_outputs,
    free_streaming_moments,
    read_history,
    velocity_nodes,
)

logger = logging.getLogger("riot" + __name__[7:])

input_id = "kinetics/freestream"
resolutions = [64, 128, 256]
nv = (16, 8, 8)
vbox = 6.0
min_order = 0.9
min_order_linear = 1.8
smr = [
    "parthenon/mesh/refinement=static",
    "parthenon/mesh/numlevel=2",
    "parthenon/static_refinement0/x1min=0.375",
    "parthenon/static_refinement0/x1max=0.625",
    "parthenon/static_refinement0/level=1",
]
common = [
    f"kinetics/nv1={nv[0]}",
    f"kinetics/nv2={nv[1]}",
    f"kinetics/nv3={nv[2]}",
]
coupled = ["kinetics/nu0=10.0", "kinetics/closure_coupling=true"]
first = ["kinetics/sl_order=1"]


def pid(kind, nx=None):
    return f"kinetics_smr_{kind}" + ("" if nx is None else f"_n{nx}")


linear = [
    "kinetics/closure_coupling=false",
    "kinetics/sl_order=2",
    "kinetics/sl_limiter=none",
    "kinetics/amr_prolong=linear",
]


def cases():
    for nx in resolutions:
        yield pid("smr", nx), nx, smr + first + ["kinetics/closure_coupling=false"]
        yield pid("root", nx), nx, first + ["kinetics/closure_coupling=false"]
        yield pid("linear", nx), nx, smr + linear
    yield pid("coupled"), 64, smr + first + coupled
    yield pid("lomac"), 64, smr + first + coupled + ["kinetics/lomac=true"]


def run(**kwargs):
    clean_outputs(*[c[0] for c in cases()])
    riot.generate(input_id + ".py")
    for problem_id, nx, extra in cases():
        riot.run(
            input_id + ".rin",
            [
                "parthenon/job/problem_id=" + problem_id,
                f"parthenon/mesh/nx1={nx}",
                f"parthenon/meshblock/nx1={nx // 8}",
            ]
            + common
            + extra,
        )


def line(dump, var):
    """Cell centers, values and widths (sorted) on a 1D mesh of any level layout."""
    x = np.asarray(dump.x).ravel()
    xf = np.asarray(dump.xf)
    order = np.argsort(x)
    a = np.asarray(dump.Get(var, flatten=False)).reshape(-1)
    return x[order], a[order], (xf[:, 1:] - xf[:, :-1]).ravel()[order]


def density_error(problem_id, final=True):
    """L1 (volume-weighted) density error against the exact free-streaming solution."""
    d0 = phdf(f"build/src/{problem_id}.out1.00000.phdf")
    d = phdf(f"build/src/{problem_id}.out1.{'final' if final else '00000'}.phdf")
    x, r0, dx = line(d0, "kinetics.rho")
    _, T0, _ = line(d0, "kinetics.temperature")
    # Initial state rho = 1 + A sin(2 pi x), uniform p and u = (0.5, 0, 0).
    s = np.sin(2 * np.pi * x)
    A = np.sum((r0 - 1.0) * s) / np.sum(s * s)
    p0 = np.mean(r0 * T0)
    grids = [velocity_nodes(n, -vbox, vbox) for n in nv]

    def state(xf):
        rho = 1.0 + A * np.sin(2 * np.pi * xf)
        u = np.zeros((xf.size, 3))
        u[:, 0] = 0.5
        return rho, u, p0 / rho

    _, r, _ = line(d, "kinetics.rho")
    n, _, _ = free_streaming_moments(
        x, d.Time, state, [g[0] for g in grids], [g[1] for g in grids]
    )
    return np.sum(np.abs(r - n) * dx)


def drift(h, c):
    return np.max(np.abs(h[:, c] - h[0, c])) / np.abs(h[0, c])


def analyze():
    ok = True
    e0 = density_error(pid("smr", resolutions[0]), final=False)
    if e0 > 1.0e-12:
        logger.warning(f"comparator does not reproduce the initial state: {e0:.3e}")
        ok = False
    d = phdf(f"build/src/{pid('smr', resolutions[0])}.out1.00000.phdf")
    if len(set(np.asarray(d.level).tolist())) != 2:
        logger.warning(f"SMR mesh is not two-level: levels {d.level}")
        ok = False

    err_smr = [density_error(pid("smr", nx)) for nx in resolutions]
    err_root = [density_error(pid("root", nx)) for nx in resolutions]
    rates = [np.log2(err_smr[i] / err_smr[i + 1]) for i in range(len(err_smr) - 1)]
    logger.debug(f"SMR L1 {err_smr}, orders {rates}; root level L1 {err_root}")
    if min(rates) < min_order:
        logger.warning(f"SMR observed orders {rates} < {min_order}")
        ok = False
    for es, er, nx in zip(err_smr, err_root, resolutions):
        if not es < er:
            logger.warning(f"nx {nx}: SMR error {es:.3e} >= root-level error {er:.3e}")
            ok = False

    leak = []
    for nx in resolutions:
        h, c = read_history(pid("smr", nx))
        leak.append(drift(h, c["kinetics_sums_0"]))
    leak_rates = [np.log2(leak[i] / leak[i + 1]) for i in range(len(leak) - 1)]
    logger.debug(f"kinetic mass leak {leak}, rates {leak_rates}")
    if min(leak_rates) < min_order:
        logger.warning(f"kinetic mass leak {leak} is not O(dx): rates {leak_rates}")
        ok = False

    err_lin = [density_error(pid("linear", nx)) for nx in resolutions]
    rates_lin = [np.log2(err_lin[i] / err_lin[i + 1]) for i in range(len(err_lin) - 1)]
    logger.debug(f"linear prolongation, sl_order 2: L1 {err_lin}, orders {rates_lin}")
    if min(rates_lin) < min_order_linear:
        logger.warning(f"linear prolongation orders {rates_lin} < {min_order_linear}")
        ok = False

    h, c = read_history(pid("coupled"))
    for name in ("kinetics_sums_7", "kinetics_sums_11"):
        dr = drift(h, c[name])
        logger.debug(f"coupled SMR: hydro {name} drift {dr:.3e}")
        if dr > 1.0e-12:
            logger.warning(f"coupled SMR: hydro {name} drift {dr:.3e}")
            ok = False

    h, c = read_history(pid("lomac"))
    for kin, hyd in ((0, 7), (1, 8), (4, 11)):
        a, b = h[:, c[f"kinetics_sums_{kin}"]], h[:, c[f"kinetics_sums_{hyd}"]]
        gap = np.max(np.abs(a - b)) / np.max(np.abs(b))
        logger.debug(f"lomac SMR: kinetic {kin} vs hydro {hyd}: {gap:.3e}")
        if gap > 1.0e-12:
            logger.warning(f"lomac SMR: kinetic {kin} vs hydro {hyd}: {gap:.3e}")
            ok = False
    return ok
