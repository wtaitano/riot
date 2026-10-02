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

# Regression test R7 (claude_sessions/kinetic_bgk/S0_DESIGN.md): planar Couette flow with
# heat conduction between diffuse walls, near the continuum limit (mean free path 1/20
# of the gap). At steady state the shear stress P_xy and the heat flux q_x are uniform
# and the interior profiles are linear; the transport coefficients must match the BGK
# Chapman-Enskog values mu = p / nu and kappa = (5/2)(k_B/m) mu (Pr = 1). The walls
# produce a velocity slip and a temperature jump of the order of the mean free path.
# A free-molecular run (nu = 0) is the negative control: its "viscosity" from the same
# measurement is far from p / nu.

import logging

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs, read_line

logger = logging.getLogger("riot" + __name__[7:])

input_id = "kinetics/couette"
pid = "kinetics_r7"
pid_fm = "kinetics_r7_free_molecular"
mfp = 0.05
nu = np.sqrt(2.0) / mfp
U, dT = 0.05, 0.02
tlim = 10.0


def run(**kwargs):
    clean_outputs(pid, pid_fm)
    riot.generate(input_id + ".py")
    riot.run(
        input_id + ".rin",
        ["parthenon/job/problem_id=" + pid, f"parthenon/time/tlim={tlim}"],
    )
    riot.run(
        input_id + ".rin",
        [
            "parthenon/job/problem_id=" + pid_fm,
            "kinetics/nu0=0.0",
            "kinetics/sl_order=1",
            f"parthenon/time/tlim={tlim}",
        ],
    )


def transport(problem_id):
    d = phdf(f"build/src/{problem_id}.out1.final.phdf")
    x, uy = read_line(d, "kinetics.velocity", 1)
    _, T = read_line(d, "kinetics.temperature")
    _, p = read_line(d, "kinetics.pressure")
    _, pxy = read_line(d, "kinetics.stress", 3)
    _, qx = read_line(d, "kinetics.heat_flux", 0)
    m = (x > 0.125) & (x < 0.875)  # interior, away from the Knudsen layers
    su = np.polyfit(x[m], uy[m], 1)
    sT = np.polyfit(x[m], T[m], 1)
    return dict(
        mu=-pxy[m].mean() / su[0],
        kappa=-qx[m].mean() / sT[0],
        mu_bgk=p[m].mean() / nu,
        pxy_spread=pxy[m].std() / abs(pxy[m].mean()),
        slip=(np.polyval(su, 0.0) + U) / su[0],
        jump=(1.0 + dT - np.polyval(sT, 0.0)) / (-sT[0]),
    )


def analyze():
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            logger.warning(msg)
            ok = False

    r = transport(pid)
    logger.debug(f"continuum run: {r}")
    check(
        abs(r["mu"] / r["mu_bgk"] - 1.0) < 0.01,
        f"viscosity {r['mu']:.4e} vs {r['mu_bgk']:.4e}",
    )
    kappa_bgk = 2.5 * r["mu_bgk"]  # k_B / m = 1
    check(
        abs(r["kappa"] / kappa_bgk - 1.0) < 0.01,
        f"conductivity {r['kappa']:.4e} vs {kappa_bgk:.4e}",
    )
    check(r["pxy_spread"] < 1.0e-3, f"shear stress not uniform: {r['pxy_spread']:.2e}")
    # Slip and jump lengths are O(mfp) (Maxwell: ~1.0 l and ~1.2 l, scheme-dependent).
    check(0.5 * mfp < r["slip"] < 2.0 * mfp, f"slip length {r['slip']:.4f}")
    check(0.5 * mfp < r["jump"] < 3.0 * mfp, f"temperature-jump length {r['jump']:.4f}")

    fm = transport(pid_fm)
    logger.debug(f"free-molecular run: {fm}")
    check(
        abs(fm["mu"] / r["mu_bgk"] - 1.0) > 0.5,
        "free-molecular control matched the BGK viscosity",
    )
    return ok
