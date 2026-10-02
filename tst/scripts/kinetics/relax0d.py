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

# Regression test R1 (claude_sessions/kinetic_bgk/S0_DESIGN.md): 0D BGK relaxation of a
# discrete bi-Maxwellian with constant nu, with both integrators. Pass criteria:
#   * kinetic mass, momentum and energy drift <= 1e-12 (relative),
#   * entropy H = sum f (ln f - 1) never increases,
#   * strang (exact relaxation): T_x - T_perp decays as exp(-nu t) to 1e-12 (relative),
#     uniformly in space,
#   * sl_dirk2: T_x - T_perp decays by the discrete factor (1 - c1)(1 - c2) per substep
#     (src/kinetics/kinetics_tasks.cpp) to 1e-12, and differs from exp(-nu t) by more
#     than 1e-9 (the comparator tells the two integrators apart).
# Negative control: the same run with the equilibrium Newton solve disabled on a coarse
# velocity grid (relaxation toward the sampled Maxwellian) must fail the drift check.

import glob
import logging
import os

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot
from scripts.utils.kinetics_utils import clean_outputs

logger = logging.getLogger("riot" + __name__[7:])

input_id = "kinetics/relax0d"
pid = "kinetics_r1"
pid_strang = "kinetics_r1_strang"
pid_ctrl = "kinetics_r1_ctrl"
nu = 1.0
tol_drift = 1.0e-12
tol_decay = 1.0e-12


def run(**kwargs):
    clean_outputs(pid, pid_strang, pid_ctrl)
    riot.generate(input_id + ".py")
    riot.run(input_id + ".rin", ["parthenon/job/problem_id=" + pid])
    riot.run(
        input_id + ".rin",
        ["parthenon/job/problem_id=" + pid_strang, "kinetics/integrator=strang"],
    )
    riot.run(
        input_id + ".rin",
        [
            "parthenon/job/problem_id=" + pid_ctrl,
            "kinetics/nv1=10",
            "kinetics/nv2=10",
            "kinetics/nv3=10",
            "kinetics/eq_max_iter=0",
            "kinetics/eq_fallback_abort=1.0",
            "kinetics/min_vth_over_dv=0.0",
            "parthenon/time/tlim=1.0",
        ],
    )


def read_history(problem_id):
    fname = f"build/src/{problem_id}.out2.hst"
    with open(fname) as fh:
        header = [line for line in fh if line.startswith("# [1]")][0]
    cols = {}
    for tok in header[2:].split():
        if "=" in tok:
            idx, name = tok.split("=")
            cols[name] = int(idx[1:-1]) - 1
    return np.loadtxt(fname), cols


def drift(problem_id):
    hst, cols = read_history(problem_id)
    mass = hst[:, cols["kinetics_sums_0"]]
    energy = hst[:, cols["kinetics_sums_4"]]
    mom = hst[:, [cols[f"kinetics_sums_{d}"] for d in (1, 2, 3)]]
    # Thermal speed is 1 in these units, so momentum is scaled by the mass.
    return (
        max(
            np.max(np.abs(mass - mass[0])) / mass[0],
            np.max(np.abs(energy - energy[0])) / energy[0],
            np.max(np.abs(mom - mom[0])) / mass[0],
        ),
        hst[:, cols["kinetics_sums_5"]],
    )


def dirk_factor(problem_id, t):
    # Product of (1 - c1)(1 - c2) over the substeps taken up to time t. Row k of the
    # history holds the dt of the step from time_k to time_{k+1}; the substep count of
    # that step is reported in row k + 1.
    hst, cols = read_history(problem_id)
    g = 1.0 - 1.0 / np.sqrt(2.0)
    fac = 1.0
    for k in range(len(hst) - 1):
        if hst[k + 1, 0] > t * (1.0 + 1.0e-12):
            break
        nsub = hst[k + 1, cols["kinetics_substeps"]]
        z = nu * hst[k, 1] / nsub
        c1 = (1.0 - g) * z / (1.0 + g * z)
        c2 = g * z / (1.0 + g * z)
        fac *= ((1.0 - c1) * (1.0 - c2)) ** nsub
    return fac


def anisotropy(dump):
    data = phdf(dump)
    s = np.asarray(data.Get("kinetics.stress", flatten=False))
    s = np.moveaxis(s, 1, -1).reshape(-1, 6)
    rho = np.asarray(data.Get("kinetics.rho", flatten=False)).ravel()
    return (
        data.Time,
        (s[:, 0] - 0.5 * (s[:, 1] + s[:, 2])) / rho,
    )  # theta_x - theta_perp


def analyze():
    ok = True
    for problem_id in (pid, pid_strang):
        d, H = drift(problem_id)
        logger.debug(f"{problem_id}: invariant drift {d:.3e}")
        if d > tol_drift:
            logger.warning(f"{problem_id}: invariant drift {d:.3e} above {tol_drift}")
            ok = False
        dH = np.diff(H)
        if np.any(dH > 1.0e-14 * np.abs(H[:-1])):
            logger.warning(f"{problem_id}: entropy increased: max dH {dH.max():.3e}")
            ok = False

    exp_err_dirk = 0.0
    for problem_id in (pid, pid_strang):
        dumps = sorted(glob.glob(f"build/src/{problem_id}.out1.*.phdf"))
        if len(dumps) < 3:
            logger.warning(f"{problem_id}: too few dumps for the decay check")
            ok = False
            continue
        _, a0 = anisotropy(dumps[0])
        for f in dumps[1:]:
            t, a = anisotropy(f)
            err_exp = np.max(np.abs(a - a0 * np.exp(-nu * t)) / a0)
            if problem_id == pid_strang:
                err = err_exp
            else:
                err = np.max(np.abs(a - a0 * dirk_factor(problem_id, t)) / a0)
                exp_err_dirk = max(exp_err_dirk, err_exp)
            logger.debug(f"{problem_id} t = {t:.4g}: decay error {err:.3e}")
            if err > tol_decay:
                logger.warning(f"{problem_id} t = {t:.4g}: decay error {err:.3e}")
                ok = False
    logger.debug(f"sl_dirk2 vs exp(-nu t): {exp_err_dirk:.3e}")
    if exp_err_dirk <= 1.0e-9:
        logger.warning("sl_dirk2 matches exp(-nu t); the decay comparator is blind")
        ok = False

    d_ctrl, _ = drift(pid_ctrl)
    logger.debug(f"negative control drift {d_ctrl:.3e}")
    if d_ctrl <= tol_drift:
        logger.warning("negative control passed the drift check; comparator is blind")
        ok = False
    return ok
