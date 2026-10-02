#!/usr/bin/env python3
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

# Smooth free streaming (S0 test R3b): a hydro entropy wave (density and temperature
# perturbed, uniform pressure and velocity) initializes the kinetic gas, which then
# streams freely (nu = 0) on a periodic domain. The exact kinetic solution is
# f(x, v, t) = f0(x - v t, v); the regression compares density moments with it and
# measures the spatial order of the semi-Lagrangian scheme.
# Units: k_B / m = 1 (Gamma = 5/3, Cv = 1.5); background rho = 1, p = 0.6, u = 0.5.

import riot


def make_input():

    riot.input("riot", problem="linear_modes")

    # Entropy mode along x1, amplitude 0.1. The problem generator sets tlim to
    # nperiod * wavelength / u = 0.25 * 1 / 0.5.
    riot.input(
        "problem",
        iprob=1,
        wave_flag=3,
        amp=0.1,
        vflow=0.5,
        along_x1=True,
        nperiod=0.25,
    )

    riot.input("parthenon/job", problem_id="kinetics_freestream")

    riot.input(
        "parthenon/output1",
        variables=["kinetics.rho", "kinetics.velocity", "kinetics.temperature"],
        file_type="hdf5",
        dt=0.25,
        write_xdmf=False,
    )

    riot.input("parthenon/output2", file_type="hst", dn=1, data_format="%.17e")

    riot.input(
        "parthenon/time",
        nlim=-1,
        tlim=0.5,
        integrator="rk2",
        ncycle_out=100,
    )

    riot.input(
        "parthenon/mesh",
        nx1=64,
        x1min=0.0,
        x1max=1.0,
        ix1_bc="periodic",
        ox1_bc="periodic",
        nx2=1,
        x2min=-0.5,
        x2max=0.5,
        ix2_bc="periodic",
        ox2_bc="periodic",
        nx3=1,
        x3min=-0.5,
        x3max=0.5,
        ix3_bc="periodic",
        ox3_bc="periodic",
    )

    riot.input("parthenon/meshblock", nx1=16)

    riot.input("material0", eos_type="IdealGas", Gamma=5.0 / 3.0, Cv=1.5)

    riot.input("physics", hydro=True, kinetics=True)

    riot.input("hydro", recon="plm", cfl=0.8)

    riot.input(
        "kinetics",
        nv1=24,
        nv2=24,
        nv3=24,
        v1min=-6.0,
        v1max=6.0,
        v2min=-6.0,
        v2max=6.0,
        v3min=-6.0,
        v3max=6.0,
        nu_model="constant",
        nu0=0.0,
        sl_order=2,
        sl_limiter="none",
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
