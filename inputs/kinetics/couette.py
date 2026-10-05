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

# Planar Couette flow with heat conduction between diffuse walls (S0 test R7).
# A uniform gas at rest (rho = P = T = 1, so theta = k_B T / m = 1 with Gamma = 5/3,
# Cv = 1.5) between walls at x = 0 and x = 1. The walls move tangentially at -/+U and
# are held at T_w = 1 +/- dT, small enough that the flow stays in the linear regime.
# Constant nu = sqrt(2 theta) / l sets the mean free path l = mu sqrt(2 theta) / p
# (mu = p / nu); l = 0.05 here, resolved by 4 cells. Hydro sees the walls as
# no-penetration (reflecting) walls, which kinetic walls require with closure coupling.

import math

import riot

mfp = 0.05
U = 0.05
dT = 0.02


def make_input():

    riot.input("riot", problem="shock_tube")

    riot.input("shock_tube", rho_l=1.0, P_l=1.0, rho_r=1.0, P_r=1.0)

    riot.input("parthenon/job", problem_id="kinetics_couette")

    riot.input(
        "parthenon/output1",
        variables=[
            "kinetics.rho",
            "kinetics.velocity",
            "kinetics.temperature",
            "kinetics.pressure",
            "kinetics.stress",
            "kinetics.heat_flux",
        ],
        file_type="hdf5",
        dt=5.0,
        write_xdmf=False,
    )

    riot.input("parthenon/output2", file_type="hst", dt=0.5, data_format="%.17e")

    riot.input(
        "parthenon/time",
        nlim=-1,
        tlim=25.0,
        integrator="rk2",
        ncycle_out=1000,
    )

    riot.input(
        "parthenon/mesh",
        nx1=80,
        x1min=0.0,
        x1max=1.0,
        ix1_bc="reflecting",
        ox1_bc="reflecting",
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

    riot.input("parthenon/meshblock", nx1=20)

    riot.input("material0", eos_type="IdealGas", Gamma=5.0 / 3.0, Cv=1.5)

    riot.input("physics", hydro=True, kinetics=True)

    riot.input("hydro", recon="plm", cfl=0.8)

    riot.input(
        "kinetics",
        nv1=24,
        nv2=24,
        nv3=12,
        v1min=-7.0,
        v1max=7.0,
        v2min=-7.0,
        v2max=7.0,
        v3min=-7.0,
        v3max=7.0,
        nu_model="constant",
        nu0=math.sqrt(2.0) / mfp,
        sl_order=2,
        sl_limiter="none",
        min_vth_over_dv=1.0,
        # Kinetic transport test at sl_order 2; coupled kinetic walls need sl_order 1.
        closure_coupling=False,
        ix1_bc="diffuse",
        ix1_wall_T=1.0 + dT,
        ix1_wall_u2=-U,
        ox1_bc="diffuse",
        ox1_wall_T=1.0 - dT,
        ox1_wall_u2=U,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
