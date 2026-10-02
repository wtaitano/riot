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

# 0D BGK relaxation (S0 test R1): a uniform, periodic gas starts as a discrete
# bi-Maxwellian with T_x = 2 T_perp and relaxes with constant nu = 1. Each cell is an
# independent 0D problem. Units: k_B / m = 1 (Gamma = 5/3, Cv = 1.5), rho = P = T = 1.
# The anisotropy T_x - T_perp must decay as exp(-nu t) exactly, the invariants stay
# fixed to roundoff and the entropy never increases.

import riot


def make_input():

    riot.input("riot", problem="shock_tube")

    riot.input("shock_tube", rho_l=1.0, P_l=1.0, rho_r=1.0, P_r=1.0)

    riot.input("parthenon/job", problem_id="kinetics_relax0d")

    riot.input(
        "parthenon/output1",
        variables=["kinetics.rho", "kinetics.temperature", "kinetics.stress"],
        file_type="hdf5",
        dt=0.5,
    )

    riot.input("parthenon/output2", file_type="hst", dn=1, data_format="%.17e")

    riot.input(
        "parthenon/time",
        nlim=-1,
        tlim=5.0,
        integrator="rk2",
        ncycle_out=10,
    )

    riot.input(
        "parthenon/mesh",
        nx1=4,
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

    riot.input("parthenon/meshblock", nx1=4)

    riot.input("material0", eos_type="IdealGas", Gamma=5.0 / 3.0, Cv=1.5)

    riot.input("physics", hydro=True, kinetics=True)

    riot.input("hydro", recon="plm", cfl=0.8)

    riot.input(
        "kinetics",
        nv1=30,
        nv2=30,
        nv3=30,
        v1min=-10.0,
        v1max=10.0,
        v2min=-10.0,
        v2max=10.0,
        v3min=-10.0,
        v3max=10.0,
        nu_model="constant",
        nu0=1.0,
        init="bimaxwellian",
        init_T_ratio=2.0,
        init_axis=1,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
