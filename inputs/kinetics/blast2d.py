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

# 2D cylindrical pressure pulse (S1 step 6 test deck for multi-D kinetic streaming): a
# gas at rest with a disk of 4x pressure and 2x density at the center, drifting
# diagonally with u = (0.3, 0.2), periodic in x1 and x2 (region_pgen). The flow structure
# is not aligned with a mesh or velocity axis, so both TT sweeps and all three velocity
# cores are exercised. Units: k_B / m = 1 (Gamma = 5/3, Cv = 1.5), background
# rho = P = T = 1. Small enough for dense and TT runs on a laptop.

import riot


def make_input():

    riot.input("riot", problem="region_pgen")

    riot.input("parthenon/job", problem_id="kinetics_blast2d")

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
        dt=0.05,
        write_xdmf=False,
    )

    riot.input("parthenon/output2", file_type="hst", dt=0.01, data_format="%.17e")

    # Restarts carry kinetics.f or kinetics.f_tt; see inputs/kinetics/sod.py for the
    # write_xdmf / compression settings.
    riot.input(
        "parthenon/output3",
        file_type="rst",
        dt=0.05,
        write_xdmf=False,
        hdf5_compression_level=0,
    )

    riot.input(
        "parthenon/time",
        nlim=-1,
        tlim=0.1,
        integrator="rk2",
        ncycle_out=10,
    )

    riot.input(
        "parthenon/mesh",
        nx1=24,
        x1min=-0.5,
        x1max=0.5,
        ix1_bc="periodic",
        ox1_bc="periodic",
        nx2=24,
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

    riot.input("parthenon/meshblock", nx1=12, nx2=12)

    riot.input("materials", sparse_dealloc=False)

    riot.input(
        "material0",
        eos_type="IdealGas",
        Gamma=5.0 / 3.0,
        Cv=1.5,
        max_bnd_level=0,
        max_mat_level=0,
    )

    riot.input(
        "region0",
        name="background",
        mask_type="background",
        matid=0,
        c_m_rho=1.0,
        c_m_pressure=1.0,
        c_c_bulk_velocity=[0.3, 0.2, 0.0],
    )

    riot.input(
        "region1",
        name="pulse",
        mask_type="inside_sphere",
        matid=0,
        radius=0.2,
        x0=0.0,
        y0=0.0,
        c_m_rho=2.0,
        c_m_pressure=4.0,
        c_c_bulk_velocity=[0.3, 0.2, 0.0],
    )

    riot.input("physics", hydro=True, kinetics=True)

    riot.input("hydro", recon="plm", cfl=0.8)

    riot.input(
        "kinetics",
        nv1=20,
        nv2=20,
        nv3=14,
        v1min=-9.0,
        v1max=9.0,
        v2min=-9.0,
        v2max=9.0,
        v3min=-9.5,
        v3max=9.5,
        nu_model="constant",
        nu0=1.0e2,
        min_vth_over_dv=0.0,
        tt_eps=1.0e-14,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
