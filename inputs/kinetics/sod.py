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

# Sod shock tube for the kinetics package (BGK neutral gas, 1D-3V).
#
# Units are chosen so the nondimensional groups are round numbers: with Gamma = 5/3 and
# Cv = 1.5, k_B / m = (Gamma - 1) Cv = 1, so theta = k_B T / m = T and the thermal speed
# of the left state (rho = 1, P = 1, T = 1) is 1. The velocity box [-8, 8]^3 holds the
# Maxwellians of both states with negligible mass on its edge.

import riot


def make_input():

    riot.input("riot", problem="shock_tube")

    riot.input("parthenon/job", problem_id="kinetics_sod")

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.velocity",
            "c.c.bulk.pressure",
            "c.c.bulk.temperature",
            "kinetics.rho",
            "kinetics.velocity",
            "kinetics.temperature",
            "kinetics.pressure",
            "kinetics.stress",
            "kinetics.heat_flux",
            "kinetics.eq_fallback",
        ],
        file_type="hdf5",
        dt=0.05,
    )

    riot.input("parthenon/output2", file_type="hst", dt=0.01)

    # Restarts carry f (nv1*nv2*nv3 components per cell). XDMF would describe every
    # component (66 MB of text per dump at 32^3) and gzip costs ~100x the raw write, so
    # both are off; see claude_sessions/kinetic_bgk/OPEN_QUESTIONS.md (OQ2).
    riot.input(
        "parthenon/output3",
        file_type="rst",
        dt=0.1,
        write_xdmf=False,
        hdf5_compression_level=0,
    )

    riot.input(
        "parthenon/time",
        nlim=-1,
        tlim=0.2,
        integrator="rk2",
        ncycle_out=10,
    )

    riot.input(
        "parthenon/mesh",
        nx1=128,
        x1min=0.0,
        x1max=1.0,
        ix1_bc="outflow",
        ox1_bc="outflow",
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

    riot.input("parthenon/meshblock", nx1=32)

    riot.input("material0", eos_type="IdealGas", Gamma=5.0 / 3.0, Cv=1.5)

    riot.input("physics", hydro=True, kinetics=True)

    riot.input("hydro", recon="plm", cfl=0.8)

    riot.input(
        "kinetics",
        nv1=32,
        nv2=32,
        nv3=32,
        v1min=-8.0,
        v1max=8.0,
        v2min=-8.0,
        v2max=8.0,
        v3min=-8.0,
        v3max=8.0,
        nu_model="constant",
        nu0=1.0e4,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
