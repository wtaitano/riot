//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
#ifndef KINETICS_KINETICS_BCS_HPP_
#define KINETICS_KINETICS_BCS_HPP_
// This file was made in part with generative AI.

// Kinetic boundary conditions for f on non-periodic faces, enrolled as package user
// boundary functions. Parthenon applies them after the mesh boundary function of the
// face, so they overwrite whatever the mesh BC wrote into the ghosts of f.
//
//   outflow:  zero gradient, every ghost layer copies the boundary cell (all nodes).
//   specular: f_ghost(v) = f_mirror(R v), R flipping the wall-normal velocity component
//             and the mirror cell reflected through the wall. The velocity box must be
//             symmetric about 0 along the wall-normal axis, so R v is a node.
//   diffuse:  nodes entering the domain carry n_w M_w(T_w, u_w), the discrete
//             equilibrium at the wall temperature and (tangential) velocity, with n_w
//             set so the incoming number flux equals the outgoing flux of the boundary
//             cell (zero net mass flux). Nodes leaving the domain get outflow values;
//             only the quadratic stencil reads them.

#include <memory>
#include <string>

#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

namespace Kinetics {

enum class KineticBC { periodic, outflow, specular, diffuse };

// Read kinetics/{ix1..ox3}_bc, check them against the mesh BCs and the velocity grid,
// and enroll the boundary functions on pkg.
void EnrollKineticBCs(StateDescriptor *pkg, ParameterInput *pin);

} // namespace Kinetics

#endif // KINETICS_KINETICS_BCS_HPP_
