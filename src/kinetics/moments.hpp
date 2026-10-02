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
#ifndef KINETICS_MOMENTS_HPP_
#define KINETICS_MOMENTS_HPP_
// This file was made in part with generative AI.

// Velocity moments of the distribution function of one spatial cell.
//
// All moments are in NUMBER units (no particle mass): multiply by m for mass, momentum,
// energy, stress and heat flux. The distribution is read through an accessor `f(n)`
// over the flat velocity index n (see VelocityGrid::Flat), so the same code works on a
// Parthenon pack inside a kernel and on a plain View in a unit test.

#include <cmath>

#include <parthenon/package.hpp>

#include "kinetics/equilibrium.hpp"
#include "kinetics/velocity_grid.hpp"

using namespace parthenon::package::prelude;

namespace Kinetics {

// Moments of the collision invariants, plus the diagonal second moments per axis.
struct RawMoments {
  Real n;      // sum f w
  Real nu[3];  // sum v_d f w
  Real nvv[3]; // sum v_d^2 f w; their sum is twice the energy per mass
};

// Central moments (in number units) about the mean velocity.
struct CentralMoments {
  Real stress[6]; // sum c_i c_j f w, ordered xx, yy, zz, xy, xz, yz
  Real heat[3];   // (1/2) sum c_i |c|^2 f w
};

template <class F>
KOKKOS_INLINE_FUNCTION RawMoments ComputeRawMoments(const VelocityGrid &grid,
                                                    const F &f) {
  RawMoments m{0.0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  for (int iz = 0; iz < grid.nv[2]; ++iz) {
    const Real vz = grid.Node(2, iz);
    for (int iy = 0; iy < grid.nv[1]; ++iy) {
      const Real vy = grid.Node(1, iy);
      for (int ix = 0; ix < grid.nv[0]; ++ix) {
        const Real vx = grid.Node(0, ix);
        const Real fv = f(grid.Flat(ix, iy, iz));
        m.n += fv;
        m.nu[0] += vx * fv;
        m.nu[1] += vy * fv;
        m.nu[2] += vz * fv;
        m.nvv[0] += vx * vx * fv;
        m.nvv[1] += vy * vy * fv;
        m.nvv[2] += vz * vz * fv;
      }
    }
  }
  const Real w = grid.Weight();
  m.n *= w;
  for (int d = 0; d < 3; ++d) {
    m.nu[d] *= w;
    m.nvv[d] *= w;
  }
  return m;
}

// Second pass about the mean velocity u (from ComputeRawMoments).
template <class F>
KOKKOS_INLINE_FUNCTION CentralMoments ComputeCentralMoments(const VelocityGrid &grid,
                                                            const F &f, const Real u[3]) {
  CentralMoments m{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  for (int iz = 0; iz < grid.nv[2]; ++iz) {
    const Real cz = grid.Node(2, iz) - u[2];
    for (int iy = 0; iy < grid.nv[1]; ++iy) {
      const Real cy = grid.Node(1, iy) - u[1];
      for (int ix = 0; ix < grid.nv[0]; ++ix) {
        const Real cx = grid.Node(0, ix) - u[0];
        const Real fv = f(grid.Flat(ix, iy, iz));
        const Real hc2 = 0.5 * (cx * cx + cy * cy + cz * cz) * fv;
        m.stress[0] += cx * cx * fv;
        m.stress[1] += cy * cy * fv;
        m.stress[2] += cz * cz * fv;
        m.stress[3] += cx * cy * fv;
        m.stress[4] += cx * cz * fv;
        m.stress[5] += cy * cz * fv;
        m.heat[0] += cx * hc2;
        m.heat[1] += cy * hc2;
        m.heat[2] += cz * hc2;
      }
    }
  }
  const Real w = grid.Weight();
  for (int a = 0; a < 6; ++a)
    m.stress[a] *= w;
  for (int d = 0; d < 3; ++d)
    m.heat[d] *= w;
  return m;
}

// Isotropic equilibrium target (n, u, theta = k_B T / m) from raw moments.
KOKKOS_INLINE_FUNCTION EquilibriumTarget IsotropicTarget(const RawMoments &m) {
  EquilibriumTarget t;
  t.n = m.n;
  const Real inv_n = (m.n > 0.0) ? 1.0 / m.n : 0.0;
  Real u2 = 0.0;
  for (int d = 0; d < 3; ++d) {
    t.u[d] = m.nu[d] * inv_n;
    u2 += t.u[d] * t.u[d];
  }
  const Real theta = ((m.nvv[0] + m.nvv[1] + m.nvv[2]) * inv_n - u2) / 3.0;
  for (int d = 0; d < 3; ++d)
    t.theta[d] = theta;
  return t;
}

// Per-axis target (n, u, theta_d) from raw moments, for the anisotropic solve.
KOKKOS_INLINE_FUNCTION EquilibriumTarget AnisotropicTarget(const RawMoments &m) {
  EquilibriumTarget t;
  t.n = m.n;
  const Real inv_n = (m.n > 0.0) ? 1.0 / m.n : 0.0;
  for (int d = 0; d < 3; ++d) {
    t.u[d] = m.nu[d] * inv_n;
    t.theta[d] = m.nvv[d] * inv_n - t.u[d] * t.u[d];
  }
  return t;
}

// Write the equilibrium onto every node through a writable accessor `f(n) = value`.
template <class F>
KOKKOS_INLINE_FUNCTION void FillEquilibrium(const VelocityGrid &grid,
                                            const Maxwellian &eq, const F &f) {
  for (int iz = 0; iz < grid.nv[2]; ++iz)
    for (int iy = 0; iy < grid.nv[1]; ++iy)
      for (int ix = 0; ix < grid.nv[0]; ++ix)
        f(grid.Flat(ix, iy, iz)) = eq(grid, ix, iy, iz);
}

} // namespace Kinetics

#endif // KINETICS_MOMENTS_HPP_
