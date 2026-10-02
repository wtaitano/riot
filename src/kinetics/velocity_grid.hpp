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
#ifndef KINETICS_VELOCITY_GRID_HPP_
#define KINETICS_VELOCITY_GRID_HPP_
// This file was made in part with generative AI.

// Fixed uniform velocity grid shared by every spatial cell.
//
// Per axis d the box [vmin_d, vmin_d + nv_d dv_d] is split into nv_d equal cells, and
// the nodes are the cell centers v_{d,i} = vmin_d + (i + 1/2) dv_d with equal (midpoint)
// weights. The distribution function of one spatial cell is stored flat over the three
// velocity indices, n = (iz * nv_y + iy) * nv_x + ix.
//
// The grid is uniform, so node positions are computed rather than stored. The struct is
// plain data and can be captured by value in device kernels.

#include <cmath>

#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

using namespace parthenon::package::prelude;

namespace Kinetics {

struct VelocityGrid {
  int nv[3];
  Real vmin[3];
  Real dv[3];

  KOKKOS_INLINE_FUNCTION int Size() const { return nv[0] * nv[1] * nv[2]; }

  // Node position along axis d.
  KOKKOS_INLINE_FUNCTION Real Node(const int d, const int i) const {
    return vmin[d] + (i + 0.5) * dv[d];
  }

  // Quadrature weight of every node (the velocity cell volume).
  KOKKOS_INLINE_FUNCTION Real Weight() const { return dv[0] * dv[1] * dv[2]; }

  KOKKOS_INLINE_FUNCTION int Flat(const int ix, const int iy, const int iz) const {
    return (iz * nv[1] + iy) * nv[0] + ix;
  }

  KOKKOS_INLINE_FUNCTION void Unflatten(const int n, int &ix, int &iy, int &iz) const {
    ix = n % nv[0];
    iy = (n / nv[0]) % nv[1];
    iz = n / (nv[0] * nv[1]);
  }

  // Largest node speed along axis d (outermost node, not the box edge).
  KOKKOS_INLINE_FUNCTION Real MaxNodeSpeed(const int d) const {
    return std::max(std::abs(Node(d, 0)), std::abs(Node(d, nv[d] - 1)));
  }
};

// Build a grid from per-axis bounds and node counts.
inline VelocityGrid MakeVelocityGrid(const int nv[3], const Real vmin[3],
                                     const Real vmax[3]) {
  VelocityGrid grid;
  for (int d = 0; d < 3; ++d) {
    PARTHENON_REQUIRE(nv[d] >= 1, "kinetics: velocity node count must be >= 1");
    PARTHENON_REQUIRE(vmax[d] > vmin[d], "kinetics: velocity box needs vmax > vmin");
    grid.nv[d] = nv[d];
    grid.vmin[d] = vmin[d];
    grid.dv[d] = (vmax[d] - vmin[d]) / nv[d];
  }
  return grid;
}

} // namespace Kinetics

#endif // KINETICS_VELOCITY_GRID_HPP_
