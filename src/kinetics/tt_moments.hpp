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
#ifndef KINETICS_TT_MOMENTS_HPP_
#define KINETICS_TT_MOMENTS_HPP_
// This file was made in part with generative AI.

// Velocity moments of a tensor-train f, by exact contraction of the cores with 1D
// polynomial weights: every moment up to total degree 3 is
//
//   M[p][q][s] = sum_{i,j,k} (v_x - s_x)^p (v_y - s_y)^q (v_z - s_z)^s f(i,j,k)
//              = X_p^T Y_q Z_s,   X_p(a) = sum_i (v_i - s_x)^p G1(i, a), ...
//
// at O(n r1 r2) cost per cell. Results use the RawMoments / CentralMoments structs of
// moments.hpp and agree with the dense sums up to the order of summation.

#include <parthenon/package.hpp>

#include "kinetics/moments.hpp"
#include "kinetics/tt_tensor.hpp"
#include "kinetics/velocity_grid.hpp"

using namespace parthenon::package::prelude;

namespace Kinetics {
namespace TT {

// M[p][q][s] for p, q, s <= 3 about the point `shift`, without the quadrature weight.
// Entries with p + q + s > 3 are computed too (they cost nothing extra) but unused.
template <class Data>
KOKKOS_INLINE_FUNCTION void PolyMoments(const VelocityGrid &grid, const TTRef<Data> &t,
                                        const Real shift[3], Real M[4][4][4]) {
  const int r1 = t.R1(), r2 = t.R2();
  PARTHENON_DEBUG_REQUIRE(r1 <= kMaxRank && r2 <= kMaxRank,
                          "kinetics TT: PolyMoments rank above kMaxRank");
  // X_p(a) = sum_i (v_i - s_x)^p G1(i, a): the only rank-sized array (2 KB at kMaxRank).
  Real X[4][kMaxRank];
  for (int p = 0; p < 4; ++p)
    for (int a = 0; a < r1; ++a)
      X[p][a] = 0.0;
  for (int i = 0; i < grid.nv[0]; ++i) {
    const Real c = grid.Node(0, i) - shift[0];
    for (int a = 0; a < r1; ++a) {
      Real w = t.G1(i, a);
      for (int p = 0; p < 4; ++p) {
        X[p][a] += w;
        w *= c;
      }
    }
  }
  for (int p = 0; p < 4; ++p)
    for (int q = 0; q < 4; ++q)
      for (int s = 0; s < 4; ++s)
        M[p][q][s] = 0.0;
  // One second-core column b at a time, so no rank-sized XY or Z arrays are needed.
  for (int b = 0; b < r2; ++b) {
    Real XY[4][4] = {}; // sum_j (v_j - s_y)^q sum_a X_p(a) G2(a, j, b)
    for (int j = 0; j < grid.nv[1]; ++j) {
      const Real c = grid.Node(1, j) - shift[1];
      const Real cq[4] = {1.0, c, c * c, c * c * c};
      for (int p = 0; p < 4; ++p) {
        Real xg = 0.0;
        for (int a = 0; a < r1; ++a)
          xg += X[p][a] * t.G2(a, j, b);
        for (int q = 0; q < 4; ++q)
          XY[p][q] += cq[q] * xg;
      }
    }
    Real Z[4] = {}; // sum_k (v_k - s_z)^s G3(b, k)
    for (int k = 0; k < grid.nv[2]; ++k) {
      const Real c = grid.Node(2, k) - shift[2];
      Real w = t.G3(b, k);
      for (int s = 0; s < 4; ++s) {
        Z[s] += w;
        w *= c;
      }
    }
    for (int p = 0; p < 4; ++p)
      for (int q = 0; q < 4; ++q)
        for (int s = 0; s < 4; ++s)
          M[p][q][s] += XY[p][q] * Z[s];
  }
}

template <class Data>
KOKKOS_INLINE_FUNCTION RawMoments ComputeRawMoments(const VelocityGrid &grid,
                                                    const TTRef<Data> &t) {
  Real M[4][4][4];
  const Real zero[3] = {0.0, 0.0, 0.0};
  PolyMoments(grid, t, zero, M);
  const Real w = grid.Weight();
  RawMoments m;
  m.n = w * M[0][0][0];
  m.nu[0] = w * M[1][0][0];
  m.nu[1] = w * M[0][1][0];
  m.nu[2] = w * M[0][0][1];
  m.nvv[0] = w * M[2][0][0];
  m.nvv[1] = w * M[0][2][0];
  m.nvv[2] = w * M[0][0][2];
  return m;
}

template <class Data>
KOKKOS_INLINE_FUNCTION CentralMoments ComputeCentralMoments(const VelocityGrid &grid,
                                                            const TTRef<Data> &t,
                                                            const Real u[3]) {
  Real M[4][4][4];
  PolyMoments(grid, t, u, M);
  const Real w = grid.Weight();
  CentralMoments m;
  m.stress[0] = w * M[2][0][0];
  m.stress[1] = w * M[0][2][0];
  m.stress[2] = w * M[0][0][2];
  m.stress[3] = w * M[1][1][0];
  m.stress[4] = w * M[1][0][1];
  m.stress[5] = w * M[0][1][1];
  m.heat[0] = 0.5 * w * (M[3][0][0] + M[1][2][0] + M[1][0][2]);
  m.heat[1] = 0.5 * w * (M[2][1][0] + M[0][3][0] + M[0][1][2]);
  m.heat[2] = 0.5 * w * (M[2][0][1] + M[0][2][1] + M[0][0][3]);
  return m;
}

} // namespace TT
} // namespace Kinetics

#endif // KINETICS_TT_MOMENTS_HPP_
