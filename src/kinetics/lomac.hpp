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
#ifndef KINETICS_LOMAC_HPP_
#define KINETICS_LOMAC_HPP_
// This file was made in part with generative AI.

// LoMaC correction of one cell (claude_sessions/kinetic_bgk/S3_DESIGN.md, S3-Q1..Q6):
// enslave the discrete moments (n, n u, sum |v|^2 f w) of f to a target state (the hydro
// solution at t^{n+1}) by
//
//   f <- f + M P,   P = c0 + c1 xi_x + c2 xi_y + c3 xi_z + c4 |xi|^2,  xi = (v - u) / c,
//
// with M the sampled continuous Maxwellian of the target (rank 1, positive, no Newton
// solve) and u, c = sqrt(theta) its velocity and thermal speed. P spans {1, v, |v|^2};
// the xi basis only conditions the system. The five coefficients solve the 5 x 5 SPD
// system A c = b with A_ab = sum phi_a phi_b M w built from the discrete moments of M up
// to fourth order (separable 1D sums, exact on the grid), so the corrected f has the
// target moments to roundoff whatever the grid resolution.
//
// Tensor train: M P is an exact TT of ranks (2, 2),
//   G1 = [m_x A(xi_x), m_x], G2 = [[m_y, 0], [m_y B(xi_y), m_y]], G3 = [m_z; m_z
//   C(xi_z)], A = c0 + c1 xi + c4 xi^2, B = c2 xi + c4 xi^2, C = c3 xi + c4 xi^2,
// and f + M P is their block sum, ranks (r1 + 2, r2 + 2), stored without rounding
// (S3-Q5: the last rounding of a step leaves the room).

#include <cmath>

#include <parthenon/package.hpp>

#include "kinetics/equilibrium.hpp"
#include "kinetics/moments.hpp"
#include "kinetics/tt_tensor.hpp"
#include "kinetics/velocity_grid.hpp"

using namespace parthenon::package::prelude;

namespace Kinetics {

// Target moments in number units: n, n u_d and sum |v|^2 f w (twice the energy per mass).
struct LomacTarget {
  Real n;
  Real nu[3];
  Real nvv;
};

struct LomacResult {
  enum class Status { applied, invalid_target, singular, rank_overflow };
  Status status;
  Maxwellian M; // window (the sampled Maxwellian of the target)
  Real c[5];    // coefficients of P in the xi basis
};

namespace lomac_detail {

// sum_n xi^p m_d(v_n), p = 0..4, along axis d (m_d without scale and b0).
KOKKOS_INLINE_FUNCTION void AxisSums(const VelocityGrid &grid, const Maxwellian &M,
                                     const int d, Real S[5]) {
  for (int p = 0; p < 5; ++p)
    S[p] = 0.0;
  for (int i = 0; i < grid.nv[d]; ++i) {
    const Real v = grid.Node(d, i);
    const Real xi = (v - M.u[d]) * M.inv_c;
    Real w = std::exp(M.AxisExponent(d, v));
    for (int p = 0; p < 5; ++p) {
      S[p] += w;
      w *= xi;
    }
  }
}

// Basis phi_a as monomials in xi: a = 0: 1; a = 1..3: xi_d; a = 4: sum_d xi_d^2.
// Number of monomials and the exponents of monomial t of phi_a.
KOKKOS_INLINE_FUNCTION int NumMono(const int a) { return (a == 4) ? 3 : 1; }
KOKKOS_INLINE_FUNCTION void Mono(const int a, const int t, int e[3]) {
  e[0] = e[1] = e[2] = 0;
  if (a >= 1 && a <= 3) e[a - 1] = 1;
  if (a == 4) e[t] = 2;
}

// P(xi) along one axis contribution, see the header comment (A, B, C).
KOKKOS_INLINE_FUNCTION Real AxisPoly(const Real c[5], const int d, const Real xi) {
  return ((d == 0) ? c[0] : 0.0) + (c[1 + d] + c[4] * xi) * xi;
}

} // namespace lomac_detail

// Window and coefficients for f with raw moments m (moments.hpp, weight included).
KOKKOS_INLINE_FUNCTION LomacResult LomacSolve(const VelocityGrid &grid,
                                              const RawMoments &m,
                                              const LomacTarget &target) {
  using namespace lomac_detail;
  LomacResult res;
  for (int a = 0; a < 5; ++a)
    res.c[a] = 0.0;
  res.M = Maxwellian{0.0, {0.0, 0.0, 0.0}, 1.0, 0.0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  if (!(target.n > 0.0)) {
    res.status = LomacResult::Status::invalid_target;
    return res;
  }
  Real u[3], u2 = 0.0;
  for (int d = 0; d < 3; ++d) {
    u[d] = target.nu[d] / target.n;
    u2 += u[d] * u[d];
  }
  const Real theta = (target.nvv / target.n - u2) / 3.0;
  if (!(theta > 0.0)) {
    res.status = LomacResult::Status::invalid_target;
    return res;
  }
  res.M = ContinuousMaxwellian(
      EquilibriumTarget{target.n, {u[0], u[1], u[2]}, {theta, theta, theta}});
  const Maxwellian &M = res.M;
  const Real c = 1.0 / M.inv_c;

  // Discrete moments of M: Mom(p, q, r) = K S_x[p] S_y[q] S_z[r].
  Real S[3][5];
  for (int d = 0; d < 3; ++d)
    AxisSums(grid, M, d, S[d]);
  const Real K = M.scale * std::exp(M.b0) * grid.Weight();
  Real A[5][5];
  for (int a = 0; a < 5; ++a)
    for (int b = 0; b <= a; ++b) {
      Real s = 0.0;
      for (int ta = 0; ta < NumMono(a); ++ta)
        for (int tb = 0; tb < NumMono(b); ++tb) {
          int ea[3], eb[3];
          Mono(a, ta, ea);
          Mono(b, tb, eb);
          s += S[0][ea[0] + eb[0]] * S[1][ea[1] + eb[1]] * S[2][ea[2] + eb[2]];
        }
      A[a][b] = K * s;
      A[b][a] = K * s;
    }

  // Moment deficit in the xi basis.
  const Real dn = target.n - m.n;
  Real dnu[3], udnu = 0.0;
  for (int d = 0; d < 3; ++d) {
    dnu[d] = target.nu[d] - m.nu[d];
    udnu += u[d] * dnu[d];
  }
  const Real dnvv = target.nvv - (m.nvv[0] + m.nvv[1] + m.nvv[2]);
  Real rhs[5];
  rhs[0] = dn;
  for (int d = 0; d < 3; ++d)
    rhs[1 + d] = (dnu[d] - u[d] * dn) / c;
  rhs[4] = (dnvv - 2.0 * udnu + u2 * dn) / (c * c);
  if (!equilibrium_detail::CholeskySolve<5>(A, rhs)) {
    res.status = LomacResult::Status::singular;
    return res;
  }
  for (int a = 0; a < 5; ++a)
    res.c[a] = rhs[a];
  res.status = LomacResult::Status::applied;
  return res;
}

// Correction M P at node (ix, iy, iz).
KOKKOS_INLINE_FUNCTION Real LomacCorrection(const VelocityGrid &grid,
                                            const LomacResult &r, const int ix,
                                            const int iy, const int iz) {
  const int idx[3] = {ix, iy, iz};
  Real P = 0.0;
  for (int d = 0; d < 3; ++d)
    P += lomac_detail::AxisPoly(r.c, d, (grid.Node(d, idx[d]) - r.M.u[d]) * r.M.inv_c);
  return r.M(grid, ix, iy, iz) * P;
}

// Dense: f(n) += M P at every node. f(n) returns a reference to node n's value.
template <class F>
KOKKOS_INLINE_FUNCTION void LomacApplyDense(const VelocityGrid &grid,
                                            const LomacResult &r, const F &f) {
  for (int iz = 0; iz < grid.nv[2]; ++iz)
    for (int iy = 0; iy < grid.nv[1]; ++iy)
      for (int ix = 0; ix < grid.nv[0]; ++ix)
        f(grid.Flat(ix, iy, iz)) += LomacCorrection(grid, r, ix, iy, iz);
}

namespace TT {

// Tensor train: f <- f + M P (block sum, ranks + 2, no rounding). work holds
// TTLayout{n, f.L.rcap}.Size() reals. Returns false (f unchanged) if the ranks would
// exceed f.L.rcap.
template <class Data>
KOKKOS_INLINE_FUNCTION bool LomacApplyTT(const VelocityGrid &grid, const LomacResult &r,
                                         const TTRef<Data> &f, Real *work) {
  const int a1 = f.R1(), a2 = f.R2();
  const int r1 = a1 + 2, r2 = a2 + 2;
  if (r1 > f.L.rcap || r2 > f.L.rcap) return false;
  const auto out = MakeOutRef(PtrData{work}, f.L);
  out.SetRanks(r1, r2);
  const Maxwellian &M = r.M;
  const Real K = M.scale * std::exp(M.b0);
  for (int i = 0; i < grid.nv[0]; ++i) {
    for (int a = 0; a < a1; ++a)
      out.G1(i, a) = f.G1(i, a);
    const Real v = grid.Node(0, i);
    const Real mx = K * std::exp(M.AxisExponent(0, v));
    out.G1(i, a1) = mx * lomac_detail::AxisPoly(r.c, 0, (v - M.u[0]) * M.inv_c);
    out.G1(i, a1 + 1) = mx;
  }
  for (int b = 0; b < r2; ++b)
    for (int j = 0; j < grid.nv[1]; ++j) {
      const Real v = grid.Node(1, j);
      const Real my = std::exp(M.AxisExponent(1, v));
      const Real By = my * lomac_detail::AxisPoly(r.c, 1, (v - M.u[1]) * M.inv_c);
      for (int a = 0; a < r1; ++a) {
        Real g = 0.0;
        if (a < a1 && b < a2) {
          g = f.G2(a, j, b);
        } else if (a >= a1 && b >= a2) {
          const int p = a - a1, q = b - a2; // 2 x 2 block [[m_y, 0], [m_y B, m_y]]
          g = (p == q) ? my : ((p == 1 && q == 0) ? By : 0.0);
        }
        out.G2(a, j, b) = g;
      }
    }
  for (int k = 0; k < grid.nv[2]; ++k) {
    for (int b = 0; b < a2; ++b)
      out.G3(b, k) = f.G3(b, k);
    const Real v = grid.Node(2, k);
    const Real mz = std::exp(M.AxisExponent(2, v));
    out.G3(a2, k) = mz;
    out.G3(a2 + 1, k) = mz * lomac_detail::AxisPoly(r.c, 2, (v - M.u[2]) * M.inv_c);
  }
  CopyTT(MakeRef(PtrData{work}, f.L), f);
  return true;
}

} // namespace TT
} // namespace Kinetics

#endif // KINETICS_LOMAC_HPP_
