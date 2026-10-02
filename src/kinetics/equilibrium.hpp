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
#ifndef KINETICS_EQUILIBRIUM_HPP_
#define KINETICS_EQUILIBRIUM_HPP_
// This file was made in part with generative AI.

// Discrete equilibrium of Mieussens type on the velocity grid.
//
// Given target moments (number density n, mean velocity u and temperatures theta_d =
// k_B T_d / m), find the distribution
//
//   M(v) = (n / c^3) exp(b0 + sum_d (b_d xi_d + g_d xi_d^2)),  xi_d = (v_d - u_d) / c,
//
// whose DISCRETE moments on the grid match the targets to roundoff. The solve is done in
// the scaled variable xi with c^2 = mean(theta_d), so the Newton system is O(1) whatever
// the units. The exponent is a sum over axes, so M is separable (rank 1): every grid sum
// factors into three 1D sums and one Newton iteration costs O(nv_x + nv_y + nv_z).
//
// Two constraint sets share one implementation:
//   * 5 constraints (isotropic, g_x = g_y = g_z): mass, momentum and total energy. This
//     is the BGK equilibrium.
//   * 7 constraints: mass, momentum and the three diagonal second moments. This builds a
//     discrete bi-Maxwellian with exact moments, used for initial conditions.
//
// If Newton fails (target velocity near or outside the box, or temperature unresolved
// by the grid), the result keeps the coefficients of the sampled continuous Maxwellian
// and reports Status::fallback together with that state's moment error.

#include <cmath>

#include <parthenon/package.hpp>

#include "kinetics/velocity_grid.hpp"

using namespace parthenon::package::prelude;

namespace Kinetics {

// Target state of the equilibrium solve.
struct EquilibriumTarget {
  Real n;        // number density
  Real u[3];     // mean velocity
  Real theta[3]; // k_B T_d / m per axis; all equal for the isotropic solve
};

// Separable equilibrium on the velocity grid.
struct Maxwellian {
  Real scale; // n / c^3; zero for an invalid target
  Real u[3];
  Real inv_c;
  Real b0;
  Real b[3];
  Real g[3];

  KOKKOS_INLINE_FUNCTION Real AxisExponent(const int d, const Real v) const {
    const Real xi = (v - u[d]) * inv_c;
    return (b[d] + g[d] * xi) * xi;
  }

  KOKKOS_INLINE_FUNCTION Real operator()(const VelocityGrid &grid, const int ix,
                                         const int iy, const int iz) const {
    return scale * std::exp(b0 + AxisExponent(0, grid.Node(0, ix)) +
                            AxisExponent(1, grid.Node(1, iy)) +
                            AxisExponent(2, grid.Node(2, iz)));
  }
};

struct EquilibriumParams {
  Real tol = 1.0e-13; // on the scaled moment residual, |F| / |target|
  int max_iter = 20;
};

struct EquilibriumResult {
  enum class Status { converged, fallback, invalid };
  Status status;
  int iterations;
  Real residual;  // scaled moment residual of the returned state
  Real residual0; // scaled moment residual of the sampled continuous Maxwellian
};

namespace equilibrium_detail {

// Number of parameters in the full (anisotropic) exponent: b0, b_d, g_d.
constexpr int NP = 7;

// Moment multi-indices (powers of xi_x, xi_y, xi_z) of the 7 anisotropic constraints.
KOKKOS_INLINE_FUNCTION int Power(const int a, const int d) {
  // a = 0: (0,0,0); a = 1..3: xi_d; a = 4..6: xi_d^2.
  if (a == 0) return 0;
  if (a <= 3) return (a - 1 == d) ? 1 : 0;
  return (a - 4 == d) ? 2 : 0;
}

// Residual F_a = sum_xi xi^{e_a} mhat dxi^3 - T_a and Jacobian J_ab = sum xi^{e_a+e_b}
// mhat dxi^3 for the full 7-parameter exponent. Returns false on overflow.
KOKKOS_INLINE_FUNCTION bool ResidualJacobian7(const VelocityGrid &grid, const Real u[3],
                                              const Real inv_c, const Real p[NP],
                                              const Real target[NP], Real F[NP],
                                              Real J[NP][NP]) {
  // 1D sums S[d][k] = sum_i xi^k exp(b_d xi + g_d xi^2) dxi, k = 0..4.
  Real S[3][5];
  for (int d = 0; d < 3; ++d) {
    const Real dxi = grid.dv[d] * inv_c;
    for (int k = 0; k < 5; ++k)
      S[d][k] = 0.0;
    for (int i = 0; i < grid.nv[d]; ++i) {
      const Real xi = (grid.Node(d, i) - u[d]) * inv_c;
      const Real w = std::exp((p[1 + d] + p[4 + d] * xi) * xi) * dxi;
      Real xk = w;
      for (int k = 0; k < 5; ++k) {
        S[d][k] += xk;
        xk *= xi;
      }
    }
  }
  const Real e0 = std::exp(p[0]);
  for (int a = 0; a < NP; ++a) {
    for (int c = a; c < NP; ++c) {
      Real m = e0;
      for (int d = 0; d < 3; ++d)
        m *= S[d][Power(a, d) + Power(c, d)];
      J[a][c] = m;
      J[c][a] = m;
    }
  }
  bool finite = true;
  for (int a = 0; a < NP; ++a) {
    // Row 0 of J holds the moments themselves (e_0 = 0).
    F[a] = J[0][a] - target[a];
    finite = finite && (std::abs(F[a]) < 1.0e300);
  }
  return finite;
}

// In-place Cholesky solve of the SPD system A x = rhs (rhs is overwritten by x).
// Returns false if A is not numerically positive definite.
template <int N>
KOKKOS_INLINE_FUNCTION bool CholeskySolve(Real A[N][N], Real rhs[N]) {
  for (int j = 0; j < N; ++j) {
    Real diag = A[j][j];
    for (int k = 0; k < j; ++k)
      diag -= A[j][k] * A[j][k];
    if (!(diag > 0.0)) return false;
    A[j][j] = std::sqrt(diag);
    for (int i = j + 1; i < N; ++i) {
      Real s = A[i][j];
      for (int k = 0; k < j; ++k)
        s -= A[i][k] * A[j][k];
      A[i][j] = s / A[j][j];
    }
  }
  for (int i = 0; i < N; ++i) {
    Real s = rhs[i];
    for (int k = 0; k < i; ++k)
      s -= A[i][k] * rhs[k];
    rhs[i] = s / A[i][i];
  }
  for (int i = N - 1; i >= 0; --i) {
    Real s = rhs[i];
    for (int k = i + 1; k < N; ++k)
      s -= A[k][i] * rhs[k];
    rhs[i] = s / A[i][i];
  }
  return true;
}

// Reduced system for NC constraints. NC = 7 is the full system; NC = 5 ties the three
// quadratic coefficients together (q[4] = g) and sums the three energy rows.
template <int NC>
KOKKOS_INLINE_FUNCTION void Expand(const Real q[NC], Real p[NP]) {
  for (int a = 0; a < 4; ++a)
    p[a] = q[a];
  for (int d = 0; d < 3; ++d) {
    if constexpr (NC == 7) {
      p[4 + d] = q[4 + d];
    } else {
      p[4 + d] = q[4];
    }
  }
}

template <int NC>
KOKKOS_INLINE_FUNCTION void Reduce(const Real F7[NP], const Real J7[NP][NP], Real F[NC],
                                   Real J[NC][NC]) {
  if constexpr (NC == 7) {
    for (int a = 0; a < NP; ++a) {
      F[a] = F7[a];
      for (int c = 0; c < NP; ++c)
        J[a][c] = J7[a][c];
    }
  } else {
    for (int a = 0; a < 4; ++a) {
      F[a] = F7[a];
      for (int c = 0; c < 4; ++c)
        J[a][c] = J7[a][c];
      J[a][4] = J7[a][4] + J7[a][5] + J7[a][6];
      J[4][a] = J[a][4];
    }
    F[4] = F7[4] + F7[5] + F7[6];
    J[4][4] = 0.0;
    for (int d = 4; d < NP; ++d)
      for (int e = 4; e < NP; ++e)
        J[4][4] += J7[d][e];
  }
}

template <int NC>
KOKKOS_INLINE_FUNCTION Real Norm(const Real F[NC]) {
  Real s = 0.0;
  for (int a = 0; a < NC; ++a)
    s += F[a] * F[a];
  return std::sqrt(s);
}

template <int NC>
KOKKOS_INLINE_FUNCTION EquilibriumResult Solve(const VelocityGrid &grid,
                                               const EquilibriumTarget &target,
                                               const EquilibriumParams &params,
                                               Maxwellian &eq) {
  EquilibriumResult result{EquilibriumResult::Status::invalid, 0, 0.0, 0.0};
  for (int d = 0; d < 3; ++d)
    eq.u[d] = target.u[d];
  const Real c2 = (target.theta[0] + target.theta[1] + target.theta[2]) / 3.0;
  const bool valid = (target.n > 0.0) && (target.theta[0] > 0.0) &&
                     (target.theta[1] > 0.0) && (target.theta[2] > 0.0);
  if (!valid) {
    eq.scale = 0.0;
    eq.inv_c = 0.0;
    eq.b0 = 0.0;
    for (int d = 0; d < 3; ++d) {
      eq.b[d] = 0.0;
      eq.g[d] = 0.0;
    }
    return result;
  }
  const Real c = std::sqrt(c2);
  eq.inv_c = 1.0 / c;
  eq.scale = target.n / (c2 * c);

  // Scaled targets and the continuous-Maxwellian initial guess.
  Real T7[NP] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Real p[NP] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int d = 0; d < 3; ++d) {
    const Real th = target.theta[d] / c2;
    T7[4 + d] = th;
    p[0] -= 0.5 * std::log(2.0 * M_PI * th);
    p[4 + d] = -0.5 / th;
  }
  Real q[NC], q_guess[NC], Tn[NC];
  for (int a = 0; a < 4; ++a) {
    q[a] = p[a];
    Tn[a] = T7[a];
  }
  if constexpr (NC == 7) {
    for (int d = 0; d < 3; ++d) {
      q[4 + d] = p[4 + d];
      Tn[4 + d] = T7[4 + d];
    }
  } else {
    q[4] = p[4];
    Tn[4] = 3.0; // sum_d theta_d / c^2
  }
  for (int a = 0; a < NC; ++a)
    q_guess[a] = q[a];
  const Real tnorm = Norm<NC>(Tn);

  Real F7[NP], J7[NP][NP], F[NC], J[NC][NC];
  bool ok = ResidualJacobian7(grid, eq.u, eq.inv_c, p, T7, F7, J7);
  Reduce<NC>(F7, J7, F, J);
  Real r = ok ? Norm<NC>(F) / tnorm : 1.0e300;
  result.residual0 = r;

  bool failed = !ok;
  int it = 0;
  while (!failed && r > params.tol && it < params.max_iter) {
    Real dq[NC];
    for (int a = 0; a < NC; ++a)
      dq[a] = -F[a];
    if (!CholeskySolve<NC>(J, dq)) {
      failed = true;
      break;
    }
    // Backtracking line search on the residual norm (Armijo with halving). J is
    // factored and r holds the current residual, so each trial reuses F7/J7 and F/J;
    // after an accepted step they hold the new point.
    Real lam = 1.0;
    bool accepted = false;
    while (lam > 1.0e-10) {
      Real qt[NC];
      for (int a = 0; a < NC; ++a)
        qt[a] = q[a] + lam * dq[a];
      Expand<NC>(qt, p);
      const bool okt = ResidualJacobian7(grid, eq.u, eq.inv_c, p, T7, F7, J7);
      Reduce<NC>(F7, J7, F, J);
      const Real rt = okt ? Norm<NC>(F) / tnorm : 1.0e300;
      if (okt && rt < (1.0 - 1.0e-4 * lam) * r) {
        for (int a = 0; a < NC; ++a)
          q[a] = qt[a];
        r = rt;
        accepted = true;
        break;
      }
      lam *= 0.5;
    }
    if (!accepted) {
      failed = true;
      break;
    }
    ++it;
  }

  result.iterations = it;
  if (!failed && r <= params.tol) {
    result.status = EquilibriumResult::Status::converged;
    result.residual = r;
  } else {
    // Keep the sampled continuous Maxwellian.
    result.status = EquilibriumResult::Status::fallback;
    result.residual = result.residual0;
    for (int a = 0; a < NC; ++a)
      q[a] = q_guess[a];
  }
  Expand<NC>(q, p);
  eq.b0 = p[0];
  for (int d = 0; d < 3; ++d) {
    eq.b[d] = p[1 + d];
    eq.g[d] = p[4 + d];
  }
  return result;
}

} // namespace equilibrium_detail

// Isotropic discrete equilibrium (5 constraints). Uses the mean of target.theta.
KOKKOS_INLINE_FUNCTION EquilibriumResult SolveEquilibrium(const VelocityGrid &grid,
                                                          EquilibriumTarget target,
                                                          const EquilibriumParams &params,
                                                          Maxwellian &eq) {
  const Real th = (target.theta[0] + target.theta[1] + target.theta[2]) / 3.0;
  for (int d = 0; d < 3; ++d)
    target.theta[d] = th;
  return equilibrium_detail::Solve<5>(grid, target, params, eq);
}

// Anisotropic discrete equilibrium (7 constraints: n, nu, diagonal second moments).
KOKKOS_INLINE_FUNCTION EquilibriumResult
SolveAnisotropicEquilibrium(const VelocityGrid &grid, const EquilibriumTarget &target,
                            const EquilibriumParams &params, Maxwellian &eq) {
  return equilibrium_detail::Solve<7>(grid, target, params, eq);
}

// Sampled continuous Maxwellian with the target's n, u and per-axis temperatures. This
// is the Newton initial guess and the fallback state; its discrete moments are only
// approximately the targets.
KOKKOS_INLINE_FUNCTION Maxwellian ContinuousMaxwellian(const EquilibriumTarget &target) {
  Maxwellian eq;
  const Real c2 = (target.theta[0] + target.theta[1] + target.theta[2]) / 3.0;
  const Real c = std::sqrt(c2);
  eq.scale = target.n / (c2 * c);
  eq.inv_c = 1.0 / c;
  eq.b0 = 0.0;
  for (int d = 0; d < 3; ++d) {
    const Real th = target.theta[d] / c2;
    eq.u[d] = target.u[d];
    eq.b[d] = 0.0;
    eq.g[d] = -0.5 / th;
    eq.b0 -= 0.5 * std::log(2.0 * M_PI * th);
  }
  return eq;
}

} // namespace Kinetics

#endif // KINETICS_EQUILIBRIUM_HPP_
