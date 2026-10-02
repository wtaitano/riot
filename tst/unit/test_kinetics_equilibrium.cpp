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
// This file was made in part with generative AI.

// Unit tests for the kinetics velocity grid, moment sums and discrete (Mieussens)
// equilibrium: src/kinetics/{velocity_grid,moments,equilibrium}.hpp.
//
//   * U1 VELOCITY QUADRATURE: the midpoint rule on a uniform box integrates a smooth,
//     well-contained Maxwellian spectrally, so the discrete moments of the sampled
//     continuous Maxwellian converge to the exact ones faster than any power of dv.
//   * U2 DISCRETE EQUILIBRIUM: the Newton solve returns a distribution whose discrete
//     moments match the target to the solver tolerance, in a few iterations when the
//     temperature is resolved, in both scaled and CGS units, for the isotropic (5) and
//     anisotropic (7) constraint sets. Unresolvable targets fall back to the sampled
//     Maxwellian and report it; invalid targets are flagged. Nothing crashes.
//
// As in test_reconstruction.cpp, the routines run ON DEVICE inside a single-iteration
// Kokkos loop and the POD results are checked on the host.

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

#include "kinetics/equilibrium.hpp"
#include "kinetics/moments.hpp"
#include "kinetics/velocity_grid.hpp"

using parthenon::Real;

namespace {

namespace K = Kinetics;
using Status = K::EquilibriumResult::Status;

template <class T, class Fill>
T RunOnDevice(Fill fill) {
  Kokkos::View<T> d_out("kinetics_result");
  Kokkos::parallel_for(
      "run kinetics", 1, KOKKOS_LAMBDA(const int) {
        T out{};
        fill(out);
        d_out() = out;
      });
  auto h_out = Kokkos::create_mirror_view(d_out);
  Kokkos::deep_copy(h_out, d_out);
  return h_out();
}

// Cubic box [-L, L]^3 with nv nodes per axis.
K::VelocityGrid CubeGrid(const int nv, const Real L) {
  const int n[3] = {nv, nv, nv};
  const Real lo[3] = {-L, -L, -L};
  const Real hi[3] = {L, L, L};
  return K::MakeVelocityGrid(n, lo, hi);
}

K::EquilibriumTarget Target(const Real n, const Real ux, const Real uy, const Real uz,
                            const Real tx, const Real ty, const Real tz) {
  return K::EquilibriumTarget{n, {ux, uy, uz}, {tx, ty, tz}};
}

// Solve result plus the moments recomputed from the returned distribution.
struct SolveCheck {
  K::EquilibriumResult result;
  K::RawMoments m;
  Real min_f;
};

// Error of recomputed moments against the target. Momentum and energy are scaled by n c
// and n c^2 (c^2 = mean theta) so the errors are dimensionless and a zero target
// velocity is handled. `rel` is |dU| / |U| over the constraint vector U = (n, nu, E)
// (E summed, or per axis), the U2 criterion of S0_DESIGN.md section 5.
struct MomentErrors {
  Real n, u, theta, rel;
};

MomentErrors Errors(const K::RawMoments &m, const K::EquilibriumTarget &t,
                    const bool per_axis) {
  const Real c2 = (t.theta[0] + t.theta[1] + t.theta[2]) / 3.0;
  const Real c = std::sqrt(c2);
  MomentErrors e{std::abs(m.n - t.n) / t.n, 0.0, 0.0, 0.0};
  Real err2 = e.n * e.n, norm2 = 1.0;
  Real e_sum = 0.0, t_sum = 0.0;
  for (int d = 0; d < 3; ++d) {
    const Real du = (m.nu[d] - t.n * t.u[d]) / (t.n * c);
    e.u = std::max(e.u, std::abs(du));
    err2 += du * du;
    norm2 += (t.u[d] / c) * (t.u[d] / c);
    const Real target_vv = t.n * (t.theta[d] + t.u[d] * t.u[d]);
    const Real dvv = (m.nvv[d] - target_vv) / (t.n * c2);
    if (per_axis) {
      e.theta = std::max(e.theta, std::abs(dvv));
      err2 += dvv * dvv;
      norm2 += (target_vv / (t.n * c2)) * (target_vv / (t.n * c2));
    }
    e_sum += m.nvv[d];
    t_sum += target_vv;
  }
  if (!per_axis) {
    e.theta = std::abs(e_sum - t_sum) / (t.n * c2);
    err2 += e.theta * e.theta;
    norm2 += (t_sum / (t.n * c2)) * (t_sum / (t.n * c2));
  }
  e.rel = std::sqrt(err2 / norm2);
  return e;
}

// U2 bound on the recomputed moments. The solver stops at |F| / |T| <= 1e-13 in the
// scaled variables; recomputing the sums on the unscaled nodes, in a different order,
// adds summation roundoff on top, hence the factor 2.
constexpr Real kU2Recomputed = 2.0e-13;

template <bool Aniso>
SolveCheck SolveOnDevice(const K::VelocityGrid grid, const K::EquilibriumTarget target,
                         const K::EquilibriumParams params = K::EquilibriumParams{}) {
  return RunOnDevice<SolveCheck>(KOKKOS_LAMBDA(SolveCheck & out) {
    K::Maxwellian eq;
    if constexpr (Aniso) {
      out.result = K::SolveAnisotropicEquilibrium(grid, target, params, eq);
    } else {
      out.result = K::SolveEquilibrium(grid, target, params, eq);
    }
    auto f = [&](const int n) {
      int ix, iy, iz;
      grid.Unflatten(n, ix, iy, iz);
      return eq(grid, ix, iy, iz);
    };
    out.m = K::ComputeRawMoments(grid, f);
    out.min_f = 1.0e300;
    for (int n = 0; n < grid.Size(); ++n)
      out.min_f = std::min(out.min_f, f(n));
  });
}

// Raw and central moments of one distribution. At namespace scope because nvcc rejects
// a function-local type as the template argument of a function holding a device lambda.
struct AllMoments {
  K::RawMoments raw;
  K::CentralMoments central;
};

} // namespace

//----------------------------------------------------------------------------------------
// Velocity grid
//----------------------------------------------------------------------------------------

TEST_CASE("velocity grid nodes, weight and flat index", "[kinetics][velocity_grid]") {
  const int n[3] = {4, 3, 2};
  const Real lo[3] = {-2.0, 0.0, -1.0};
  const Real hi[3] = {2.0, 3.0, 1.0};
  const auto grid = K::MakeVelocityGrid(n, lo, hi);

  CHECK(grid.Size() == 24);
  CHECK(grid.Weight() == Catch::Approx(1.0));
  CHECK(grid.Node(0, 0) == Catch::Approx(-1.5));
  CHECK(grid.Node(0, 3) == Catch::Approx(1.5));
  CHECK(grid.Node(1, 0) == Catch::Approx(0.5));
  CHECK(grid.Node(2, 1) == Catch::Approx(0.5));
  CHECK(grid.MaxNodeSpeed(0) == Catch::Approx(1.5));
  CHECK(grid.MaxNodeSpeed(1) == Catch::Approx(2.5));

  // Flat and Unflatten are inverse, x fastest.
  bool round_trip = true;
  for (int iz = 0; iz < 2; ++iz)
    for (int iy = 0; iy < 3; ++iy)
      for (int ix = 0; ix < 4; ++ix) {
        int jx, jy, jz;
        grid.Unflatten(grid.Flat(ix, iy, iz), jx, jy, jz);
        round_trip = round_trip && jx == ix && jy == iy && jz == iz;
      }
  CHECK(round_trip);
  CHECK(grid.Flat(1, 0, 0) == 1);
  CHECK(grid.Flat(0, 1, 0) == 4);
  CHECK(grid.Flat(0, 0, 1) == 12);
}

//----------------------------------------------------------------------------------------
// U1: quadrature of the continuous Maxwellian
//----------------------------------------------------------------------------------------

TEST_CASE("U1: midpoint moments of a Maxwellian converge spectrally",
          "[kinetics][moments][U1]") {
  // Unit thermal speed, box [-8, 8], drift small compared with the box.
  const auto target = Target(1.0, 0.3, -0.2, 0.1, 1.0, 1.0, 1.0);
  const int nvs[5] = {8, 12, 16, 24, 32};
  MomentErrors err[5];
  for (int k = 0; k < 5; ++k) {
    const auto grid = CubeGrid(nvs[k], 8.0);
    const auto m = RunOnDevice<K::RawMoments>(KOKKOS_LAMBDA(K::RawMoments & out) {
      const auto eq = K::ContinuousMaxwellian(target);
      out = K::ComputeRawMoments(grid, [&](const int n) {
        int ix, iy, iz;
        grid.Unflatten(n, ix, iy, iz);
        return eq(grid, ix, iy, iz);
      });
    });
    err[k] = Errors(m, target, true);
    INFO("nv = " << nvs[k] << ": n " << err[k].n << ", u " << err[k].u << ", theta "
                 << err[k].theta);
    CHECK(std::isfinite(err[k].n + err[k].u + err[k].theta));
  }
  // Faster than any power: each refinement up to nv = 16 gains far more than the
  // factor (nv_{k+1}/nv_k)^2 of a second-order rule.
  for (int k = 0; k < 2; ++k) {
    CHECK(err[k + 1].n < 1.0e-2 * err[k].n);
    CHECK(err[k + 1].theta < 1.0e-2 * err[k].theta);
  }
  // At nv = 24 and 32 only roundoff remains.
  for (int k = 3; k < 5; ++k) {
    CHECK(err[k].n < 1.0e-13);
    CHECK(err[k].u < 1.0e-12);
    CHECK(err[k].theta < 1.0e-11);
  }
}

//----------------------------------------------------------------------------------------
// U2: discrete equilibrium
//----------------------------------------------------------------------------------------

TEST_CASE("U2: isotropic discrete equilibrium matches the moments exactly",
          "[kinetics][equilibrium][U2]") {
  const auto target = Target(1.0, 0.3, -0.2, 0.1, 1.0, 1.0, 1.0);
  for (const int nv : {12, 16, 24, 32}) {
    const auto grid = CubeGrid(nv, 8.0);
    const auto s = SolveOnDevice<false>(grid, target);
    const auto e = Errors(s.m, target, false);
    INFO("nv = " << nv << ": iterations " << s.result.iterations << ", residual "
                 << s.result.residual << " (guess " << s.result.residual0 << ")");
    CHECK(s.result.status == Status::converged);
    CHECK(s.result.residual <= 1.0e-13);
    CHECK(s.result.iterations <= 5);
    CHECK(e.rel < kU2Recomputed);
    CHECK(s.min_f > 0.0);
  }
}

TEST_CASE("U2: discrete equilibrium corrects an under-resolved Maxwellian",
          "[kinetics][equilibrium][U2]") {
  // At nv = 8 the sampled Maxwellian is off by ~1e-2; the discrete one is exact.
  const auto target = Target(1.0, 0.3, -0.2, 0.1, 1.0, 1.0, 1.0);
  const auto s = SolveOnDevice<false>(CubeGrid(8, 8.0), target);
  const auto e = Errors(s.m, target, false);
  CHECK(s.result.residual0 > 1.0e-3);
  CHECK(s.result.status == Status::converged);
  CHECK(e.rel < kU2Recomputed);

  // Negative control for the comparator: the sampled Maxwellian on the same grid must
  // fail the bound the discrete equilibrium passes.
  const auto grid = CubeGrid(8, 8.0);
  const auto m = RunOnDevice<K::RawMoments>(KOKKOS_LAMBDA(K::RawMoments & out) {
    const auto eq = K::ContinuousMaxwellian(target);
    out = K::ComputeRawMoments(grid, [&](const int n) {
      int ix, iy, iz;
      grid.Unflatten(n, ix, iy, iz);
      return eq(grid, ix, iy, iz);
    });
  });
  CHECK(Errors(m, target, false).rel > 1.0e-3);
}

TEST_CASE("U2: anisotropic discrete equilibrium matches per-axis temperatures",
          "[kinetics][equilibrium][U2]") {
  const auto target = Target(1.0, 0.3, -0.2, 0.1, 1.5, 0.75, 0.75);
  for (const int nv : {16, 24, 32}) {
    const auto s = SolveOnDevice<true>(CubeGrid(nv, 8.0), target);
    const auto e = Errors(s.m, target, true);
    INFO("nv = " << nv << ": iterations " << s.result.iterations << ", residual "
                 << s.result.residual);
    CHECK(s.result.status == Status::converged);
    CHECK(s.result.iterations <= 5);
    CHECK(e.rel < kU2Recomputed);
  }
}

TEST_CASE("U2: discrete equilibrium in CGS units on a non-cubic box",
          "[kinetics][equilibrium][U2]") {
  // Argon-like gas at 300 K: theta = k_B T / m ~ 6.2e8 (cm/s)^2, c ~ 2.5e4 cm/s.
  const Real theta = 6.24e8;
  const Real c = std::sqrt(theta);
  const auto target = Target(2.7e19, 0.4 * c, 0.0, -0.1 * c, theta, theta, theta);
  const int n[3] = {28, 20, 24};
  const Real lo[3] = {-7.0 * c, -6.0 * c, -7.5 * c};
  const Real hi[3] = {8.0 * c, 6.0 * c, 7.0 * c};
  const auto s = SolveOnDevice<false>(K::MakeVelocityGrid(n, lo, hi), target);
  const auto e = Errors(s.m, target, false);
  CHECK(s.result.status == Status::converged);
  CHECK(s.result.iterations <= 5);
  CHECK(e.rel < kU2Recomputed);
}

TEST_CASE("U2: unresolvable targets fall back without crashing",
          "[kinetics][equilibrium][U2]") {
  const auto grid = CubeGrid(16, 8.0);
  const Real dv = grid.dv[0];

  SECTION("temperature far below the node spacing") {
    const Real th = (0.05 * dv) * (0.05 * dv);
    const auto s = SolveOnDevice<false>(grid, Target(1.0, 0.1, 0.0, 0.0, th, th, th));
    CHECK(s.result.status == Status::fallback);
    CHECK(std::isfinite(s.m.n));
    CHECK(s.min_f >= 0.0);
  }
  SECTION("mean velocity outside the box") {
    const auto s = SolveOnDevice<false>(grid, Target(1.0, 10.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    CHECK(s.result.status == Status::fallback);
    CHECK(std::isfinite(s.m.n));
    CHECK(s.min_f >= 0.0);
  }
  SECTION("iteration limit reached") {
    K::EquilibriumParams params;
    params.max_iter = 0;
    const auto s = SolveOnDevice<false>(
        CubeGrid(8, 8.0), Target(1.0, 0.3, 0.0, 0.0, 1.0, 1.0, 1.0), params);
    CHECK(s.result.status == Status::fallback);
    CHECK(s.result.residual == s.result.residual0);
  }
}

TEST_CASE("U2: invalid targets are flagged", "[kinetics][equilibrium][U2]") {
  const auto grid = CubeGrid(16, 8.0);
  for (const auto &t : {Target(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                        Target(1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0),
                        Target(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0)}) {
    const auto s = SolveOnDevice<true>(grid, t);
    CHECK(s.result.status == Status::invalid);
    CHECK(s.m.n == 0.0);
  }
}

TEST_CASE("moments recover the target from a filled equilibrium", "[kinetics][moments]") {
  // FillEquilibrium writes through a View; central moments of an isotropic
  // equilibrium are diagonal pressure n theta with zero heat flux.
  const auto grid = CubeGrid(24, 8.0);
  const auto target = Target(1.0, 0.3, -0.2, 0.1, 1.0, 1.0, 1.0);
  Kokkos::View<Real *> f("f", grid.Size());
  const auto out = RunOnDevice<AllMoments>(KOKKOS_LAMBDA(AllMoments & o) {
    K::Maxwellian eq;
    K::SolveEquilibrium(grid, target, K::EquilibriumParams{}, eq);
    K::FillEquilibrium(grid, eq, f);
    o.raw = K::ComputeRawMoments(grid, f);
    const auto t = K::IsotropicTarget(o.raw);
    o.central = K::ComputeCentralMoments(grid, f, t.u);
  });
  const auto t = K::IsotropicTarget(out.raw);
  CHECK(t.n == Catch::Approx(1.0).epsilon(1e-13));
  CHECK(t.theta[0] == Catch::Approx(1.0).epsilon(1e-12));
  for (int d = 0; d < 3; ++d) {
    CHECK(out.central.stress[d] == Catch::Approx(1.0).epsilon(1e-6));
    CHECK(std::abs(out.central.stress[3 + d]) < 1.0e-10);
    CHECK(std::abs(out.central.heat[d]) < 1.0e-10);
  }
}
