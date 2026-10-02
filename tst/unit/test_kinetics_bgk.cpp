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

// U4: exact BGK relaxation of one cell (src/kinetics/bgk.hpp).
//
//   * The discrete invariants (n, nu, energy) are unchanged to roundoff for any nu h.
//   * f - M decays exactly as exp(-nu h): checked node by node against the analytic
//     update, and through the semigroup property (two steps of h equal one of 2h).
//   * A rational (DIRK stage) step, c = a nu h / (1 + b nu h), satisfies the implicit
//     stage equation f = f* + a h nu (M[f] - f) and conserves, also for c > 1.
//   * The collision-frequency models return nu0, or p / mu(T).
//
// The initial f is a discrete bi-Maxwellian plus a skewed perturbation, so f - M has
// nonzero odd and even moments. Kernels run on device as in test_kinetics_equilibrium.

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

#include "kinetics/bgk.hpp"
#include "kinetics/equilibrium.hpp"
#include "kinetics/moments.hpp"
#include "kinetics/velocity_grid.hpp"

using parthenon::Real;

namespace {

namespace K = Kinetics;
using View = Kokkos::View<Real *>;

K::VelocityGrid CubeGrid(const int nv, const Real L) {
  const int n[3] = {nv, nv, nv};
  const Real lo[3] = {-L, -L, -L};
  const Real hi[3] = {L, L, L};
  return K::MakeVelocityGrid(n, lo, hi);
}

K::CollisionModel ConstantNu(const Real nu) {
  return K::CollisionModel{K::CollisionModel::Type::constant, nu, 1.0, 1.0, 0.5, 1.0};
}

// Discrete bi-Maxwellian (T_x = 2 T_y = 2 T_z, drift) times (1 + 0.1 xi_x^3 e^{...})
// so that f has a heat flux; all values stay positive.
void FillNonEquilibrium(const K::VelocityGrid grid, View f) {
  Kokkos::parallel_for(
      "fill", 1, KOKKOS_LAMBDA(const int) {
        K::EquilibriumTarget t{1.0, {0.3, -0.2, 0.1}, {1.5, 0.75, 0.75}};
        K::Maxwellian eq;
        K::SolveAnisotropicEquilibrium(grid, t, K::EquilibriumParams{}, eq);
        for (int iz = 0; iz < grid.nv[2]; ++iz)
          for (int iy = 0; iy < grid.nv[1]; ++iy)
            for (int ix = 0; ix < grid.nv[0]; ++ix) {
              const Real cx = grid.Node(0, ix) - 0.3;
              const Real skew = 1.0 + 0.1 * cx * cx * cx * std::exp(-0.25 * cx * cx);
              f(grid.Flat(ix, iy, iz)) = eq(grid, ix, iy, iz) * skew;
            }
      });
}

struct Relaxed {
  K::RawMoments before, after;
  K::RelaxResult result;
};

Relaxed RelaxOnDevice(const K::VelocityGrid grid, View f, const K::RelaxationStep step,
                      const K::CollisionModel model,
                      const K::EquilibriumParams eq_params = K::EquilibriumParams{}) {
  Kokkos::View<Relaxed> d_out("relaxed");
  Kokkos::parallel_for(
      "relax", 1, KOKKOS_LAMBDA(const int) {
        Relaxed out;
        out.before = K::ComputeRawMoments(grid, f);
        out.result = K::RelaxCell(grid, f, step, model, 1.0, eq_params);
        out.after = K::ComputeRawMoments(grid, f);
        d_out() = out;
      });
  auto h_out = Kokkos::create_mirror_view(d_out);
  Kokkos::deep_copy(h_out, d_out);
  return h_out();
}

Relaxed RelaxOnDevice(const K::VelocityGrid grid, View f, const Real h,
                      const K::CollisionModel model,
                      const K::EquilibriumParams eq_params = K::EquilibriumParams{}) {
  return RelaxOnDevice(grid, f, K::ExactRelaxation(h), model, eq_params);
}

// The equilibrium of f on the host side, sampled into a View.
void FillEquilibriumOf(const K::VelocityGrid grid, View f, View m) {
  Kokkos::parallel_for(
      "eq of f", 1, KOKKOS_LAMBDA(const int) {
        const auto t = K::IsotropicTarget(K::ComputeRawMoments(grid, f));
        K::Maxwellian eq;
        K::SolveEquilibrium(grid, t, K::EquilibriumParams{}, eq);
        K::FillEquilibrium(grid, eq, m);
      });
}

Real MaxRelInvariantChange(const K::RawMoments &a, const K::RawMoments &b) {
  // Momentum and energy scaled by n c, n c^2 with c^2 = theta ~ 1 here.
  Real e = std::abs(b.n - a.n) / a.n;
  Real ea = 0.0, eb = 0.0;
  for (int d = 0; d < 3; ++d) {
    e = std::max(e, std::abs(b.nu[d] - a.nu[d]) / a.n);
    ea += a.nvv[d];
    eb += b.nvv[d];
  }
  return std::max(e, std::abs(eb - ea) / ea);
}

} // namespace

TEST_CASE("U4: BGK relaxation conserves the discrete invariants", "[kinetics][bgk][U4]") {
  const auto grid = CubeGrid(24, 8.0);
  for (const Real nuh : {1.0e-3, 0.5, 3.0, 50.0}) {
    View f("f", grid.Size());
    FillNonEquilibrium(grid, f);
    const auto r = RelaxOnDevice(grid, f, nuh, ConstantNu(1.0));
    INFO("nu h = " << nuh);
    CHECK(r.result.eq.status == K::EquilibriumResult::Status::converged);
    CHECK(r.result.nu == 1.0);
    CHECK(MaxRelInvariantChange(r.before, r.after) < 1.0e-13);
  }

  // Negative control: relaxing toward the sampled continuous Maxwellian (Newton
  // disabled) on a coarse grid must fail the same bound.
  const auto coarse = CubeGrid(8, 8.0);
  View f("f", coarse.Size());
  FillNonEquilibrium(coarse, f);
  K::EquilibriumParams no_newton;
  no_newton.max_iter = 0;
  const auto r = RelaxOnDevice(coarse, f, 50.0, ConstantNu(1.0), no_newton);
  CHECK(r.result.eq.status == K::EquilibriumResult::Status::fallback);
  CHECK(MaxRelInvariantChange(r.before, r.after) > 1.0e-4);
}

TEST_CASE("U4: f - M decays exactly as exp(-nu h)", "[kinetics][bgk][U4]") {
  const auto grid = CubeGrid(24, 8.0);
  const Real nu = 2.0, h = 0.3;
  View f("f", grid.Size()), f0("f0", grid.Size()), m("m", grid.Size());
  FillNonEquilibrium(grid, f);
  Kokkos::deep_copy(f0, f);
  FillEquilibriumOf(grid, f0, m);
  RelaxOnDevice(grid, f, h, ConstantNu(nu));

  auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
  auto hf0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f0);
  auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m);
  const Real decay = std::exp(-nu * h);
  Real err = 0.0, dev0 = 0.0, fmax = 0.0;
  for (int n = 0; n < grid.Size(); ++n) {
    const Real expect = hm(n) + (hf0(n) - hm(n)) * decay;
    err = std::max(err, std::abs(hf(n) - expect));
    dev0 = std::max(dev0, std::abs(hf0(n) - hm(n)));
    fmax = std::max(fmax, hf0(n));
  }
  INFO("max |f - expected| " << err << ", max |f0 - M| " << dev0);
  CHECK(dev0 > 1.0e-3 * fmax); // the test state is far from equilibrium
  CHECK(err < 1.0e-15 * fmax);

  // Semigroup: two steps of h equal one of 2h (M is unchanged by relaxation).
  View g("g", grid.Size());
  Kokkos::deep_copy(g, f0);
  RelaxOnDevice(grid, g, 2.0 * h, ConstantNu(nu));
  RelaxOnDevice(grid, f, h, ConstantNu(nu));
  auto hg = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g);
  hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
  Real semi = 0.0;
  for (int n = 0; n < grid.Size(); ++n)
    semi = std::max(semi, std::abs(hf(n) - hg(n)));
  CHECK(semi < 1.0e-14 * fmax);
}

TEST_CASE("U4: a rational step solves the implicit BGK stage", "[kinetics][bgk][U4]") {
  // Backward Euler stage f = f* + a h nu (M[f] - f) is c = a nu h / (1 + a nu h); the
  // collapsed SL-DIRK2 stage c = (1-g) nu h / (1 + g nu h) exceeds 1 for nu h > 2.41.
  const auto grid = CubeGrid(24, 8.0);
  const Real g = 1.0 - 1.0 / std::sqrt(2.0);
  for (const Real nuh : {0.1, 3.0, 1.0e4}) {
    View f("f", grid.Size()), f0("f0", grid.Size()), m("m", grid.Size());
    FillNonEquilibrium(grid, f);
    Kokkos::deep_copy(f0, f);
    FillEquilibriumOf(grid, f0, m);
    const auto r =
        RelaxOnDevice(grid, f, K::RationalRelaxation(nuh, g, g), ConstantNu(1.0));
    INFO("nu h = " << nuh);
    CHECK(MaxRelInvariantChange(r.before, r.after) < 1.0e-13);

    // M[f] = M[f*]: recompute the equilibrium of the result and check the stage equation.
    View m1("m1", grid.Size());
    FillEquilibriumOf(grid, f, m1);
    auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
    auto hf0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f0);
    auto hm1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m1);
    Real res = 0.0, fmax = 0.0;
    for (int n = 0; n < grid.Size(); ++n) {
      res = std::max(res, std::abs(hf(n) - hf0(n) - g * nuh * (hm1(n) - hf(n))));
      fmax = std::max(fmax, hf0(n));
    }
    CHECK(res < 1.0e-12 * fmax * std::max(1.0, g * nuh));

    // Over-relaxing stage (c > 1 for nu h > 2.41) still conserves.
    Kokkos::deep_copy(f, f0);
    const auto o =
        RelaxOnDevice(grid, f, K::RationalRelaxation(nuh, 1.0 - g, g), ConstantNu(1.0));
    CHECK(MaxRelInvariantChange(o.before, o.after) < 1.0e-13);
  }
  CHECK(K::RationalRelaxation(1.0, 1.0 - g, g).Fraction(1.0e4) > 1.0);
  CHECK(K::ExactRelaxation(0.5).Fraction(2.0) == Catch::Approx(1.0 - std::exp(-1.0)));
}

TEST_CASE("U4: anisotropy decays as exp(-nu t)", "[kinetics][bgk][U4]") {
  const auto grid = CubeGrid(24, 8.0);
  const Real nu = 1.0;
  View f("f", grid.Size());
  FillNonEquilibrium(grid, f);
  auto aniso = [&](const K::RawMoments &m) {
    const auto t = K::AnisotropicTarget(m);
    return t.theta[0] - 0.5 * (t.theta[1] + t.theta[2]);
  };
  const auto r0 = RelaxOnDevice(grid, f, 0.0, ConstantNu(nu));
  const Real a0 = aniso(r0.after);
  CHECK(a0 > 0.5);
  Real t = 0.0;
  for (int step = 0; step < 10; ++step) {
    const auto r = RelaxOnDevice(grid, f, 0.25, ConstantNu(nu));
    t += 0.25;
    INFO("t = " << t);
    CHECK(std::abs(aniso(r.after) - a0 * std::exp(-nu * t)) < 1.0e-12 * a0);
  }
}

TEST_CASE("collision frequency models", "[kinetics][bgk]") {
  const K::CollisionModel c = ConstantNu(3.5);
  CHECK(c.Frequency(2.0, 10.0) == 3.5);
  // nu = n k_B T / (mu_ref (T / T_ref)^omega)
  const K::CollisionModel p{
      K::CollisionModel::Type::power_law, 0.0, 2.0e-4, 300.0, 0.75, 1.380649e-16};
  const Real n = 2.5e19, T = 600.0;
  const Real expect = n * 1.380649e-16 * T / (2.0e-4 * std::pow(2.0, 0.75));
  CHECK(p.Frequency(n, T) == Catch::Approx(expect).epsilon(1e-14));
}

TEST_CASE("relaxation leaves an empty cell unchanged", "[kinetics][bgk]") {
  const auto grid = CubeGrid(8, 8.0);
  View f("f", grid.Size()); // zero
  const auto r = RelaxOnDevice(grid, f, 1.0, ConstantNu(1.0));
  CHECK(r.result.eq.status == K::EquilibriumResult::Status::invalid);
  CHECK(r.after.n == 0.0);
}
