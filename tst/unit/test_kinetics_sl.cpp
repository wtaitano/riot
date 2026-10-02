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

// U3: semi-Lagrangian streaming weights and stencil (src/kinetics/semi_lagrangian.hpp).
//
//   * Under the |s| <= 1 cap, linear SL equals first-order upwind and unlimited
//     quadratic SL equals Lax-Wendroff, to roundoff, for shifts of both signs.
//   * One SL step of a 1D periodic row conserves sum_i f_i to roundoff (unlimited), and
//     exactly translates the data for s = +-1.
//   * The limiter keeps every value inside the range of the bracketing cells; quadratic
//     unlimited SL creates new extrema at a jump (negative control for the bound check).
//   * 2D tensor product: separable data advect as the product of the 1D results.

#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

#include "kinetics/semi_lagrangian.hpp"

using parthenon::Real;

namespace {

namespace K = Kinetics;

// One periodic 1D SL step on the host (the stencil function is host+device inline).
std::vector<Real> Step1D(const std::vector<Real> &f, const Real s, const int order,
                         const bool limiter) {
  const int n = f.size();
  std::vector<Real> out(n);
  const K::SLParams sl{order, limiter};
  const Real sh[3] = {s, 0.0, 0.0};
  for (int i = 0; i < n; ++i) {
    out[i] = K::SLInterpolate(sl, 1, sh,
                              [&](int a, int, int) { return f[((i + a) % n + n) % n]; });
  }
  return out;
}

std::vector<Real> Upwind(const std::vector<Real> &f, const Real s) {
  const int n = f.size();
  std::vector<Real> out(n);
  for (int i = 0; i < n; ++i) {
    const Real fm = f[(i - 1 + n) % n], f0 = f[i], fp = f[(i + 1) % n];
    out[i] = (s > 0.0) ? f0 - s * (f0 - fm) : f0 - s * (fp - f0);
  }
  return out;
}

std::vector<Real> LaxWendroff(const std::vector<Real> &f, const Real s) {
  const int n = f.size();
  std::vector<Real> out(n);
  for (int i = 0; i < n; ++i) {
    const Real fm = f[(i - 1 + n) % n], f0 = f[i], fp = f[(i + 1) % n];
    // Flux form: F_{i+1/2} = (s/2)(f_i + f_{i+1}) - (s^2/2)(f_{i+1} - f_i)
    const Real Fp = 0.5 * s * (f0 + fp) - 0.5 * s * s * (fp - f0);
    const Real Fm = 0.5 * s * (fm + f0) - 0.5 * s * s * (f0 - fm);
    out[i] = f0 - (Fp - Fm);
  }
  return out;
}

std::vector<Real> Smooth(const int n) {
  std::vector<Real> f(n);
  for (int i = 0; i < n; ++i)
    f[i] = 1.0 + 0.5 * std::sin(2.0 * M_PI * (i + 0.5) / n) +
           0.2 * std::cos(6.0 * M_PI * (i + 0.5) / n);
  return f;
}

std::vector<Real> Step(const int n) {
  std::vector<Real> f(n, 0.1);
  for (int i = n / 4; i < 3 * n / 4; ++i)
    f[i] = 1.0;
  return f;
}

Real MaxDiff(const std::vector<Real> &a, const std::vector<Real> &b) {
  Real m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

Real Sum(const std::vector<Real> &a) {
  Real s = 0.0;
  for (const Real x : a)
    s += x;
  return s;
}

const Real kShifts[] = {-1.0, -0.73, -0.5, -0.1, 0.0, 0.1, 0.37, 0.5, 0.9, 1.0};

} // namespace

TEST_CASE("U3: weights are a partition of unity and reproduce linear data",
          "[kinetics][sl][U3]") {
  for (const int order : {1, 2}) {
    for (const Real s : kShifts) {
      Real w[3];
      K::SLWeights(order, s, w);
      INFO("order " << order << ", s = " << s);
      CHECK(w[0] + w[1] + w[2] == Catch::Approx(1.0).epsilon(1e-15));
      // Departure point x - s: sum_a w_a * a = -s.
      CHECK(w[2] - w[0] == Catch::Approx(-s).margin(1e-15));
    }
  }
}

TEST_CASE("U3: linear SL is upwind and quadratic SL is Lax-Wendroff",
          "[kinetics][sl][U3]") {
  const auto f = Smooth(32);
  for (const Real s : kShifts) {
    INFO("s = " << s);
    CHECK(MaxDiff(Step1D(f, s, 1, false), Upwind(f, s)) < 1.0e-15);
    CHECK(MaxDiff(Step1D(f, s, 2, false), LaxWendroff(f, s)) < 1.0e-15);
  }
}

TEST_CASE("U3: unlimited SL conserves on a periodic row", "[kinetics][sl][U3]") {
  for (const auto &f : {Smooth(37), Step(40)}) {
    const Real s0 = Sum(f);
    for (const int order : {1, 2}) {
      for (const Real s : kShifts) {
        std::vector<Real> g = f;
        for (int it = 0; it < 25; ++it)
          g = Step1D(g, s, order, false);
        INFO("order " << order << ", s = " << s);
        CHECK(std::abs(Sum(g) - s0) < 1.0e-13 * s0);
      }
    }
  }
}

TEST_CASE("U3: shifts of one cell translate exactly", "[kinetics][sl][U3]") {
  const auto f = Smooth(16);
  const int n = f.size();
  for (const int order : {1, 2}) {
    const auto right = Step1D(f, 1.0, order, false);
    const auto left = Step1D(f, -1.0, order, false);
    Real err = 0.0;
    for (int i = 0; i < n; ++i) {
      err = std::max(err, std::abs(right[i] - f[(i - 1 + n) % n]));
      err = std::max(err, std::abs(left[i] - f[(i + 1) % n]));
    }
    CHECK(err == 0.0);
  }
}

TEST_CASE("U3: limiter bounds quadratic SL by the bracketing cells",
          "[kinetics][sl][U3]") {
  const auto f = Step(40);
  const int n = f.size();
  for (const Real s : kShifts) {
    for (const bool limiter : {true, false}) {
      const auto g = Step1D(f, s, 2, limiter);
      Real excess = 0.0;
      for (int i = 0; i < n; ++i) {
        const int up = (s > 0.0) ? i - 1 : ((s < 0.0) ? i + 1 : i);
        const Real a = f[i], b = f[(up + n) % n];
        const Real lo = std::min(a, b), hi = std::max(a, b);
        excess = std::max(excess, std::max(g[i] - hi, lo - g[i]));
      }
      INFO("s = " << s << ", limiter " << limiter);
      if (limiter) {
        CHECK(excess <= 0.0);
      } else if (std::abs(s) > 0.0 && std::abs(s) < 1.0) {
        CHECK(excess > 1.0e-3); // negative control: Lax-Wendroff overshoots at a jump
      }
    }
  }
}

TEST_CASE("U3: 2D tensor-product SL of separable data", "[kinetics][sl][U3]") {
  const int nx = 12, ny = 10;
  const auto fx = Smooth(nx), fy = Smooth(ny);
  const Real sx = 0.37, sy = -0.62;
  for (const int order : {1, 2}) {
    const auto gx = Step1D(fx, sx, order, false);
    const auto gy = Step1D(fy, sy, order, false);
    const K::SLParams sl{order, false};
    const Real sh[3] = {sx, sy, 0.0};
    Real err = 0.0;
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        const Real v = K::SLInterpolate(sl, 2, sh, [&](int a, int b, int) {
          return fx[((i + a) % nx + nx) % nx] * fy[((j + b) % ny + ny) % ny];
        });
        err = std::max(err, std::abs(v - gx[i] * gy[j]));
      }
    INFO("order " << order);
    CHECK(err < 1.0e-14);
  }
}
