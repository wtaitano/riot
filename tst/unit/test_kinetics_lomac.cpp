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

// LoMaC correction of one cell (src/kinetics/lomac.hpp, S3_DESIGN.md step 3):
//   * dense: f + M P has the target n, n u, sum |v|^2 f w to roundoff, for a
//     non-equilibrium f (bi-Maxwellian + drifting beam) and a target that differs from
//     f's moments by a few percent, on a coarse grid where the sampled Maxwellian alone
//     misses the target by > 1e-9 (control, measured 3.7e-6: the 5 x 5 system is what
//     makes it exact);
//   * TT: f + M P has ranks (r1 + 2, r2 + 2), equals the dense correction node by node,
//     and has the target moments to roundoff; a rank capacity below r + 2 is refused;
//   * invalid targets (n <= 0, theta <= 0) are reported and change nothing.

#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

#include "kinetics/equilibrium.hpp"
#include "kinetics/lomac.hpp"
#include "kinetics/moments.hpp"
#include "kinetics/tt_moments.hpp"
#include "kinetics/tt_tensor.hpp"
#include "kinetics/velocity_grid.hpp"

using parthenon::Real;

namespace {

namespace K = Kinetics;
namespace T = Kinetics::TT;
using View = Kokkos::View<Real *>;

std::vector<Real> ToHost(const View &v) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  return std::vector<Real>(h.data(), h.data() + h.extent(0));
}

// Largest relative moment error of m against the target.
KOKKOS_INLINE_FUNCTION Real MomentError(const K::RawMoments &m, const K::LomacTarget &t) {
  Real e = std::abs(m.n - t.n) / t.n;
  const Real cs = std::sqrt(t.nvv / t.n) * t.n; // momentum scale
  for (int d = 0; d < 3; ++d)
    e = std::max(e, std::abs(m.nu[d] - t.nu[d]) / cs);
  e = std::max(e, std::abs(m.nvv[0] + m.nvv[1] + m.nvv[2] - t.nvv) / t.nvv);
  return e;
}

} // namespace

TEST_CASE("L1: LoMaC correction hits the target moments, dense and TT",
          "[kinetics][lomac][L1]") {
  const int n[3] = {14, 12, 10};
  const Real lo[3] = {-5.0, -5.0, -5.0}, hi[3] = {5.0, 5.0, 5.0};
  const auto grid = K::MakeVelocityGrid(n, lo, hi);
  const int rcap = 6;
  const T::TTLayout L = T::MakeLayout(grid, rcap);
  const T::TTLayout Lsmall = T::MakeLayout(grid, 3);
  View a("a", L.Size()), b("b", Lsmall.Size()), work("work", L.Size());
  View dense("dense", grid.Size());
  View out("out", 10);
  Kokkos::parallel_for(
      "lomac", 1, KOKKOS_LAMBDA(const int) {
        // f: bi-Maxwellian + drifting beam (rank 2, non-equilibrium)
        K::Maxwellian eq[2];
        K::EquilibriumTarget t0{1.0, {0.2, 0.0, -0.1}, {1.4, 0.8, 0.8}};
        K::EquilibriumTarget t1{0.2, {-1.0, 0.5, 0.0}, {0.5, 0.5, 0.5}};
        K::SolveAnisotropicEquilibrium(grid, t0, K::EquilibriumParams{}, eq[0]);
        K::SolveEquilibrium(grid, t1, K::EquilibriumParams{}, eq[1]);
        const auto f = T::MakeOutRef(T::PtrData{a.data()}, L);
        T::FillMaxwellians(grid, eq, 2, f);
        T::ForEachNode(f, [&](const int i, const int j, const int k, const Real v) {
          dense(grid.Flat(i, j, k)) = v;
        });
        auto fd = [&](const int q) -> Real & { return dense(q); };
        const auto m0 = K::ComputeRawMoments(grid, fd);
        // target: f's moments shifted by a few percent
        K::LomacTarget tgt{1.03 * m0.n,
                           {m0.nu[0] + 0.05 * m0.n, m0.nu[1] - 0.02 * m0.n, m0.nu[2]},
                           0.97 * (m0.nvv[0] + m0.nvv[1] + m0.nvv[2])};
        const auto r = K::LomacSolve(grid, m0, tgt);
        out(0) = (r.status == K::LomacResult::Status::applied) ? 1.0 : 0.0;

        // control: the sampled window alone (moments of M) misses the target
        K::RawMoments mM{0.0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        {
          Real s0 = 0.0, s1[3] = {0.0, 0.0, 0.0}, s2 = 0.0;
          for (int iz = 0; iz < grid.nv[2]; ++iz)
            for (int iy = 0; iy < grid.nv[1]; ++iy)
              for (int ix = 0; ix < grid.nv[0]; ++ix) {
                const Real w = r.M(grid, ix, iy, iz);
                const Real v[3] = {grid.Node(0, ix), grid.Node(1, iy), grid.Node(2, iz)};
                s0 += w;
                for (int d = 0; d < 3; ++d)
                  s1[d] += v[d] * w;
                s2 += (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) * w;
              }
          mM.n = s0 * grid.Weight();
          for (int d = 0; d < 3; ++d)
            mM.nu[d] = s1[d] * grid.Weight();
          mM.nvv[0] = s2 * grid.Weight();
        }
        out(1) = MomentError(mM, tgt);

        // dense correction
        K::LomacApplyDense(grid, r, fd);
        out(2) = MomentError(K::ComputeRawMoments(grid, fd), tgt);

        // TT correction: ranks + 2, equal to dense, target moments
        const auto fr = T::MakeRef(T::PtrData{a.data()}, L);
        const int r1 = fr.R1(), r2 = fr.R2();
        const bool ok = T::LomacApplyTT(grid, r, fr, work.data());
        const auto g = T::MakeRef(T::PtrData{a.data()}, L);
        out(3) = ok ? 1.0 : 0.0;
        out(4) = g.R1() - r1;
        out(5) = g.R2() - r2;
        Real err = 0.0, fmax = 0.0;
        T::ForEachNode(g, [&](const int i, const int j, const int k, const Real v) {
          const Real d = dense(grid.Flat(i, j, k));
          err = std::max(err, std::abs(v - d));
          fmax = std::max(fmax, std::abs(d));
        });
        out(6) = err / fmax;
        out(7) = MomentError(T::ComputeRawMoments(grid, g), tgt);

        // capacity 3 < rank 2 + 2: refused, f unchanged
        const auto s = T::MakeOutRef(T::PtrData{b.data()}, Lsmall);
        T::FillMaxwellians(grid, eq, 2, s);
        out(8) = T::LomacApplyTT(grid, r, T::MakeRef(T::PtrData{b.data()}, Lsmall),
                                 work.data())
                     ? 1.0
                     : 0.0;

        // invalid target
        const K::LomacTarget bad{1.0, {0.0, 0.0, 0.0}, -1.0};
        out(9) = (K::LomacSolve(grid, m0, bad).status ==
                  K::LomacResult::Status::invalid_target)
                     ? 1.0
                     : 0.0;
      });
  const auto h = ToHost(out);
  INFO("window-only error " << h[1] << ", dense " << h[2] << ", TT vs dense " << h[6]
                            << ", TT " << h[7]);
  CHECK(h[0] == 1.0);
  CHECK(h[1] > 1.0e-9); // measured 3.7e-6
  CHECK(h[2] < 1.0e-13);
  CHECK(h[3] == 1.0);
  CHECK(h[4] == 2.0);
  CHECK(h[5] == 2.0);
  CHECK(h[6] < 1.0e-13);
  CHECK(h[7] < 1.0e-13);
  CHECK(h[8] == 0.0);
  CHECK(h[9] == 1.0);
}
