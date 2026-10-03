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

// T1: tensor-train core type and rounding (src/kinetics/tt_{linalg,tensor}.hpp).
//
//   * Householder QR reconstructs A with orthonormal Q; one-sided Jacobi returns the
//     singular values of a matrix built with known ones, for tall and wide shapes.
//   * The discrete equilibrium is exactly rank 1: FillMaxwellian equals the dense
//     Maxwellian node by node, and rounding keeps it rank 1.
//   * A tensor sum_l s_l u_l (x) v_l (x) w_l with orthonormal factor sets has both TT
//     unfolding spectra equal to {s_l}. Rounding it reproduces the analytic ranks and
//     the optimal error sqrt(sum_{l > r} s_l^2), meets ||f - f_r|| <= eps ||f||, and
//     reports the discarded norm equal to the actual error. A rank cap below the
//     tolerance rank is reported as a cap hit.
//   * A + A (rank 2r block form) rounds back to rank r with value 2A (eps = 0, the
//     fixed-rank mode), and alpha A + beta B combines exactly.
//   * T2: moments by core contraction (tt_moments.hpp) equal the dense node sums of the
//     decompressed tensor (raw and central, incl. heat flux), and the decompression
//     ForEachNode equals entrywise evaluation, for a generic rank-(3, 4) TT and for a
//     rank-2 sum of two drifting equilibria.
//
// Kernels run on device in a single-iteration loop, as in test_kinetics_equilibrium.

#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

#include "kinetics/equilibrium.hpp"
#include "kinetics/moments.hpp"
#include "kinetics/tt_linalg.hpp"
#include "kinetics/tt_moments.hpp"
#include "kinetics/tt_tensor.hpp"
#include "kinetics/velocity_grid.hpp"

using parthenon::Real;

namespace {

namespace K = Kinetics;
namespace T = Kinetics::TT;
using View = Kokkos::View<Real *>;

K::VelocityGrid Grid(const int nx, const int ny, const int nz) {
  const int n[3] = {nx, ny, nz};
  const Real lo[3] = {-6.0, -6.0, -6.0};
  const Real hi[3] = {6.0, 6.0, 6.0};
  return K::MakeVelocityGrid(n, lo, hi);
}

std::vector<Real> ToHost(const View &v) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  return std::vector<Real>(h.data(), h.data() + h.extent(0));
}

// Orthonormal DCT-II vector l on n points.
KOKKOS_INLINE_FUNCTION Real Dct(const int n, const int l, const int i) {
  const Real pi = 3.14159265358979323846;
  const Real s = (l == 0) ? std::sqrt(1.0 / n) : std::sqrt(2.0 / n);
  return s * std::cos(pi * (i + 0.5) * l / n);
}

// Pseudo-random value in [-1, 1) from an integer (device safe).
KOKKOS_INLINE_FUNCTION Real Hash(const int i) {
  unsigned int x = 2654435761u * static_cast<unsigned int>(i + 1);
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  return (x & 0xFFFFFF) / static_cast<Real>(0x800000) - 1.0;
}

// Place sum_{l < L} s_l u_l (x) v_l (x) w_l in TT form with ranks (L, L): G1 = [s_l u_l],
// G2(a, j, b) = delta_ab v_a(j), G3(b, k) = w_b(k). v uses DCT index l + 1 and w index
// l + 2 so that the three factor sets differ.
template <class Data>
KOKKOS_INLINE_FUNCTION void FillOrthoSum(const T::TTRef<Data> &t, const int L,
                                         const Real decay) {
  t.SetRanks(L, L);
  const int n0 = t.L.n[0], n1 = t.L.n[1], n2 = t.L.n[2];
  for (int a = 0; a < L; ++a) {
    const Real s = std::pow(decay, a);
    for (int i = 0; i < n0; ++i)
      t.G1(i, a) = s * Dct(n0, a, i);
    for (int k = 0; k < n2; ++k)
      t.G3(a, k) = Dct(n2, a + 2, k);
  }
  for (int b = 0; b < L; ++b)
    for (int j = 0; j < n1; ++j)
      for (int a = 0; a < L; ++a)
        t.G2(a, j, b) = (a == b) ? Dct(n1, a + 1, j) : 0.0;
}

} // namespace

TEST_CASE("T1: Householder QR and one-sided Jacobi SVD", "[kinetics][tt][T1]") {
  for (const auto shape : {std::pair<int, int>{9, 5}, std::pair<int, int>{4, 7}}) {
    const int m = shape.first, n = shape.second;
    const int k = std::min(m, n);
    // out: [ qr_recon_err, q_orth_err, svd_recon_err, |s - s_exact| max ]
    View out("out", 4);
    View work("work", 4 * m * n + n * n + 2 * (m + n)); // A0, A, Q, W, V, tau, s
    Kokkos::parallel_for(
        "linalg", 1, KOKKOS_LAMBDA(const int) {
          Real *p = work.data();
          Real *A0 = p;
          Real *A = A0 + m * n;
          Real *Q = A + m * n;
          Real *tau = Q + m * n;
          Real *W = tau + m + n;
          Real *V = W + m * n;
          Real *s = V + n * n;
          const T::Mat mA0 = T::ColMajor(A0, m, n), mA = T::ColMajor(A, m, n);
          for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
              mA0(i, j) = Hash(i + 37 * j);
          // QR
          T::Copy(mA0, mA);
          T::HouseholderQR(mA, tau);
          const T::Mat mQ = T::ColMajor(Q, m, k);
          T::SetIdentity(mQ);
          T::ApplyQ(mA, tau, k, mQ);
          Real e_qr = 0.0, e_orth = 0.0;
          for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
              Real v = 0.0;
              for (int l = 0; l <= j && l < k; ++l)
                v += mQ(i, l) * mA(l, j);
              e_qr = std::max(e_qr, std::abs(v - mA0(i, j)));
            }
          for (int a = 0; a < k; ++a)
            for (int b = 0; b < k; ++b) {
              Real v = 0.0;
              for (int i = 0; i < m; ++i)
                v += mQ(i, a) * mQ(i, b);
              e_orth = std::max(e_orth, std::abs(v - (a == b ? 1.0 : 0.0)));
            }
          // A = U1 diag(sx) U2^T with orthonormal U1 (m x k), U2 (n x k) from the DCT.
          const T::Mat mW = T::ColMajor(W, m, n);
          for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
              Real v = 0.0;
              for (int l = 0; l < k; ++l)
                v += Dct(m, l, i) * std::pow(0.1, l) * Dct(n, l, j);
              mW(i, j) = v;
              mA(i, j) = v;
            }
          const T::Mat mV = T::ColMajor(V, n, n);
          T::JacobiSVD(mW, mV, s);
          Real e_svd = 0.0, e_s = 0.0;
          for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
              Real v = 0.0;
              for (int l = 0; l < n; ++l)
                v += mW(i, l) * mV(j, l);
              e_svd = std::max(e_svd, std::abs(v - mA(i, j)));
            }
          for (int l = 0; l < n; ++l) {
            const Real exact = (l < k) ? std::pow(0.1, l) : 0.0;
            e_s = std::max(e_s, std::abs(s[l] - exact));
          }
          out(0) = e_qr;
          out(1) = e_orth;
          out(2) = e_svd;
          out(3) = e_s;
        });
    const auto h = ToHost(out);
    INFO("m = " << m << ", n = " << n);
    CHECK(h[0] < 1.0e-14);
    CHECK(h[1] < 1.0e-14);
    CHECK(h[2] < 1.0e-14);
    CHECK(h[3] < 1.0e-14);
  }
}

TEST_CASE("T1: discrete equilibrium is exactly rank 1", "[kinetics][tt][T1]") {
  const auto grid = Grid(16, 12, 10);
  const int rcap = 4;
  const T::TTLayout L = T::MakeLayout(grid, rcap);
  const T::RoundScratch sc = T::MakeRoundScratch(grid, rcap);
  View store("store", L.Size());
  View work("work", sc.Size());
  View out("out", 4); // [fill err, rounded err, r1, r2]
  Kokkos::parallel_for(
      "rank1", 1, KOKKOS_LAMBDA(const int) {
        K::EquilibriumTarget tgt{1.3, {0.4, -0.3, 0.2}, {1.1, 1.1, 1.1}};
        K::Maxwellian eq;
        K::SolveEquilibrium(grid, tgt, K::EquilibriumParams{}, eq);
        const auto t = T::MakeRef(T::PtrData{store.data()}, L);
        T::FillMaxwellian(grid, eq, t);
        Real e_fill = 0.0, mmax = 0.0;
        for (int k = 0; k < grid.nv[2]; ++k)
          for (int j = 0; j < grid.nv[1]; ++j)
            for (int i = 0; i < grid.nv[0]; ++i) {
              const Real m = eq(grid, i, j, k);
              mmax = std::max(mmax, m);
              e_fill = std::max(e_fill, std::abs(t(i, j, k) - m));
            }
        const auto in = T::InputRef(sc, work.data());
        T::CopyTT(t, in);
        const auto info = T::Round(sc, work.data(), t, T::RoundParams{});
        Real e_round = 0.0;
        for (int k = 0; k < grid.nv[2]; ++k)
          for (int j = 0; j < grid.nv[1]; ++j)
            for (int i = 0; i < grid.nv[0]; ++i)
              e_round = std::max(e_round, std::abs(t(i, j, k) - eq(grid, i, j, k)));
        out(0) = e_fill / mmax;
        out(1) = e_round / mmax;
        out(2) = info.r1;
        out(3) = info.r2;
      });
  const auto h = ToHost(out);
  CHECK(h[0] < 1.0e-14);
  CHECK(h[1] < 1.0e-13);
  CHECK(h[2] == 1);
  CHECK(h[3] == 1);
}

TEST_CASE("T1: rounding meets the tolerance with the optimal ranks",
          "[kinetics][tt][T1]") {
  const auto grid = Grid(16, 14, 12);
  const int Lr = 8;       // exact rank of the test tensor
  const Real decay = 0.1; // s_l = 0.1^l
  const T::TTLayout Lin = T::MakeLayout(grid, Lr);
  const T::RoundScratch sc = T::MakeRoundScratch(grid, Lr);

  struct Case {
    Real eps;
    int rcap;
    int r_expected;
    bool cap_expected;
  };
  // Tail after rank r: sqrt(sum_{l >= r} 0.01^l) / ||f|| ~ 0.1^r. Each SVD gets
  // eps / sqrt(2), so the tolerance rank is the smallest r with 0.1^r <~ eps / sqrt 2.
  const std::vector<Case> cases = {{1.0e-2, 8, 3, false},
                                   {1.0e-5, 8, 6, false},
                                   {0.0, 8, 8, false},
                                   {1.0e-5, 4, 4, true},
                                   {1.0e-1, 8, 2, false}};
  for (const auto &c : cases) {
    const T::TTLayout Lout = T::MakeLayout(grid, c.rcap);
    View ref("ref", Lin.Size());
    View dst("dst", Lout.Size());
    View work("work", sc.Size());
    // [r1, r2, cap_hit, rel err, discarded / ||f||, optimal / ||f||, svd_ok]
    View out("out", 7);
    const Real eps = c.eps;
    Kokkos::parallel_for(
        "round", 1, KOKKOS_LAMBDA(const int) {
          const auto f = T::MakeRef(T::PtrData{ref.data()}, Lin);
          FillOrthoSum(f, Lr, decay);
          T::CopyTT(f, T::InputRef(sc, work.data()));
          const auto g = T::MakeRef(T::PtrData{dst.data()}, Lout);
          T::RoundParams prm;
          prm.eps = eps;
          const auto info = T::Round(sc, work.data(), g, prm);
          Real err2 = 0.0, norm2 = 0.0;
          for (int k = 0; k < grid.nv[2]; ++k)
            for (int j = 0; j < grid.nv[1]; ++j)
              for (int i = 0; i < grid.nv[0]; ++i) {
                const Real a = f(i, j, k), b = g(i, j, k);
                err2 += (a - b) * (a - b);
                norm2 += a * a;
              }
          Real opt2 = 0.0;
          for (int l = info.r1; l < Lr; ++l)
            opt2 += std::pow(decay, 2 * l);
          const Real norm = std::sqrt(norm2);
          out(0) = info.r1;
          out(1) = info.r2;
          out(2) = info.cap_hit ? 1.0 : 0.0;
          out(6) = info.svd_ok ? 1.0 : 0.0;
          out(3) = std::sqrt(err2) / norm;
          out(4) = info.discarded / norm;
          out(5) = std::sqrt(opt2) / norm;
        });
    const auto h = ToHost(out);
    INFO("eps = " << c.eps << ", rcap = " << c.rcap);
    CHECK(h[0] == c.r_expected);
    CHECK(h[1] == c.r_expected);
    CHECK((h[2] == 1.0) == c.cap_expected);
    CHECK(h[6] == 1.0);
    if (!c.cap_expected) CHECK(h[3] <= c.eps + 1.0e-14);
    // The first cut also removes the tail of the second unfolding (the spectra are the
    // same), so the second cut discards nothing and the error is the one-cut optimum.
    CHECK(h[3] == Catch::Approx(h[5]).epsilon(1.0e-8).margin(1.0e-14));
    CHECK(h[4] == Catch::Approx(h[3]).epsilon(1.0e-8).margin(1.0e-14));
  }
}

TEST_CASE("T1: sums of tensor trains round back to the exact rank",
          "[kinetics][tt][T1]") {
  const auto grid = Grid(12, 10, 8);
  const int r = 3;
  const T::TTLayout L = T::MakeLayout(grid, r);
  const T::TTLayout L2 = T::MakeLayout(grid, 2 * r);
  const T::RoundScratch sc = T::MakeRoundScratch(grid, 2 * r);
  View a("a", L.Size()), b("b", L.Size()), c("c", L.Size()), d("d", L2.Size());
  View work("work", sc.Size());
  View out("out", 6); // [r1, r2 of A + A, err of A + A, r1, r2 of 2A - 3B, err]
  Kokkos::parallel_for(
      "sum", 1, KOKKOS_LAMBDA(const int) {
        const auto A = T::MakeRef(T::PtrData{a.data()}, L);
        const auto B = T::MakeRef(T::PtrData{b.data()}, L);
        const auto C = T::MakeRef(T::PtrData{c.data()}, L);
        const auto D = T::MakeRef(T::PtrData{d.data()}, L2);
        // Generic (non-orthogonal) cores of full rank r.
        A.SetRanks(r, r);
        B.SetRanks(1, 2);
        int h = 0;
        for (int x = 0; x < grid.nv[0] * r; ++x)
          a.data()[A.L.Slot1() + x] = Hash(h++);
        for (int x = 0; x < r * grid.nv[1] * r; ++x)
          a.data()[A.L.Slot2() + x] = Hash(h++);
        for (int x = 0; x < r * grid.nv[2]; ++x)
          a.data()[A.L.Slot3() + x] = Hash(h++);
        for (int x = 0; x < grid.nv[0]; ++x)
          b.data()[B.L.Slot1() + x] = Hash(h++);
        for (int x = 0; x < grid.nv[1] * 2; ++x)
          b.data()[B.L.Slot2() + x] = Hash(h++);
        for (int x = 0; x < 2 * grid.nv[2]; ++x)
          b.data()[B.L.Slot3() + x] = Hash(h++);

        T::RoundParams fixed;
        fixed.eps = 0.0;
        T::AddInto(sc, work.data(), 1.0, A, 1.0, A);
        const auto i1 = T::Round(sc, work.data(), C, fixed);
        Real e1 = 0.0, n1 = 0.0;
        for (int k = 0; k < grid.nv[2]; ++k)
          for (int j = 0; j < grid.nv[1]; ++j)
            for (int i = 0; i < grid.nv[0]; ++i) {
              e1 = std::max(e1, std::abs(C(i, j, k) - 2.0 * A(i, j, k)));
              n1 = std::max(n1, std::abs(2.0 * A(i, j, k)));
            }
        T::AddInto(sc, work.data(), 2.0, A, -3.0, B);
        const auto i2 = T::Round(sc, work.data(), D, fixed);
        Real e2 = 0.0, n2 = 0.0;
        for (int k = 0; k < grid.nv[2]; ++k)
          for (int j = 0; j < grid.nv[1]; ++j)
            for (int i = 0; i < grid.nv[0]; ++i) {
              const Real v = 2.0 * A(i, j, k) - 3.0 * B(i, j, k);
              e2 = std::max(e2, std::abs(D(i, j, k) - v));
              n2 = std::max(n2, std::abs(v));
            }
        out(0) = i1.r1;
        out(1) = i1.r2;
        out(2) = e1 / n1;
        out(3) = i2.r1;
        out(4) = i2.r2;
        out(5) = e2 / n2;
      });
  const auto h = ToHost(out);
  CHECK(h[0] == r);
  CHECK(h[1] == r);
  CHECK(h[2] < 1.0e-13);
  // 2A - 3B: B has ranks (1, 2), so the sum has ranks at most (r + 1, r + 2) and the
  // fixed-rank rounding into capacity 2r must reproduce it to roundoff.
  CHECK(h[3] <= r + 1);
  CHECK(h[4] <= r + 2);
  CHECK(h[5] < 1.0e-13);
}

TEST_CASE("T2: TT moments equal dense moments of the decompressed tensor",
          "[kinetics][tt][T2]") {
  const auto grid = Grid(14, 12, 10);
  const int rcap = 4;
  const T::TTLayout L = T::MakeLayout(grid, rcap);
  for (const int which : {0, 1}) {
    View a("a", L.Size()), dense("dense", grid.Size());
    View out("out", 3); // [max raw moment error, max central error, decompression err]
    Kokkos::parallel_for(
        "moments", 1, KOKKOS_LAMBDA(const int) {
          const auto A = T::MakeOutRef(T::PtrData{a.data()}, L);
          if (which == 0) {
            A.SetRanks(3, 4);
            int h = 0;
            for (int x = 0; x < grid.nv[0] * 3; ++x)
              a.data()[L.Slot1() + x] = 1.0 + 0.5 * Hash(h++);
            for (int x = 0; x < 3 * grid.nv[1] * 4; ++x)
              a.data()[L.Slot2() + x] = Hash(h++);
            for (int x = 0; x < 4 * grid.nv[2]; ++x)
              a.data()[L.Slot3() + x] = Hash(h++);
          } else {
            K::Maxwellian eq[2];
            K::EquilibriumTarget t0{0.5, {-0.8, 0.1, 0.0}, {0.9, 0.9, 0.9}};
            K::EquilibriumTarget t1{0.5, {0.8, 0.1, 0.0}, {0.9, 0.9, 0.9}};
            K::SolveEquilibrium(grid, t0, K::EquilibriumParams{}, eq[0]);
            K::SolveEquilibrium(grid, t1, K::EquilibriumParams{}, eq[1]);
            T::FillMaxwellians(grid, eq, 2, A);
          }
          const auto R = T::MakeRef(T::PtrData{a.data()}, L);
          Real e_dec = 0.0;
          T::ForEachNode(R, [&](const int i, const int j, const int k, const Real val) {
            dense(grid.Flat(i, j, k)) = val;
            e_dec = std::max(e_dec, std::abs(val - R(i, j, k)));
          });
          auto f = [&](const int n) { return dense(n); };
          const auto rd = K::ComputeRawMoments(grid, f);
          const auto rt = T::ComputeRawMoments(grid, R);
          Real scale = std::abs(rd.n), e_raw = std::abs(rd.n - rt.n);
          for (int d = 0; d < 3; ++d) {
            scale = std::max(scale, std::max(std::abs(rd.nu[d]), std::abs(rd.nvv[d])));
            e_raw = std::max(e_raw, std::abs(rd.nu[d] - rt.nu[d]));
            e_raw = std::max(e_raw, std::abs(rd.nvv[d] - rt.nvv[d]));
          }
          const Real u[3] = {0.3, -0.2, 0.1};
          const auto cd = K::ComputeCentralMoments(grid, f, u);
          const auto ct = T::ComputeCentralMoments(grid, R, u);
          Real cscale = 0.0, e_c = 0.0;
          for (int q = 0; q < 6; ++q) {
            cscale = std::max(cscale, std::abs(cd.stress[q]));
            e_c = std::max(e_c, std::abs(cd.stress[q] - ct.stress[q]));
          }
          for (int d = 0; d < 3; ++d) {
            cscale = std::max(cscale, std::abs(cd.heat[d]));
            e_c = std::max(e_c, std::abs(cd.heat[d] - ct.heat[d]));
          }
          out(0) = e_raw / scale;
          out(1) = e_c / cscale;
          out(2) = e_dec;
        });
    const auto h = ToHost(out);
    INFO((which == 0 ? "generic rank (3, 4)" : "two drifting equilibria"));
    CHECK(h[0] < 1.0e-13);
    CHECK(h[1] < 1.0e-13);
    CHECK(h[2] < 1.0e-14);
  }
}
