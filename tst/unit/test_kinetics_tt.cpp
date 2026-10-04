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
//   * T3: one TT relaxation step (tt_relax.hpp) equals the dense RelaxCell (bgk.hpp)
//     applied to the decompressed f, node by node, for exact and rational (c > 1) steps;
//     ranks grow by at most 1 and the invariants are kept.
//   * T4: TT streaming of one cell along each velocity axis (tt_stream.hpp) equals the
//     dense SL update node by node, for linear and quadratic weights; ReverseAxis
//     mirrors one velocity axis exactly; Contract equals the dense weighted sum.
//   * T5: cross approximation (tt_cross.hpp). phi = identity reproduces a generic TT
//     node by node with ranks <= those of f; the regularized entropy of two drifting
//     equilibria (plus a small negative perturbation) matches phi(f) node by node and
//     sums to the exact entropy; a rank cap below the needed rank is reported;
//     SampledMinF finds a planted smooth negative lobe.
//
// Kernels run on device in a single-iteration loop, as in test_kinetics_equilibrium.

#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

#include "kinetics/bgk.hpp"
#include "kinetics/equilibrium.hpp"
#include "kinetics/moments.hpp"
#include "kinetics/semi_lagrangian.hpp"
#include "kinetics/tt_cross.hpp"
#include "kinetics/tt_linalg.hpp"
#include "kinetics/tt_moments.hpp"
#include "kinetics/tt_relax.hpp"
#include "kinetics/tt_stream.hpp"
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

TEST_CASE("T3: TT relaxation equals dense relaxation of the decompressed f",
          "[kinetics][tt][T3]") {
  const auto grid = Grid(16, 14, 12);
  const int rcap = 6;
  const T::TTLayout L = T::MakeLayout(grid, rcap);
  const auto sc = T::MakeRelaxScratch(grid, rcap);
  const K::CollisionModel model{
      K::CollisionModel::Type::constant, 2.0, 1.0, 1.0, 0.5, 1.0};
  // exact (c = 1 - e^{-1}) and a rational DIRK-like step with c > 1
  for (const auto step :
       {K::ExactRelaxation(0.5), K::RationalRelaxation(4.0, 0.7, 0.2)}) {
    View a("a", L.Size()), dense("dense", grid.Size()), work("work", sc.Size());
    View out("out", 6); // [max node err / max f, r1, r2, mass drift, energy drift, c]
    Kokkos::parallel_for(
        "relax", 1, KOKKOS_LAMBDA(const int) {
          // f = bi-Maxwellian + 0.2 x a drifting equilibrium: rank 2, non-equilibrium.
          K::Maxwellian eq[2];
          K::EquilibriumTarget t0{1.0, {0.2, 0.0, -0.1}, {1.5, 0.75, 0.75}};
          K::EquilibriumTarget t1{0.2, {-1.0, 0.5, 0.0}, {0.5, 0.5, 0.5}};
          K::SolveAnisotropicEquilibrium(grid, t0, K::EquilibriumParams{}, eq[0]);
          K::SolveEquilibrium(grid, t1, K::EquilibriumParams{}, eq[1]);
          const auto f = T::MakeOutRef(T::PtrData{a.data()}, L);
          T::FillMaxwellians(grid, eq, 2, f);
          T::ForEachNode(f, [&](const int i, const int j, const int k, const Real v) {
            dense(grid.Flat(i, j, k)) = v;
          });
          auto fd = [&](const int n) -> Real & { return dense(n); };
          const auto m0 = K::ComputeRawMoments(grid, fd);
          const auto rd =
              K::RelaxCell(grid, fd, step, model, 1.0, K::EquilibriumParams{});
          const auto fr = T::MakeRef(T::PtrData{a.data()}, L);
          T::RoundParams prm;
          prm.eps = 1.0e-14;
          const auto rt = T::RelaxCellTT(grid, fr, step, model, 1.0,
                                         K::EquilibriumParams{}, sc, work.data(), prm);
          const auto g = T::MakeRef(T::PtrData{a.data()}, L);
          Real err = 0.0, fmax = 0.0;
          T::ForEachNode(g, [&](const int i, const int j, const int k, const Real v) {
            const Real d = dense(grid.Flat(i, j, k));
            err = std::max(err, std::abs(v - d));
            fmax = std::max(fmax, std::abs(d));
          });
          const auto m1 = T::ComputeRawMoments(grid, g);
          out(0) = err / fmax;
          out(1) = rt.round.r1;
          out(2) = rt.round.r2;
          out(3) = std::abs(m1.n - m0.n) / m0.n;
          out(4) = std::abs((m1.nvv[0] + m1.nvv[1] + m1.nvv[2]) -
                            (m0.nvv[0] + m0.nvv[1] + m0.nvv[2])) /
                   (m0.nvv[0] + m0.nvv[1] + m0.nvv[2]);
          out(5) = step.Fraction(rd.nu);
        });
    const auto h = ToHost(out);
    INFO("c = " << h[5]);
    CHECK(h[0] < 1.0e-13);
    CHECK(h[1] <= 3);
    CHECK(h[2] <= 3);
    CHECK(h[3] < 1.0e-13);
    CHECK(h[4] < 1.0e-13);
  }
}

TEST_CASE("T4: TT streaming equals dense SL of the decompressed cells",
          "[kinetics][tt][T4]") {
  const auto grid = Grid(12, 10, 8);
  const int rcap = 8;
  const T::TTLayout L = T::MakeLayout(grid, rcap);
  const auto sc = T::MakeStreamScratch(grid, rcap);
  for (const int order : {1, 2}) {
    for (const int d : {0, 1, 2}) {
      View cells("cells", 4 * L.Size()), work("work", sc.Size());
      View out("out", 4); // [stream err, reverse err, contract err, total ranks]
      Kokkos::parallel_for(
          "stream", 1, KOKKOS_LAMBDA(const int) {
            // Three neighbor cells: different two-Maxwellian states (rank 2 each).
            T::TTRef<T::PtrData> c[3] = {
                T::MakeOutRef(T::PtrData{cells.data()}, L),
                T::MakeOutRef(T::PtrData{cells.data() + L.Size()}, L),
                T::MakeOutRef(T::PtrData{cells.data() + 2 * L.Size()}, L)};
            for (int q = 0; q < 3; ++q) {
              K::Maxwellian eq[2];
              K::EquilibriumTarget t0{
                  1.0 + 0.3 * q, {0.4 - 0.2 * q, 0.1, -0.2}, {1.0, 1.0, 1.0}};
              K::EquilibriumTarget t1{
                  0.5, {-0.6, 0.3 * q, 0.2}, {0.6 + 0.1 * q, 0.6, 0.6}};
              K::SolveEquilibrium(grid, t0, K::EquilibriumParams{}, eq[0]);
              K::SolveEquilibrium(grid, t1, K::EquilibriumParams{}, eq[1]);
              T::FillMaxwellians(grid, eq, 2, c[q]);
            }
            const auto o = T::MakeOutRef(T::PtrData{cells.data() + 3 * L.Size()}, L);
            T::RoundParams prm;
            prm.eps = 1.0e-15;
            T::RoundTally tally;
            const Real hdx = 0.07; // |s| = |v| 0.07 <= 0.42 on the +-6 box
            T::StreamCellTT(grid, order, d, hdx, T::MakeRef(c[0].data, L),
                            T::MakeRef(c[1].data, L), T::MakeRef(c[2].data, L), o, sc,
                            work.data(), prm, tally);
            const auto r = T::MakeRef(o.data, L);
            Real err = 0.0, fmax = 0.0;
            for (int k = 0; k < grid.nv[2]; ++k)
              for (int j = 0; j < grid.nv[1]; ++j)
                for (int i = 0; i < grid.nv[0]; ++i) {
                  const int idx[3] = {i, j, k};
                  Real w[3];
                  K::SLWeights(order, grid.Node(d, idx[d]) * hdx, w);
                  const Real ref = w[0] * T::MakeRef(c[0].data, L)(i, j, k) +
                                   w[1] * T::MakeRef(c[1].data, L)(i, j, k) +
                                   w[2] * T::MakeRef(c[2].data, L)(i, j, k);
                  err = std::max(err, std::abs(r(i, j, k) - ref));
                  fmax = std::max(fmax, std::abs(ref));
                }
            // ReverseAxis of cell 1 along d, compared with mirrored evaluation.
            T::ReverseAxis(T::MakeRef(c[1].data, L), o, d);
            const auto rv = T::MakeRef(o.data, L);
            const auto c1 = T::MakeRef(c[1].data, L);
            Real erev = 0.0;
            for (int k = 0; k < grid.nv[2]; ++k)
              for (int j = 0; j < grid.nv[1]; ++j)
                for (int i = 0; i < grid.nv[0]; ++i) {
                  int m[3] = {i, j, k};
                  m[d] = grid.nv[d] - 1 - m[d];
                  erev = std::max(erev, std::abs(rv(i, j, k) - c1(m[0], m[1], m[2])));
                }
            // Contract with weights |v_d| (other axes 1) vs the dense sum.
            auto wd = [&](const int n) { return std::abs(grid.Node(d, n)); };
            auto one = [](const int) { return 1.0; };
            const Real con = (d == 0)   ? T::Contract(c1, wd, one, one)
                             : (d == 1) ? T::Contract(c1, one, wd, one)
                                        : T::Contract(c1, one, one, wd);
            Real dsum = 0.0;
            for (int k = 0; k < grid.nv[2]; ++k)
              for (int j = 0; j < grid.nv[1]; ++j)
                for (int i = 0; i < grid.nv[0]; ++i) {
                  const int idx[3] = {i, j, k};
                  dsum += std::abs(grid.Node(d, idx[d])) * c1(i, j, k);
                }
            out(0) = err / fmax;
            out(1) = erev;
            out(2) = std::abs(con - dsum) / std::abs(dsum);
            out(3) = tally.cap_hits + tally.svd_failures;
          });
      const auto h = ToHost(out);
      INFO("order " << order << ", axis " << d);
      CHECK(h[0] < 1.0e-13);
      CHECK(h[1] == 0.0);
      CHECK(h[2] < 1.0e-13);
      CHECK(h[3] == 0.0);
    }
  }
}

TEST_CASE("T5: cross approximation of pointwise functions of a TT",
          "[kinetics][tt][T5]") {
  const auto grid = Grid(20, 18, 16);
  const int rcap = 4;
  const T::TTLayout L = T::MakeLayout(grid, rcap);
  // which: 0 identity on a generic rank-(3, 4) TT, 1 entropy of two equilibria + a
  // rank-1 perturbation, 2 the same with rank cap 2, 3 SampledMinF with a planted min.
  for (const int which : {0, 1, 2, 3}) {
    T::CrossParams prm;
    if (which == 2) prm.rank_max = 2;
    const auto sc = T::MakeCrossScratch(grid, rcap, prm);
    View a("a", L.Size()), work("work", sc.RealSize());
    Kokkos::View<int *> iwork("iwork", sc.IntSize());
    // out: [max node err / max |g|, |sum g - exact| / |exact|, max(q1, q2), cap hit,
    //       converged, evals, min f found, true min f]
    View out("out", 8);
    Kokkos::parallel_for(
        "cross", 1, KOKKOS_LAMBDA(const int) {
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
            K::EquilibriumTarget t0{0.5, {-1.2, 0.3, 0.0}, {0.8, 0.8, 0.8}};
            K::EquilibriumTarget t1{0.5, {1.2, -0.2, 0.1}, {1.1, 1.1, 1.1}};
            K::SolveEquilibrium(grid, t0, K::EquilibriumParams{}, eq[0]);
            K::SolveEquilibrium(grid, t1, K::EquilibriumParams{}, eq[1]);
            T::FillMaxwellians(grid, eq, 2, A);
            // Third rank-1 term: -1e-12 at every node (rounding-noise-like negative
            // tails), or for which = 3 a smooth negative lobe -1e-3 at node (13, 5, 11)
            // of width 3 nodes (an isolated single-node spike is not found by the
            // fiber search; the cross diagnostics assume smooth f, S1-Q21).
            A.SetRanks(3, 3);
            for (int b = 0; b < 3; ++b)
              for (int j = 0; j < grid.nv[1]; ++j)
                for (int c = 0; c < 3; ++c)
                  if (b == 2 || c == 2) A.G2(c, j, b) = 0.0;
            const auto lobe = [](const int i, const int c) {
              return std::exp(-0.5 * (i - c) * (i - c) / 9.0);
            };
            for (int i = 0; i < grid.nv[0]; ++i)
              A.G1(i, 2) = (which == 3) ? -1.0e-3 * lobe(i, 13) : -1.0e-12;
            for (int j = 0; j < grid.nv[1]; ++j)
              A.G2(2, j, 2) = (which == 3) ? lobe(j, 5) : 1.0;
            for (int k = 0; k < grid.nv[2]; ++k)
              A.G3(2, k) = (which == 3) ? lobe(k, 11) : 1.0;
          }
          const auto F = T::MakeRef(T::PtrData{a.data()}, L);
          Real fmin = 1.0e300;
          T::ForEachNode(F, [&](const int, const int, const int, const Real v) {
            fmin = std::min(fmin, v);
          });
          out(7) = fmin;
          out(6) = T::SampledMinF(F, prm.trial_factor);
          if (which == 3) return;
          const T::EntropyPhi ent{1.0e-12};
          Real fscale = 0.0;
          T::CrossInfo info;
          if (which == 0) {
            const auto id = [](const Real f, const Real) { return f; };
            info = T::TTCross(sc, work.data(), iwork.data(), F, id, prm);
          } else {
            info = T::TTCross(sc, work.data(), iwork.data(), F, ent, prm);
          }
          fscale = info.fscale;
          const auto G = T::CrossOutput(sc, work.data());
          Real emax = 0.0, gmax = 0.0, gsum = 0.0, hsum = 0.0;
          T::ForEachNode(F, [&](const int i, const int j, const int k, const Real v) {
            const Real gx = (which == 0) ? v : ent(v, fscale);
            gmax = std::max(gmax, std::abs(gx));
            emax = std::max(emax, std::abs(G(i, j, k) - gx));
            gsum += G(i, j, k);
            hsum += (v > 0.0) ? v * (std::log(v) - 1.0) : 0.0;
          });
          out(0) = emax / gmax;
          out(1) = (which == 0) ? 0.0 : std::abs(gsum - hsum) / std::abs(hsum);
          out(2) = std::max(info.q1, info.q2);
          out(3) = info.cap_hit ? 1.0 : 0.0;
          out(4) = info.converged ? 1.0 : 0.0;
          out(5) = info.evals;
        });
    const auto h = ToHost(out);
    INFO("case " << which << ": node err " << h[0] << ", sum err " << h[1] << ", rank "
                 << h[2] << ", evals " << h[5] << ", min f " << h[6] << " (true " << h[7]
                 << ")");
    CHECK(h[6] >= h[7]); // a sampled min is an upper bound
    if (which == 0) {
      CHECK(h[0] < 1.0e-12);
      CHECK(h[2] <= 4);
      CHECK(h[4] == 1.0);
    } else if (which == 1) {
      CHECK(h[0] < 1.0e-9);
      CHECK(h[1] < 1.0e-10);
      CHECK(h[3] == 0.0);
      CHECK(h[4] == 1.0);
    } else if (which == 2) {
      CHECK(h[3] == 1.0);
      CHECK(h[4] == 0.0);
      CHECK(h[2] <= 2);
    } else {
      CHECK(h[6] == Catch::Approx(h[7]).epsilon(1.0e-12));
      CHECK(h[7] < -5.0e-4);
    }
  }
}
