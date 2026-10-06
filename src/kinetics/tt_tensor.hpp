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
#ifndef KINETICS_TT_TENSOR_HPP_
#define KINETICS_TT_TENSOR_HPP_
// This file was made in part with generative AI.

// Three-core tensor train over the velocity indices of one spatial cell:
//
//   f(ix, iy, iz) = sum_{a < r1, b < r2} G1(ix, a) G2(a, iy, b) G3(b, iz).
//
// Storage of one cell (TTLayout), a flat array of Size() reals:
//   [0]       r1          (stored as a Real)
//   [1]       r2
//   [2, ...)  G1, then G2, then G3, each at a fixed offset sized for the capacity rank.
// Inside each slot the core is packed with its actual ranks, column-major:
//   G1(i, a)    = slot1[i + n0 a]                 (n0 x r1)
//   G2(a, j, b) = slot2[a + r1 (j + n1 b)]        (r1 x n1 x r2)
//   G3(b, k)    = slot3[b + r2 k]                 (r2 x n2)
// so the left unfolding of G2, (r1 n1) x r2, and the right one, r1 x (n1 r2), are both
// contiguous column-major matrices.
//
// Access goes through a Data functor, data(n) -> Real &, so the same code works on a raw
// pointer (scratch, unit tests) and on a Parthenon variable (one component per n).
//
// Round (Oseledets' TT rounding): right-to-left QR orthogonalization, then left-to-right
// truncated SVD. With eps > 0 each SVD drops the largest tail whose norm is at most
// eps ||f|| / sqrt(2), so ||f - f_rounded|| <= eps ||f||; with eps = 0 only the
// numerically zero tail (rank_floor ||f||) is dropped. Ranks are capped at the output
// capacity; Info reports cap hits and the discarded norm, which equals the actual
// rounding error up to roundoff (the two truncations are orthogonal projections).

#include <cmath>
#include <string>

#include <parthenon/package.hpp>

#include "kinetics/equilibrium.hpp"
#include "kinetics/tt_linalg.hpp"
#include "kinetics/velocity_grid.hpp"

using namespace parthenon::package::prelude;

namespace Kinetics {
namespace TT {

// Largest rank supported by kernels that keep one rank-sized vector on the stack.
constexpr int kMaxRank = 64;

struct PtrData {
  Real *p;
  KOKKOS_INLINE_FUNCTION Real &operator()(const int n) const { return p[n]; }
};

struct TTLayout {
  int n[3];
  int rcap;

  KOKKOS_INLINE_FUNCTION int Slot1() const { return 2; }
  KOKKOS_INLINE_FUNCTION int Slot2() const { return 2 + n[0] * rcap; }
  KOKKOS_INLINE_FUNCTION int Slot3() const { return Slot2() + rcap * n[1] * rcap; }
  KOKKOS_INLINE_FUNCTION int Size() const { return Slot3() + rcap * n[2]; }
};

// Host check that a per-team level-1 scratch request fits the device limit.
inline void RequireTeamScratch(const std::size_t bytes, const char *what) {
  const std::size_t cap = Kokkos::TeamPolicy<DevExecSpace>::scratch_size_max(1);
  PARTHENON_REQUIRE(bytes <= cap, std::string("kinetics TT: ") + what + " needs " +
                                      std::to_string(bytes) +
                                      " bytes of team scratch per cell, above the "
                                      "device limit " +
                                      std::to_string(cap) +
                                      "; lower tt_rank_max or the velocity resolution");
}

inline TTLayout MakeLayout(const VelocityGrid &grid, const int rcap) {
  PARTHENON_REQUIRE(rcap >= 1, "kinetics: tensor-train rank capacity must be >= 1");
  return TTLayout{{grid.nv[0], grid.nv[1], grid.nv[2]}, rcap};
}

// Accessor of one cell of a Parthenon pack variable: data(n) = v(b, Var(n), k, j, i).
// Holds a reference to the pack, so it must not outlive the kernel's captured copy.
template <class Pack, class Var>
struct PackCell {
  const Pack &v;
  int b, k, j, i;
  KOKKOS_INLINE_FUNCTION Real &operator()(const int n) const {
    return v(b, Var(n), k, j, i);
  }
};

// View of one TT. The ranks are cached when the view is made (MakeRef) and kept in sync
// by SetRanks; another view of the same data that changes the ranks makes this one stale.
template <class Data>
struct TTRef {
  Data data;
  TTLayout L;
  mutable int r1 = 0, r2 = 0;

  KOKKOS_INLINE_FUNCTION int R1() const { return r1; }
  KOKKOS_INLINE_FUNCTION int R2() const { return r2; }
  KOKKOS_INLINE_FUNCTION void SetRanks(const int new_r1, const int new_r2) const {
    data(0) = new_r1;
    data(1) = new_r2;
    r1 = new_r1;
    r2 = new_r2;
  }
  KOKKOS_INLINE_FUNCTION Real &G1(const int i, const int a) const {
    return data(L.Slot1() + i + L.n[0] * a);
  }
  KOKKOS_INLINE_FUNCTION Real &G2(const int a, const int j, const int b) const {
    return data(L.Slot2() + a + r1 * (j + L.n[1] * b));
  }
  KOKKOS_INLINE_FUNCTION Real &G3(const int b, const int k) const {
    return data(L.Slot3() + b + r2 * k);
  }

  KOKKOS_INLINE_FUNCTION Real operator()(const int i, const int j, const int k) const {
    Real sum = 0.0;
    for (int b = 0; b < r2; ++b) {
      Real left = 0.0;
      for (int a = 0; a < r1; ++a)
        left += G1(i, a) * G2(a, j, b);
      sum += left * G3(b, k);
    }
    return sum;
  }
};

// View of an existing TT (reads the stored ranks).
template <class Data>
KOKKOS_INLINE_FUNCTION TTRef<Data> MakeRef(const Data &data, const TTLayout &L) {
  return TTRef<Data>{data, L, static_cast<int>(data(0)), static_cast<int>(data(1))};
}

// View of storage about to be written (ranks 0 until SetRanks; nothing is read).
template <class Data>
KOKKOS_INLINE_FUNCTION TTRef<Data> MakeOutRef(const Data &data, const TTLayout &L) {
  return TTRef<Data>{data, L, 0, 0};
}

// Copy src into dst; dst.L.rcap must be >= the ranks of src.
template <class D1, class D2>
KOKKOS_INLINE_FUNCTION void CopyTT(const TTRef<D1> &src, const TTRef<D2> &dst) {
  const int r1 = src.R1(), r2 = src.R2();
  PARTHENON_DEBUG_REQUIRE(r1 <= dst.L.rcap && r2 <= dst.L.rcap,
                          "kinetics TT: CopyTT ranks exceed destination capacity");
  dst.SetRanks(r1, r2);
  for (int a = 0; a < r1; ++a)
    for (int i = 0; i < src.L.n[0]; ++i)
      dst.G1(i, a) = src.G1(i, a);
  for (int b = 0; b < r2; ++b)
    for (int j = 0; j < src.L.n[1]; ++j)
      for (int a = 0; a < r1; ++a)
        dst.G2(a, j, b) = src.G2(a, j, b);
  for (int k = 0; k < src.L.n[2]; ++k)
    for (int b = 0; b < r2; ++b)
      dst.G3(b, k) = src.G3(b, k);
}

// Sum of count separable equilibria, sum_c scale_c exp(b0_c + sum_d A_{c,d}), as an
// exact TT of ranks (count, count): G1 = [m_1x ... m_cx], G2 = diag(m_cy), G3 = [m_cz].
// count = 1 is the rank-1 discrete equilibrium.
template <class Data>
KOKKOS_INLINE_FUNCTION void FillMaxwellians(const VelocityGrid &grid,
                                            const Maxwellian *eqs, const int count,
                                            const TTRef<Data> &t) {
  PARTHENON_DEBUG_REQUIRE(count <= t.L.rcap,
                          "kinetics TT: FillMaxwellians count exceeds rank capacity");
  t.SetRanks(count, count);
  for (int c = 0; c < count; ++c) {
    const Maxwellian &eq = eqs[c];
    const Real c0 = eq.scale * std::exp(eq.b0);
    for (int i = 0; i < grid.nv[0]; ++i)
      t.G1(i, c) = c0 * std::exp(eq.AxisExponent(0, grid.Node(0, i)));
    for (int b = 0; b < count; ++b)
      for (int j = 0; j < grid.nv[1]; ++j)
        t.G2(c, j, b) = (b == c) ? std::exp(eq.AxisExponent(1, grid.Node(1, j))) : 0.0;
    for (int k = 0; k < grid.nv[2]; ++k)
      t.G3(c, k) = std::exp(eq.AxisExponent(2, grid.Node(2, k)));
  }
}

template <class Data>
KOKKOS_INLINE_FUNCTION void FillMaxwellian(const VelocityGrid &grid, const Maxwellian &eq,
                                           const TTRef<Data> &t) {
  FillMaxwellians(grid, &eq, 1, t);
}

// Call func(i, j, k, value) for every node, in (j, i, k) loop order. O(n^2 r1 r2 + n^3
// r2).
template <class Data, class Func>
KOKKOS_INLINE_FUNCTION void ForEachNode(const TTRef<Data> &t, const Func &func) {
  Real left[kMaxRank];
  const int r1 = t.R1(), r2 = t.R2();
  PARTHENON_DEBUG_REQUIRE(r1 <= kMaxRank && r2 <= kMaxRank,
                          "kinetics TT: ForEachNode rank above kMaxRank");
  for (int j = 0; j < t.L.n[1]; ++j)
    for (int i = 0; i < t.L.n[0]; ++i) {
      for (int b = 0; b < r2; ++b) {
        Real l = 0.0;
        for (int a = 0; a < r1; ++a)
          l += t.G1(i, a) * t.G2(a, j, b);
        left[b] = l;
      }
      for (int k = 0; k < t.L.n[2]; ++k) {
        Real v = 0.0;
        for (int b = 0; b < r2; ++b)
          v += left[b] * t.G3(b, k);
        func(i, j, k, v);
      }
    }
}

// Frobenius inner product <A, B> = sum_{ijk} A(i, j, k) B(i, j, k) by core contraction,
// O(n r^3): X(a, b) = sum_i A1(i, a) B1(i, b); W(a, j, d) = sum_b X(a, b) B2(b, j, d);
// Y(c, d) = sum_{a, j} A2(a, j, c) W(a, j, d); then the last core. work holds
// DotWorkSize(A.L, B.L) reals.
KOKKOS_INLINE_FUNCTION int DotWorkSize(const TTLayout &LA, const TTLayout &LB) {
  return 2 * LA.rcap * LB.rcap + LA.rcap * LA.n[1] * LB.rcap;
}
template <class DA, class DB>
KOKKOS_INLINE_FUNCTION Real Dot(const TTRef<DA> &A, const TTRef<DB> &B, Real *work) {
  const int a1 = A.R1(), a2 = A.R2(), b1 = B.R1(), b2 = B.R2();
  const int n1 = A.L.n[1];
  Real *X = work;
  Real *Y = X + A.L.rcap * B.L.rcap;
  Real *W = Y + A.L.rcap * B.L.rcap;
  for (int a = 0; a < a1; ++a)
    for (int b = 0; b < b1; ++b) {
      Real x = 0.0;
      for (int i = 0; i < A.L.n[0]; ++i)
        x += A.G1(i, a) * B.G1(i, b);
      X[a + a1 * b] = x;
    }
  for (int d = 0; d < b2; ++d)
    for (int j = 0; j < n1; ++j)
      for (int a = 0; a < a1; ++a) {
        Real w = 0.0;
        for (int b = 0; b < b1; ++b)
          w += X[a + a1 * b] * B.G2(b, j, d);
        W[a + a1 * (j + n1 * d)] = w;
      }
  for (int c = 0; c < a2; ++c)
    for (int d = 0; d < b2; ++d) {
      Real y = 0.0;
      for (int j = 0; j < n1; ++j)
        for (int a = 0; a < a1; ++a)
          y += A.G2(a, j, c) * W[a + a1 * (j + n1 * d)];
      Y[c + a2 * d] = y;
    }
  Real sum = 0.0;
  for (int k = 0; k < A.L.n[2]; ++k)
    for (int c = 0; c < a2; ++c)
      for (int d = 0; d < b2; ++d)
        sum += A.G3(c, k) * Y[c + a2 * d] * B.G3(d, k);
  return sum;
}

// Frobenius norm by full contraction, O(n^3 r^2). For tests and diagnostics only.
template <class Data>
KOKKOS_INLINE_FUNCTION Real NormSlow(const TTRef<Data> &t) {
  Real sum = 0.0;
  for (int k = 0; k < t.L.n[2]; ++k)
    for (int j = 0; j < t.L.n[1]; ++j)
      for (int i = 0; i < t.L.n[0]; ++i) {
        const Real v = t(i, j, k);
        sum += v * v;
      }
  return std::sqrt(sum);
}

struct RoundParams {
  Real eps = 1.0e-8;         // relative Frobenius tolerance; 0 = fixed rank (cap only)
  Real rank_floor = 1.0e-14; // tails below rank_floor ||f|| are always dropped
  int rank_max = 0;          // rank cap below the layout capacity; 0 = capacity
};

struct RoundInfo {
  int r1, r2;
  bool cap_hit;   // a rank was cut to the capacity before the tolerance was met
  bool svd_ok;    // every Jacobi SVD converged
  Real discarded; // norm of the discarded part
  Real norm;      // norm of the input tensor
};

// Scratch layout for rounding an input of ranks up to rin into an output of capacity
// rout. All matrices are col-major.
struct RoundScratch {
  int n[3];
  int rin;

  KOKKOS_INLINE_FUNCTION int MaxN() const {
    int m = n[0];
    if (n[1] > m) m = n[1];
    if (n[2] > m) m = n[2];
    return m;
  }
  // Largest matrix: an unfolding of G2, (rin n1) x rin.
  KOKKOS_INLINE_FUNCTION int Big() const {
    const int a = rin * n[1] * rin;
    const int b = MaxN() * rin;
    return (a > b) ? a : b;
  }
  KOKKOS_INLINE_FUNCTION int Small() const { return rin * rin; }
  // Input TT (capacity rin) + 3 big + 3 small + 2 vectors.
  KOKKOS_INLINE_FUNCTION int InputSize() const {
    return TTLayout{{n[0], n[1], n[2]}, rin}.Size();
  }
  KOKKOS_INLINE_FUNCTION int Size() const {
    return InputSize() + 3 * Big() + 3 * Small() + 2 * (MaxN() > rin ? MaxN() : rin);
  }
};

inline RoundScratch MakeRoundScratch(const VelocityGrid &grid, const int rin) {
  return RoundScratch{{grid.nv[0], grid.nv[1], grid.nv[2]}, rin};
}

namespace impl {

// Thin QR of src (m x n): Q (m x k) and R (k x n), k = min(m, n). work holds m x n.
KOKKOS_INLINE_FUNCTION int ThinQR(const Mat src, Real *qout, Real *rout, Real *work,
                                  Real *tau) {
  const int m = src.m, n = src.n;
  const int k = (m < n) ? m : n;
  const Mat A = ColMajor(work, m, n);
  Copy(src, A);
  HouseholderQR(A, tau);
  const Mat R = ColMajor(rout, k, n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < k; ++i)
      R(i, j) = (i <= j) ? A(i, j) : 0.0;
  const Mat Q = ColMajor(qout, m, k);
  SetIdentity(Q);
  ApplyQ(A, tau, k, Q);
  return k;
}

// Thin SVD of src (m x n): U (m x q), s (q), SVt = diag(s) V^T (q x n), q = min(m, n).
// Tall case: src = Q R, R = W V^T after Jacobi (W = U_R diag(s)), U = Q U_R.
// Wide case: the same on src^T, with the roles of the factors swapped.
KOKKOS_INLINE_FUNCTION int ThinSVD(const Mat src, Real *uout, Real *s, Real *svt,
                                   Real *big1, Real *big2, Real *sm1, Real *sm2,
                                   Real *tau, bool &svd_ok) {
  const bool tall = src.m >= src.n;
  const Mat A = tall ? src : Mat{src.p, src.n, src.m, src.cs, src.rs};
  const int m = A.m, n = A.n; // m >= n
  // Q (m x n) into big2, R (n x n) into sm1.
  ThinQR(A, big2, sm1, big1, tau);
  const Mat W = ColMajor(sm1, n, n);
  const Mat V = ColMajor(sm2, n, n);
  if (JacobiSVD(W, V, s) < 0) svd_ok = false;
  // U_R = W diag(1/s) in place (zero columns stay zero).
  for (int j = 0; j < n; ++j) {
    const Real inv = (s[j] > 0.0) ? 1.0 / s[j] : 0.0;
    for (int i = 0; i < n; ++i)
      W(i, j) *= inv;
  }
  const Mat Q = ColMajor(big2, m, n);
  // Left factor of A: Q U_R (m x n); right factor: V (n x n).
  if (tall) {
    const Mat U = ColMajor(uout, m, n);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        Real v = 0.0;
        for (int l = 0; l < n; ++l)
          v += Q(i, l) * W(l, j);
        U(i, j) = v;
      }
    const Mat B = ColMajor(svt, n, n); // diag(s) V^T
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        B(i, j) = s[i] * V(j, i);
  } else {
    // src = A^T = V diag(s) (Q U_R)^T: U_src = V (n x n), SVt_src = diag(s) (Q U_R)^T.
    const Mat U = ColMajor(uout, n, n);
    Copy(V, U);
    const Mat B = ColMajor(svt, n, m);
    for (int j = 0; j < m; ++j)
      for (int i = 0; i < n; ++i) {
        Real v = 0.0;
        for (int l = 0; l < n; ++l)
          v += Q(j, l) * W(l, i);
        B(i, j) = s[i] * v;
      }
  }
  return n;
}

// Smallest rank r (1 <= r <= q) whose discarded tail has norm <= delta, capped at rmax.
KOKKOS_INLINE_FUNCTION int ChooseRank(const Real *s, const int q, const Real delta,
                                      const int rmax, bool &cap_hit, Real &tail2) {
  int r = q;
  Real acc = 0.0;
  while (r > 1 && acc + s[r - 1] * s[r - 1] <= delta * delta) {
    acc += s[r - 1] * s[r - 1];
    --r;
  }
  if (r > rmax) {
    cap_hit = true;
    for (int l = rmax; l < r; ++l)
      acc += s[l] * s[l];
    r = rmax;
  }
  tail2 += acc;
  return r;
}

} // namespace impl

// The scratch input slot (capacity sc.rin) at the start of work, as an output view:
// fill it with CopyTT or AddInto, then call Round.
KOKKOS_INLINE_FUNCTION TTRef<PtrData> InputRef(const RoundScratch &sc, Real *work) {
  return MakeOutRef(PtrData{work}, TTLayout{{sc.n[0], sc.n[1], sc.n[2]}, sc.rin});
}

// Write alpha A + beta B into the scratch input slot as a block TT of ranks
// (rA1 + rB1, rA2 + rB2): G1 = [alpha A1, beta B1], G2 = blockdiag(A2, B2), G3 = [A3;
// B3]. The summed ranks must not exceed sc.rin.
template <class DA, class DB>
KOKKOS_INLINE_FUNCTION void AddInto(const RoundScratch &sc, Real *work, const Real alpha,
                                    const TTRef<DA> &A, const Real beta,
                                    const TTRef<DB> &B) {
  const TTRef<PtrData> in = InputRef(sc, work);
  const int a1 = A.R1(), a2 = A.R2(), b1 = B.R1(), b2 = B.R2();
  const int r1 = a1 + b1, r2 = a2 + b2;
  PARTHENON_DEBUG_REQUIRE(r1 <= sc.rin && r2 <= sc.rin,
                          "kinetics TT: AddInto ranks exceed scratch capacity");
  in.SetRanks(r1, r2);
  for (int i = 0; i < sc.n[0]; ++i) {
    for (int a = 0; a < a1; ++a)
      in.G1(i, a) = alpha * A.G1(i, a);
    for (int a = 0; a < b1; ++a)
      in.G1(i, a1 + a) = beta * B.G1(i, a);
  }
  for (int b = 0; b < r2; ++b)
    for (int j = 0; j < sc.n[1]; ++j)
      for (int a = 0; a < r1; ++a) {
        Real v = 0.0;
        if (a < a1 && b < a2) v = A.G2(a, j, b);
        if (a >= a1 && b >= a2) v = B.G2(a - a1, j, b - a2);
        in.G2(a, j, b) = v;
      }
  for (int k = 0; k < sc.n[2]; ++k) {
    for (int b = 0; b < a2; ++b)
      in.G3(b, k) = A.G3(b, k);
    for (int b = 0; b < b2; ++b)
      in.G3(a2 + b, k) = B.G3(b, k);
  }
}

// Value of node index n of core `axis` scaled by w(n): G1(n, a), G2(a, n, b), G3(b, n).
// Write wA(n) A + wB(n) B (both scalings on velocity axis `axis`, i.e. f(v) times a
// function of v_axis) into the scratch input slot as a block TT of ranks
// (rA1 + rB1, rA2 + rB2). Constant scalings give AddInto. The summed ranks must not
// exceed sc.rin.
template <class DA, class WA, class DB, class WB>
KOKKOS_INLINE_FUNCTION void
AddScaledInto(const RoundScratch &sc, Real *work, const int axis, const TTRef<DA> &A,
              const WA &wA, const TTRef<DB> &B, const WB &wB) {
  const TTRef<PtrData> in = InputRef(sc, work);
  const int a1 = A.R1(), a2 = A.R2(), b1 = B.R1(), b2 = B.R2();
  const int r1 = a1 + b1, r2 = a2 + b2;
  PARTHENON_DEBUG_REQUIRE(r1 <= sc.rin && r2 <= sc.rin,
                          "kinetics TT: AddScaledInto ranks exceed scratch capacity");
  in.SetRanks(r1, r2);
  for (int i = 0; i < sc.n[0]; ++i) {
    const Real sa = (axis == 0) ? wA(i) : 1.0, sb = (axis == 0) ? wB(i) : 1.0;
    for (int a = 0; a < a1; ++a)
      in.G1(i, a) = sa * A.G1(i, a);
    for (int a = 0; a < b1; ++a)
      in.G1(i, a1 + a) = sb * B.G1(i, a);
  }
  for (int b = 0; b < r2; ++b)
    for (int j = 0; j < sc.n[1]; ++j) {
      const Real sa = (axis == 1) ? wA(j) : 1.0, sb = (axis == 1) ? wB(j) : 1.0;
      for (int a = 0; a < r1; ++a) {
        Real v = 0.0;
        if (a < a1 && b < a2) v = sa * A.G2(a, j, b);
        if (a >= a1 && b >= a2) v = sb * B.G2(a - a1, j, b - a2);
        in.G2(a, j, b) = v;
      }
    }
  for (int k = 0; k < sc.n[2]; ++k) {
    const Real sa = (axis == 2) ? wA(k) : 1.0, sb = (axis == 2) ? wB(k) : 1.0;
    for (int b = 0; b < a2; ++b)
      in.G3(b, k) = sa * A.G3(b, k);
    for (int b = 0; b < b2; ++b)
      in.G3(a2 + b, k) = sb * B.G3(b, k);
  }
}

// dst = src with the node order of velocity axis `axis` reversed (v_axis -> -v_axis on
// a grid symmetric about 0). dst.L.rcap must be >= the ranks of src.
template <class D1, class D2>
KOKKOS_INLINE_FUNCTION void ReverseAxis(const TTRef<D1> &src, const TTRef<D2> &dst,
                                        const int axis) {
  const int r1 = src.R1(), r2 = src.R2();
  const int n0 = src.L.n[0], n1 = src.L.n[1], n2 = src.L.n[2];
  PARTHENON_DEBUG_REQUIRE(r1 <= dst.L.rcap && r2 <= dst.L.rcap,
                          "kinetics TT: ReverseAxis ranks exceed destination capacity");
  dst.SetRanks(r1, r2);
  for (int a = 0; a < r1; ++a)
    for (int i = 0; i < n0; ++i)
      dst.G1(i, a) = src.G1((axis == 0) ? n0 - 1 - i : i, a);
  for (int b = 0; b < r2; ++b)
    for (int j = 0; j < n1; ++j)
      for (int a = 0; a < r1; ++a)
        dst.G2(a, j, b) = src.G2(a, (axis == 1) ? n1 - 1 - j : j, b);
  for (int k = 0; k < n2; ++k)
    for (int b = 0; b < r2; ++b)
      dst.G3(b, k) = src.G3(b, (axis == 2) ? n2 - 1 - k : k);
}

// sum_{ijk} wx(i) wy(j) wz(k) t(i, j, k) for separable node weights, O(n r1 r2).
template <class Data, class WX, class WY, class WZ>
KOKKOS_INLINE_FUNCTION Real Contract(const TTRef<Data> &t, const WX &wx, const WY &wy,
                                     const WZ &wz) {
  const int r1 = t.R1(), r2 = t.R2();
  PARTHENON_DEBUG_REQUIRE(r1 <= kMaxRank, "kinetics TT: Contract rank above kMaxRank");
  Real X[kMaxRank];
  for (int a = 0; a < r1; ++a) {
    Real x = 0.0;
    for (int i = 0; i < t.L.n[0]; ++i)
      x += wx(i) * t.G1(i, a);
    X[a] = x;
  }
  Real sum = 0.0;
  for (int b = 0; b < r2; ++b) {
    Real xy = 0.0;
    for (int j = 0; j < t.L.n[1]; ++j) {
      Real xg = 0.0;
      for (int a = 0; a < r1; ++a)
        xg += X[a] * t.G2(a, j, b);
      xy += wy(j) * xg;
    }
    Real z = 0.0;
    for (int k = 0; k < t.L.n[2]; ++k)
      z += wz(k) * t.G3(b, k);
    sum += xy * z;
  }
  return sum;
}

template <class Data>
KOKKOS_INLINE_FUNCTION RoundInfo Round(const RoundScratch &sc, Real *work,
                                       const TTRef<Data> &dst, const RoundParams &prm) {
  const TTRef<PtrData> in =
      MakeRef(PtrData{work}, TTLayout{{sc.n[0], sc.n[1], sc.n[2]}, sc.rin});
  const int n0 = sc.n[0], n1 = sc.n[1], n2 = sc.n[2];
  int R1 = in.R1(), R2 = in.R2();
  PARTHENON_DEBUG_REQUIRE(R1 <= sc.rin && R2 <= sc.rin,
                          "kinetics TT: Round input ranks exceed scratch capacity");
  PARTHENON_DEBUG_REQUIRE(dst.L.n[0] == n0 && dst.L.n[1] == n1 && dst.L.n[2] == n2,
                          "kinetics TT: Round scratch and destination sizes differ");
  Real *g1 = work + in.L.Slot1();
  Real *g2 = work + in.L.Slot2();
  Real *g3 = work + in.L.Slot3();
  Real *b1 = work + sc.InputSize();
  Real *b2 = b1 + sc.Big();
  Real *b3 = b2 + sc.Big();
  Real *s1 = b3 + sc.Big();
  Real *s2 = s1 + sc.Small();
  Real *s3 = s2 + sc.Small();
  const int vlen = (sc.MaxN() > sc.rin) ? sc.MaxN() : sc.rin;
  Real *tau = s3 + sc.Small();
  Real *sv = tau + vlen;

  // --- Right-to-left orthogonalization ---
  // G3^T (n2 x R2) = Q R:  G3 <- Q^T (k2 x n2),  G2 <- G2 R^T.
  const int k2 = impl::ThinQR(Mat{g3, n2, R2, R2, 1}, b2, s1, b1, tau);
  for (int k = 0; k < n2; ++k)
    for (int b = 0; b < k2; ++b)
      g3[b + k2 * k] = b2[k + n2 * b];
  {
    const Mat G2L = ColMajor(g2, R1 * n1, R2);
    const Mat Rm = ColMajor(s1, k2, R2);
    const Mat T = ColMajor(b1, R1 * n1, k2);
    for (int c = 0; c < k2; ++c)
      for (int i = 0; i < R1 * n1; ++i) {
        Real v = 0.0;
        for (int l = 0; l < R2; ++l)
          v += G2L(i, l) * Rm(c, l);
        T(i, c) = v;
      }
    Copy(T, ColMajor(g2, R1 * n1, k2));
  }
  R2 = k2;
  // G2 right unfolding (R1 x n1 R2), transposed (n1 R2 x R1) = Q R:
  //   G2 <- Q^T (k1 x n1 R2),  G1 <- G1 R^T.
  const int k1 = impl::ThinQR(Mat{g2, n1 * R2, R1, R1, 1}, b2, s1, b1, tau);
  for (int c = 0; c < n1 * R2; ++c)
    for (int a = 0; a < k1; ++a)
      g2[a + k1 * c] = b2[c + n1 * R2 * a];
  {
    const Mat G1 = ColMajor(g1, n0, R1);
    const Mat Rm = ColMajor(s1, k1, R1);
    const Mat T = ColMajor(b1, n0, k1);
    for (int c = 0; c < k1; ++c)
      for (int i = 0; i < n0; ++i) {
        Real v = 0.0;
        for (int l = 0; l < R1; ++l)
          v += G1(i, l) * Rm(c, l);
        T(i, c) = v;
      }
    Copy(T, ColMajor(g1, n0, k1));
  }
  R1 = k1;

  RoundInfo info{1, 1, false, true, 0.0, 0.0};
  Real norm2 = 0.0;
  for (int i = 0; i < n0 * R1; ++i)
    norm2 += g1[i] * g1[i];
  info.norm = std::sqrt(norm2);
  const int rmax =
      (prm.rank_max > 0 && prm.rank_max < dst.L.rcap) ? prm.rank_max : dst.L.rcap;
  if (info.norm == 0.0) {
    dst.SetRanks(1, 1);
    for (int i = 0; i < n0; ++i)
      dst.G1(i, 0) = 0.0;
    for (int j = 0; j < n1; ++j)
      dst.G2(0, j, 0) = 0.0;
    for (int k = 0; k < n2; ++k)
      dst.G3(0, k) = 0.0;
    return info;
  }
  Real delta = prm.eps * info.norm / std::sqrt(2.0);
  if (delta < prm.rank_floor * info.norm) delta = prm.rank_floor * info.norm;
  Real tail2 = 0.0;

  // --- Left-to-right truncation ---
  // G1 (n0 x R1) = U S V^T:  G1 <- U_r1,  G2 <- (S V^T)_r1 G2.
  const int q1 =
      impl::ThinSVD(ColMajor(g1, n0, R1), b3, sv, s3, b1, b2, s1, s2, tau, info.svd_ok);
  const int r1 = impl::ChooseRank(sv, q1, delta, rmax, info.cap_hit, tail2);
  {
    const Mat U = ColMajor(b3, n0, q1);
    const Mat B = ColMajor(s3, q1, R1);
    const Mat G2R = ColMajor(g2, R1, n1 * R2);
    for (int a = 0; a < r1; ++a)
      for (int i = 0; i < n0; ++i)
        dst.data(dst.L.Slot1() + i + n0 * a) = U(i, a);
    const Mat T = ColMajor(b1, r1, n1 * R2);
    for (int c = 0; c < n1 * R2; ++c)
      for (int a = 0; a < r1; ++a) {
        Real v = 0.0;
        for (int l = 0; l < R1; ++l)
          v += B(a, l) * G2R(l, c);
        T(a, c) = v;
      }
    Copy(T, ColMajor(g2, r1, n1 * R2));
  }
  // G2 left unfolding (r1 n1 x R2) = U S V^T:  G2 <- U_r2,  G3 <- (S V^T)_r2 G3.
  const int q2 = impl::ThinSVD(ColMajor(g2, r1 * n1, R2), b3, sv, s3, b1, b2, s1, s2, tau,
                               info.svd_ok);
  const int r2 = impl::ChooseRank(sv, q2, delta, rmax, info.cap_hit, tail2);
  {
    const Mat U = ColMajor(b3, r1 * n1, q2);
    const Mat B = ColMajor(s3, q2, R2);
    const Mat G3 = ColMajor(g3, R2, n2);
    for (int b = 0; b < r2; ++b)
      for (int i = 0; i < r1 * n1; ++i)
        dst.data(dst.L.Slot2() + i + r1 * n1 * b) = U(i, b);
    for (int k = 0; k < n2; ++k)
      for (int b = 0; b < r2; ++b) {
        Real v = 0.0;
        for (int l = 0; l < R2; ++l)
          v += B(b, l) * G3(l, k);
        dst.data(dst.L.Slot3() + b + r2 * k) = v;
      }
  }
  dst.SetRanks(r1, r2);
  info.r1 = r1;
  info.r2 = r2;
  info.discarded = std::sqrt(tail2);
  return info;
}

} // namespace TT
} // namespace Kinetics

#endif // KINETICS_TT_TENSOR_HPP_
