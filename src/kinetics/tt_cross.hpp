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
#ifndef KINETICS_TT_CROSS_HPP_
#define KINETICS_TT_CROSS_HPP_
// This file was made in part with generative AI.

// Cross approximation of a pointwise function of a tensor-train f, g = phi(f), as a TT,
// from a few node evaluations (S1_DESIGN.md, S1-Q14, Q19-Q24). Two DEIM projections:
//
//   mode 1:  C1 = g(:, J2)               (n0 x p1 fibers at the (y, z) pairs J2)
//            Q1 = orth(C1), I1 = QDEIM(Q1), G1 = Q1 Q1[I1]^-1,   g ~ G1 g(I1, :, :)
//   mode 3:  C2 = g(I1, :, J3)           ((q1 n1) x p2, row a + q1 j)
//            Q2 = orth(C2), I2 = QDEIM(Q2), G2 = Q2 Q2[I2]^-1,   g(I1, :, :) ~ G2 g(I2,
//            :)
//   G3 = g(I2, :)                        (q2 x n2 fibers along z)
//
// so g ~ G1 G2 G3, of ranks (q1, q2) <= (|J2|, |J3|). orth is pivoted modified
// Gram-Schmidt (columns below orth_tol of the largest are dropped), QDEIM is pivoted
// Gram-Schmidt on the rows. The initial J3 and J2 are the QDEIM points of f's own right
// interface bases (Q19). Greedy enrichment (Q20): the residual is measured on a fixed
// pseudo-random trial set of trial_factor * max(r1, r2) * max(n) nodes; the (y, z) pair
// of the worst node is added to J2 and its z to J3, until max |residual| <=
// eps max |g(trial)|, or a set reaches rank_max (cap hit), or nothing new can be added.
//
// phi(f, fscale) is called with fscale = max |f| over the trial set, so a phi can scale
// a regularization with f (the entropy one below). Cost per build: (n0 p1 + q1 n1 p2 +
// q2 n2) evaluations of f, each O(r1 r2); the matrices are tens of columns wide.
//
// The trial set and min f search are the same in every cell (fixed hash sequence), so
// results do not depend on the MPI decomposition.

#include <cmath>

#include <parthenon/package.hpp>

#include "kinetics/tt_linalg.hpp"
#include "kinetics/tt_tensor.hpp"
#include "kinetics/velocity_grid.hpp"

using namespace parthenon::package::prelude;

namespace Kinetics {
namespace TT {

struct CrossParams {
  Real eps = 1.0e-10;      // relative max-residual tolerance on the trial set
  int rank_max = 32;       // cap on |J2| and |J3| (and so on the ranks of g)
  int trial_factor = 16;   // trial nodes = trial_factor * max(r1, r2, 1) * max(n)
  Real orth_tol = 1.0e-14; // relative column-norm floor of orth
};

struct CrossInfo {
  int q1, q2;        // ranks of g
  int iters;         // greedy enrichments
  int evals;         // evaluations of phi
  bool cap_hit;      // stopped at rank_max before the tolerance was met
  bool converged;    // tolerance met on the trial set
  Real fmin, fscale; // min f and max |f| over the trial set
  Real residual;     // final max |residual| / max |g(trial)|
};

// Work arrays of one cross: Real and int parts, sized for f of capacity rcap.
struct CrossScratch {
  int n[3];
  int rcap;  // capacity rank of f
  int pmax;  // CrossParams::rank_max
  int ntmax; // largest trial set

  KOKKOS_INLINE_FUNCTION int MaxN() const {
    int m = n[0];
    if (n[1] > m) m = n[1];
    if (n[2] > m) m = n[2];
    return m;
  }
  KOKKOS_INLINE_FUNCTION int P() const { return (pmax > rcap) ? pmax : rcap; }
  KOKKOS_INLINE_FUNCTION int Big() const { return MaxN() * P() * P(); }
  KOKKOS_INLINE_FUNCTION TTLayout G() const { return TTLayout{{n[0], n[1], n[2]}, P()}; }
  // Real: g (TT), trial g values, 2 big matrices, small solve matrix, norms vector.
  KOKKOS_INLINE_FUNCTION int RealSize() const {
    return G().Size() + ntmax + 2 * Big() + P() * P() + MaxN() * P();
  }
  // int: trial (i, j, k), J3, J2 (j, k), I1, I2 (a, j), I2 as rows of C2.
  KOKKOS_INLINE_FUNCTION int IntSize() const { return 3 * ntmax + 7 * P(); }
};

inline CrossScratch MakeCrossScratch(const VelocityGrid &grid, const int rcap,
                                     const CrossParams &prm) {
  CrossScratch sc{{grid.nv[0], grid.nv[1], grid.nv[2]}, rcap, prm.rank_max, 0};
  sc.ntmax = prm.trial_factor * rcap * sc.MaxN();
  return sc;
}

namespace impl {

// Node index in [0, n) of trial point t, axis d (fixed hash sequence).
KOKKOS_INLINE_FUNCTION int TrialIndex(const int t, const int d, const int n) {
  unsigned int x = 2654435761u * static_cast<unsigned int>(3 * t + d + 1);
  x ^= x >> 16;
  x *= 0x45d9f3bu;
  x ^= x >> 16;
  return static_cast<int>(x % static_cast<unsigned int>(n));
}

// Pivoted modified Gram-Schmidt: the first q columns of A (m x p) become orthonormal
// (columns are permuted); columns whose residual norm is <= tol times the largest
// initial column norm are dropped. Returns q. nrm holds p reals.
KOKKOS_INLINE_FUNCTION int Orth(const Mat A, const Real tol, Real *nrm) {
  const int m = A.m, p = A.n;
  Real ref = 0.0;
  for (int c = 0; c < p; ++c) {
    Real s = 0.0;
    for (int i = 0; i < m; ++i)
      s += A(i, c) * A(i, c);
    nrm[c] = s;
    if (s > ref) ref = s;
  }
  if (ref == 0.0) return 0;
  const Real floor2 = tol * tol * ref;
  int q = 0;
  for (; q < p && q < m; ++q) {
    int cmax = q;
    for (int c = q + 1; c < p; ++c)
      if (nrm[c] > nrm[cmax]) cmax = c;
    if (nrm[cmax] <= floor2) break;
    if (cmax != q) {
      for (int i = 0; i < m; ++i) {
        const Real t = A(i, q);
        A(i, q) = A(i, cmax);
        A(i, cmax) = t;
      }
      const Real t = nrm[q];
      nrm[q] = nrm[cmax];
      nrm[cmax] = t;
    }
    // Reorthogonalize against the accepted columns (twice is enough), then normalize.
    for (int pass = 0; pass < 2; ++pass)
      for (int l = 0; l < q; ++l) {
        Real d = 0.0;
        for (int i = 0; i < m; ++i)
          d += A(i, l) * A(i, q);
        for (int i = 0; i < m; ++i)
          A(i, q) -= d * A(i, l);
      }
    Real s = 0.0;
    for (int i = 0; i < m; ++i)
      s += A(i, q) * A(i, q);
    if (s <= floor2) break;
    const Real inv = 1.0 / std::sqrt(s);
    for (int i = 0; i < m; ++i)
      A(i, q) *= inv;
    // Update the residual norms of the remaining columns.
    for (int c = q + 1; c < p; ++c) {
      Real d = 0.0;
      for (int i = 0; i < m; ++i)
        d += A(i, q) * A(i, c);
      for (int i = 0; i < m; ++i)
        A(i, c) -= d * A(i, q);
      Real t = 0.0;
      for (int i = 0; i < m; ++i)
        t += A(i, c) * A(i, c);
      nrm[c] = t;
    }
  }
  return q;
}

// QDEIM: q interpolation rows of the orthonormal Q (m x q) by pivoted Gram-Schmidt on
// its rows. W (m x q) is overwritten.
KOKKOS_INLINE_FUNCTION void QDEIM(const Mat Q, const Mat W, int *idx) {
  Copy(Q, W);
  for (int t = 0; t < Q.n; ++t) {
    int best = 0;
    Real bn = -1.0;
    for (int i = 0; i < W.m; ++i) {
      Real s = 0.0;
      for (int c = 0; c < W.n; ++c)
        s += W(i, c) * W(i, c);
      if (s > bn) {
        bn = s;
        best = i;
      }
    }
    PARTHENON_DEBUG_REQUIRE(bn > 0.0, "kinetics TT: QDEIM on a rank-deficient basis");
    idx[t] = best;
    const Real inv = (bn > 0.0) ? 1.0 / bn : 0.0;
    // Project the pivot row out of every row; the pivot row itself last.
    for (int l = 0; l <= W.m; ++l) {
      const int i = (l < W.m) ? l : best;
      if (l < W.m && i == best) continue;
      Real d = 0.0;
      for (int c = 0; c < W.n; ++c)
        d += W(i, c) * W(best, c);
      d *= inv;
      for (int c = 0; c < W.n; ++c)
        W(i, c) -= d * W(best, c);
    }
  }
}

// X = Q Q[idx]^-1 in place of Q (m x q), by Gaussian elimination with partial pivoting
// on Q[idx]^T X^T = Q^T. S holds q x q.
KOKKOS_INLINE_FUNCTION void InterpolationMatrix(const Mat Q, const int *idx, Real *S) {
  const int m = Q.m, q = Q.n;
  const Mat A = ColMajor(S, q, q);
  for (int c = 0; c < q; ++c)
    for (int r = 0; r < q; ++r)
      A(r, c) = Q(idx[c], r);         // A = Q[idx]^T
  const Mat B{Q.p, q, m, Q.cs, Q.rs}; // B = X^T (q x m), a transposed view of Q
  for (int c = 0; c < q; ++c) {
    int piv = c;
    for (int r = c + 1; r < q; ++r)
      if (std::abs(A(r, c)) > std::abs(A(piv, c))) piv = r;
    if (piv != c) {
      for (int l = 0; l < q; ++l) {
        const Real t = A(c, l);
        A(c, l) = A(piv, l);
        A(piv, l) = t;
      }
      for (int l = 0; l < m; ++l) {
        const Real t = B(c, l);
        B(c, l) = B(piv, l);
        B(piv, l) = t;
      }
    }
    const Real d = A(c, c);
    const Real inv = (d != 0.0) ? 1.0 / d : 0.0;
    for (int r = c + 1; r < q; ++r) {
      const Real fct = A(r, c) * inv;
      if (fct == 0.0) continue;
      for (int l = c; l < q; ++l)
        A(r, l) -= fct * A(c, l);
      for (int l = 0; l < m; ++l)
        B(r, l) -= fct * B(c, l);
    }
  }
  for (int c = q - 1; c >= 0; --c) {
    const Real inv = (A(c, c) != 0.0) ? 1.0 / A(c, c) : 0.0;
    for (int l = 0; l < m; ++l) {
      Real v = B(c, l);
      for (int r = c + 1; r < q; ++r)
        v -= A(c, r) * B(r, l);
      B(c, l) = v * inv;
    }
  }
}

} // namespace impl

// g ~ phi(f) as a TT in the scratch slot g = CrossOutput(sc, work), of ranks (q1, q2).
// work holds sc.RealSize() reals, iwork sc.IntSize() ints.
template <class Data, class Phi>
KOKKOS_INLINE_FUNCTION CrossInfo TTCross(const CrossScratch &sc, Real *work, int *iwork,
                                         const TTRef<Data> &f, const Phi &phi,
                                         const CrossParams &prm) {
  const int n0 = sc.n[0], n1 = sc.n[1], n2 = sc.n[2];
  const int P = sc.P();
  const TTLayout GL = sc.G();
  const TTRef<PtrData> g = MakeOutRef(PtrData{work}, GL);
  Real *gt = work + GL.Size();
  Real *A = gt + sc.ntmax;
  Real *W = A + sc.Big();
  Real *S = W + sc.Big();
  Real *nrm = S + P * P;
  int *ti = iwork, *tj = ti + sc.ntmax, *tk = tj + sc.ntmax;
  int *J3 = tk + sc.ntmax;
  int *J2j = J3 + P, *J2k = J2j + P;
  int *I1 = J2k + P;
  int *I2a = I1 + P, *I2j = I2a + P, *I2r = I2j + P;

  CrossInfo info{1, 1, 0, 0, false, false, 0.0, 0.0, 0.0};
  const int r1 = f.R1(), r2 = f.R2();
  const int rf = (r1 > r2) ? r1 : r2;
  int nt = prm.trial_factor * ((rf > 1) ? rf : 1) * sc.MaxN();
  if (nt > sc.ntmax) nt = sc.ntmax;

  // Trial set: f values first (fscale for phi), then g.
  Real fmin = 0.0, fscale = 0.0;
  for (int t = 0; t < nt; ++t) {
    ti[t] = impl::TrialIndex(t, 0, n0);
    tj[t] = impl::TrialIndex(t, 1, n1);
    tk[t] = impl::TrialIndex(t, 2, n2);
    const Real v = f(ti[t], tj[t], tk[t]);
    gt[t] = v;
    if (t == 0 || v < fmin) fmin = v;
    if (std::abs(v) > fscale) fscale = std::abs(v);
  }
  info.fmin = fmin;
  info.fscale = fscale;
  Real gscale = 0.0;
  for (int t = 0; t < nt; ++t) {
    gt[t] = phi(gt[t], fscale);
    if (std::abs(gt[t]) > gscale) gscale = std::abs(gt[t]);
  }
  int evals = nt;
  auto G = [&](const int i, const int j, const int k) {
    ++evals;
    return phi(f(i, j, k), fscale);
  };

  // Initial right sets from f's right interface bases.
  int p3 = 0, p2 = 0;
  {
    const Mat M = ColMajor(A, n2, r2); // G3^T
    for (int b = 0; b < r2; ++b)
      for (int k = 0; k < n2; ++k)
        M(k, b) = f.G3(b, k);
    const int q = impl::Orth(M, prm.orth_tol, nrm);
    impl::QDEIM(ColMajor(A, n2, q), ColMajor(W, n2, q), J3);
    p3 = q;
  }
  if (p3 == 0) {
    J3[0] = 0;
    p3 = 1;
  }
  if (p3 > prm.rank_max) p3 = prm.rank_max;
  {
    // R^T: rows j + n1 c (c over J3), columns a: sum_b G2(a, j, b) G3(b, J3[c]).
    const Mat M = ColMajor(A, n1 * p3, r1);
    for (int a = 0; a < r1; ++a)
      for (int c = 0; c < p3; ++c)
        for (int j = 0; j < n1; ++j) {
          Real v = 0.0;
          for (int b = 0; b < r2; ++b)
            v += f.G2(a, j, b) * f.G3(b, J3[c]);
          M(j + n1 * c, a) = v;
        }
    const int q = impl::Orth(M, prm.orth_tol, nrm);
    int *rows = I1; // temporary
    impl::QDEIM(ColMajor(A, n1 * p3, q), ColMajor(W, n1 * p3, q), rows);
    for (int c = 0; c < q; ++c) {
      J2j[c] = rows[c] % n1;
      J2k[c] = J3[rows[c] / n1];
    }
    p2 = q;
  }
  if (p2 == 0) {
    J2j[0] = 0;
    J2k[0] = J3[0];
    p2 = 1;
  }
  if (p2 > prm.rank_max) p2 = prm.rank_max;

  while (true) {
    // --- Build g from the current J2, J3 ---
    int q1, q2;
    {
      const Mat C = ColMajor(A, n0, p2);
      for (int c = 0; c < p2; ++c)
        for (int i = 0; i < n0; ++i)
          C(i, c) = G(i, J2j[c], J2k[c]);
      q1 = impl::Orth(C, prm.orth_tol, nrm);
    }
    if (q1 == 0) { // g = 0 on these fibers
      q1 = 1;
      q2 = 1;
      g.SetRanks(1, 1);
      for (int i = 0; i < n0; ++i)
        g.G1(i, 0) = 0.0;
      for (int j = 0; j < n1; ++j)
        g.G2(0, j, 0) = 0.0;
      for (int k = 0; k < n2; ++k)
        g.G3(0, k) = 0.0;
    } else {
      const Mat Q = ColMajor(A, n0, q1);
      impl::QDEIM(Q, ColMajor(W, n0, q1), I1);
      impl::InterpolationMatrix(Q, I1, S);
      g.SetRanks(q1, 1); // G1 slot does not depend on the ranks
      for (int a = 0; a < q1; ++a)
        for (int i = 0; i < n0; ++i)
          g.G1(i, a) = Q(i, a);
      const int m = q1 * n1;
      const Mat C = ColMajor(A, m, p3);
      for (int c = 0; c < p3; ++c)
        for (int j = 0; j < n1; ++j)
          for (int a = 0; a < q1; ++a)
            C(a + q1 * j, c) = G(I1[a], j, J3[c]);
      q2 = impl::Orth(C, prm.orth_tol, nrm);
      if (q2 == 0) {
        q2 = 1;
        g.SetRanks(q1, 1);
        for (int j = 0; j < n1; ++j)
          for (int a = 0; a < q1; ++a)
            g.G2(a, j, 0) = 0.0;
        for (int k = 0; k < n2; ++k)
          g.G3(0, k) = 0.0;
      } else {
        const Mat Q2 = ColMajor(A, m, q2);
        impl::QDEIM(Q2, ColMajor(W, m, q2), I2r);
        for (int b = 0; b < q2; ++b) {
          I2j[b] = I2r[b] / q1;
          I2a[b] = I2r[b] % q1;
        }
        impl::InterpolationMatrix(Q2, I2r, S);
        g.SetRanks(q1, q2);
        for (int b = 0; b < q2; ++b)
          for (int r = 0; r < m; ++r)
            g.data(GL.Slot2() + r + m * b) = Q2(r, b);
        for (int k = 0; k < n2; ++k)
          for (int b = 0; b < q2; ++b)
            g.G3(b, k) = G(I1[I2a[b]], I2j[b], k);
      }
    }
    info.q1 = q1;
    info.q2 = q2;

    // --- Residual on the trial set ---
    int worst = 0;
    Real res = 0.0;
    for (int t = 0; t < nt; ++t) {
      const Real e = std::abs(gt[t] - g(ti[t], tj[t], tk[t]));
      if (e > res) {
        res = e;
        worst = t;
      }
    }
    info.residual = (gscale > 0.0) ? res / gscale : 0.0;
    if (res <= prm.eps * gscale) {
      info.converged = true;
      break;
    }
    if (p2 >= prm.rank_max || p3 >= prm.rank_max) {
      info.cap_hit = true;
      break;
    }
    const int j = tj[worst], k = tk[worst];
    bool added = false, in3 = false, in2 = false;
    for (int c = 0; c < p3; ++c)
      in3 = in3 || (J3[c] == k);
    for (int c = 0; c < p2; ++c)
      in2 = in2 || (J2j[c] == j && J2k[c] == k);
    if (!in3) {
      J3[p3++] = k;
      added = true;
    }
    if (!in2) {
      J2j[p2] = j;
      J2k[p2++] = k;
      added = true;
    }
    if (!added) break; // stalled: the worst node's fibers are already in the sets
    ++info.iters;
  }
  info.evals = evals;
  return info;
}

// The output slot of TTCross.
KOKKOS_INLINE_FUNCTION TTRef<PtrData> CrossOutput(const CrossScratch &sc, Real *work) {
  return MakeRef(PtrData{work}, sc.G());
}

// C^1 positive part of f with transition width delta: f for f >= delta, 0 for
// f <= -delta, (f + delta)^2 / (4 delta) in between (s >= 0, s = f outside the band).
KOKKOS_INLINE_FUNCTION Real SmoothPositive(const Real f, const Real delta) {
  if (f >= delta) return f;
  if (f <= -delta) return 0.0;
  return (f + delta) * (f + delta) / (4.0 * delta);
}

// Entropy integrand of the cross diagnostic (S1-Q21): s (ln(s + delta) - 1), s the
// smooth positive part, delta = rel_delta * fscale. Differs from f (ln f - 1) for f > 0
// by O(delta |ln f|) only where f is within a few delta of 0.
struct EntropyPhi {
  Real rel_delta;
  KOKKOS_INLINE_FUNCTION Real operator()(const Real f, const Real fscale) const {
    const Real delta = (fscale > 0.0) ? rel_delta * fscale : 1.0e-300;
    const Real s = SmoothPositive(f, delta);
    return s * (std::log(s + delta) - 1.0);
  }
};

// Sampled min f (S1-Q21): min over the trial set, refined by coordinate descent along
// full fibers (x, y, z, repeated `sweeps` times) from its `nstart` smallest nodes. An
// upper bound on the true min. No scratch.
template <class Data>
KOKKOS_INLINE_FUNCTION Real SampledMinF(const TTRef<Data> &f, const int trial_factor,
                                        const int sweeps = 3) {
  constexpr int nstart = 4;
  const int n0 = f.L.n[0], n1 = f.L.n[1], n2 = f.L.n[2];
  int mx = n0;
  if (n1 > mx) mx = n1;
  if (n2 > mx) mx = n2;
  const int rf = (f.R1() > f.R2()) ? f.R1() : f.R2();
  const int nt = trial_factor * ((rf > 1) ? rf : 1) * mx;
  Real sv[nstart];
  int st[nstart];
  for (int s = 0; s < nstart; ++s) {
    sv[s] = 1.0e300;
    st[s] = -1;
  }
  for (int t = 0; t < nt; ++t) {
    const Real v = f(impl::TrialIndex(t, 0, n0), impl::TrialIndex(t, 1, n1),
                     impl::TrialIndex(t, 2, n2));
    // Insert into the sorted list of the nstart smallest.
    int s = nstart;
    while (s > 0 && v < sv[s - 1])
      --s;
    if (s == nstart) continue;
    for (int l = nstart - 1; l > s; --l) {
      sv[l] = sv[l - 1];
      st[l] = st[l - 1];
    }
    sv[s] = v;
    st[s] = t;
  }
  Real best = sv[0];
  for (int s = 0; s < nstart; ++s) {
    if (st[s] < 0) continue;
    int i = impl::TrialIndex(st[s], 0, n0), j = impl::TrialIndex(st[s], 1, n1),
        k = impl::TrialIndex(st[s], 2, n2);
    for (int sw = 0; sw < sweeps; ++sw) {
      Real m = 1.0e300;
      for (int l = 0; l < n0; ++l) {
        const Real v = f(l, j, k);
        if (v < m) {
          m = v;
          i = l;
        }
      }
      for (int l = 0; l < n1; ++l) {
        const Real v = f(i, l, k);
        if (v < m) {
          m = v;
          j = l;
        }
      }
      for (int l = 0; l < n2; ++l) {
        const Real v = f(i, j, l);
        if (v < m) {
          m = v;
          k = l;
        }
      }
      if (m < best) best = m;
    }
  }
  return best;
}

} // namespace TT
} // namespace Kinetics

#endif // KINETICS_TT_CROSS_HPP_
