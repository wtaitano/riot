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
#ifndef KINETICS_TT_LINALG_HPP_
#define KINETICS_TT_LINALG_HPP_
// This file was made in part with generative AI.

// Small dense linear algebra for tensor-train rounding, callable inside device kernels.
//
// One thread works on one small matrix (tens of rows/columns for the ranks, up to
// Nv * rank rows), so the routines are serial and allocation free. Matrices are strided
// views on caller-owned memory:
//
//   * HouseholderQR: A = Q R in place (LAPACK geqrf layout: R on and above the
//     diagonal, unit-lower reflectors below it, scalar factors in tau).
//   * ApplyQ: C <- Q C from the stored reflectors (ormqr, no transpose).
//   * JacobiSVD: one-sided (Hestenes) Jacobi. On exit the columns of W are mutually
//     orthogonal with norms s (sorted descending) and W_in V = W_out, V orthogonal. With
//     U = W diag(1/s), W_in = U diag(s) V^T. Applied to the small R factor of a QR, this
//     gives the singular values to high relative accuracy.

#include <cmath>

#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

namespace Kinetics {
namespace TT {

// Strided view of an m x n matrix: element (i, j) is p[i * rs + j * cs].
struct Mat {
  Real *p;
  int m, n;
  int rs, cs;

  KOKKOS_INLINE_FUNCTION Real &operator()(const int i, const int j) const {
    return p[i * rs + j * cs];
  }
};

// Column-major (contiguous columns) m x n view.
KOKKOS_INLINE_FUNCTION Mat ColMajor(Real *p, const int m, const int n) {
  return Mat{p, m, n, 1, m};
}

KOKKOS_INLINE_FUNCTION void Copy(const Mat src, const Mat dst) {
  for (int j = 0; j < src.n; ++j)
    for (int i = 0; i < src.m; ++i)
      dst(i, j) = src(i, j);
}

// Set dst to the leading m x n block of the identity.
KOKKOS_INLINE_FUNCTION void SetIdentity(const Mat dst) {
  for (int j = 0; j < dst.n; ++j)
    for (int i = 0; i < dst.m; ++i)
      dst(i, j) = (i == j) ? 1.0 : 0.0;
}

// In-place Householder QR of A (m x n). Uses k = min(m, n) reflectors; tau has length k.
// H_j = I - tau_j v_j v_j^T with v_j(j) = 1 and v_j(i > j) stored in A(i, j).
KOKKOS_INLINE_FUNCTION void HouseholderQR(const Mat A, Real *tau) {
  const int k = (A.m < A.n) ? A.m : A.n;
  for (int j = 0; j < k; ++j) {
    Real sigma = 0.0;
    for (int i = j + 1; i < A.m; ++i)
      sigma += A(i, j) * A(i, j);
    const Real x0 = A(j, j);
    if (sigma == 0.0) {
      tau[j] = 0.0; // already upper triangular in this column
      continue;
    }
    const Real norm = std::sqrt(x0 * x0 + sigma);
    const Real beta = (x0 >= 0.0) ? -norm : norm;
    tau[j] = (beta - x0) / beta;
    const Real scale = 1.0 / (x0 - beta);
    for (int i = j + 1; i < A.m; ++i)
      A(i, j) *= scale;
    A(j, j) = beta;
    for (int c = j + 1; c < A.n; ++c) {
      Real w = A(j, c);
      for (int i = j + 1; i < A.m; ++i)
        w += A(i, j) * A(i, c);
      w *= tau[j];
      A(j, c) -= w;
      for (int i = j + 1; i < A.m; ++i)
        A(i, c) -= w * A(i, j);
    }
  }
}

// C <- Q C, Q = H_0 H_1 ... H_{k-1} from HouseholderQR(A). C has A.m rows.
KOKKOS_INLINE_FUNCTION void ApplyQ(const Mat A, const Real *tau, const int k,
                                   const Mat C) {
  for (int j = k - 1; j >= 0; --j) {
    if (tau[j] == 0.0) continue;
    for (int c = 0; c < C.n; ++c) {
      Real w = C(j, c);
      for (int i = j + 1; i < A.m; ++i)
        w += A(i, j) * C(i, c);
      w *= tau[j];
      C(j, c) -= w;
      for (int i = j + 1; i < A.m; ++i)
        C(i, c) -= w * A(i, j);
    }
  }
}

struct JacobiParams {
  Real tol = 1.0e-15; // relative orthogonality |w_p . w_q| <= tol ||w_p|| ||w_q|| (x m)
  int max_sweeps = 60;
};

// One-sided Jacobi SVD of W (m x n, overwritten). V (n x n) is set to the accumulated
// rotations, s (length n) to the column norms. Columns are sorted by descending s.
// Returns the number of sweeps, or -1 if max_sweeps was reached.
KOKKOS_INLINE_FUNCTION int JacobiSVD(const Mat W, const Mat V, Real *s,
                                     const JacobiParams params = JacobiParams{}) {
  const int n = W.n;
  SetIdentity(V);
  const Real tol = params.tol * ((W.m > 1) ? W.m : 1);
  int sweeps = -1;
  for (int sweep = 0; sweep < params.max_sweeps; ++sweep) {
    bool rotated = false;
    for (int p = 0; p < n - 1; ++p) {
      for (int q = p + 1; q < n; ++q) {
        Real alpha = 0.0, beta = 0.0, gamma = 0.0;
        for (int i = 0; i < W.m; ++i) {
          alpha += W(i, p) * W(i, p);
          beta += W(i, q) * W(i, q);
          gamma += W(i, p) * W(i, q);
        }
        if (gamma == 0.0 ||
            std::abs(gamma) <= tol * std::sqrt(alpha) * std::sqrt(beta))
          continue;
        rotated = true;
        const Real zeta = (beta - alpha) / (2.0 * gamma);
        // t = sign(zeta) / (|zeta| + sqrt(1 + zeta^2)); 1 / (2 zeta) once zeta^2 would
        // overflow (the two agree to roundoff there).
        const Real az = std::abs(zeta);
        const Real t = ((zeta >= 0.0) ? 1.0 : -1.0) *
                       ((az > 1.0e150) ? 0.5 / az : 1.0 / (az + std::sqrt(1.0 + az * az)));
        const Real c = 1.0 / std::sqrt(1.0 + t * t);
        const Real sn = c * t;
        for (int i = 0; i < W.m; ++i) {
          const Real wp = W(i, p), wq = W(i, q);
          W(i, p) = c * wp - sn * wq;
          W(i, q) = sn * wp + c * wq;
        }
        for (int i = 0; i < n; ++i) {
          const Real vp = V(i, p), vq = V(i, q);
          V(i, p) = c * vp - sn * vq;
          V(i, q) = sn * vp + c * vq;
        }
      }
    }
    if (!rotated) {
      sweeps = sweep + 1;
      break;
    }
  }
  for (int j = 0; j < n; ++j) {
    Real norm2 = 0.0;
    for (int i = 0; i < W.m; ++i)
      norm2 += W(i, j) * W(i, j);
    s[j] = std::sqrt(norm2);
  }
  // Selection sort, descending, permuting the columns of W and V with s.
  for (int j = 0; j < n - 1; ++j) {
    int jmax = j;
    for (int l = j + 1; l < n; ++l)
      if (s[l] > s[jmax]) jmax = l;
    if (jmax == j) continue;
    const Real st = s[j];
    s[j] = s[jmax];
    s[jmax] = st;
    for (int i = 0; i < W.m; ++i) {
      const Real t = W(i, j);
      W(i, j) = W(i, jmax);
      W(i, jmax) = t;
    }
    for (int i = 0; i < n; ++i) {
      const Real t = V(i, j);
      V(i, j) = V(i, jmax);
      V(i, jmax) = t;
    }
  }
  return sweeps;
}

} // namespace TT
} // namespace Kinetics

#endif // KINETICS_TT_LINALG_HPP_
