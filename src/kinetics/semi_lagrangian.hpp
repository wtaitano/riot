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
#ifndef KINETICS_SEMI_LAGRANGIAN_HPP_
#define KINETICS_SEMI_LAGRANGIAN_HPP_
// This file was made in part with generative AI.

// Nodal backward semi-Lagrangian streaming, df/dt + v . grad f = 0, over one step h.
//
// For a velocity node v the departure point of cell center x_i is x_i - v h; in units of
// the cell width that is a shift s = v h / dx, and the substep cap keeps |s| <= 1, so the
// departure point lies between cell i and its upwind neighbor i - sgn(s). The new value
// is an interpolation of the three cells i-1, i, i+1 with weights w(-1), w(0), w(+1):
//
//   linear (order 1):    f* = (1 - |s|) f_i + |s| f_{i - sgn s}
//   quadratic (order 2): f* = f_i - (s/2)(f_{i+1} - f_{i-1}) + (s^2/2)(f_{i+1} - 2 f_i
//                             + f_{i-1})
//
// With |s| <= 1 these are exactly first-order upwind and Lax-Wendroff, so the unlimited
// schemes conserve sum_i f_i on a periodic domain. In several dimensions the weights are
// a tensor product of the 1D weights (one pass, 3^d-point stencil). The optional limiter
// clips f* to the range of the 2^d cells bracketing the departure point, which removes
// the new extrema of the quadratic scheme at the cost of exact conservation.

#include <cmath>

#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

namespace Kinetics {

// One-dimensional interpolation weights w[0..2] for offsets -1, 0, +1 and shift s.
KOKKOS_INLINE_FUNCTION void SLWeights(const int order, const Real s, Real w[3]) {
  if (order == 1) {
    const Real a = std::abs(s);
    w[0] = (s > 0.0) ? a : 0.0;
    w[1] = 1.0 - a;
    w[2] = (s < 0.0) ? a : 0.0;
  } else {
    w[0] = 0.5 * s * (1.0 + s);
    w[1] = 1.0 - s * s;
    w[2] = 0.5 * s * (s - 1.0);
  }
}

// Offset (-1 or 0) of the upwind bracketing cell relative to i, as an index into the
// weight arrays (0 or 1 for offsets -1, 0; 2 for +1). For s = 0 the departure point is
// the cell center itself and both bracketing cells are i.
KOKKOS_INLINE_FUNCTION int SLUpwindSlot(const Real s) {
  return (s > 0.0) ? 0 : ((s < 0.0) ? 2 : 1);
}

struct SLParams {
  int order;    // 1 or 2
  bool limiter; // clip to the bracketing cells
};

// Interpolated departure value of one velocity node at one cell. s[d] is the shift in
// cells along spatial direction d (only d < ndim are active). `f(a, b, c)` returns the
// value at offsets (a, b, c) in (x1, x2, x3), each in {-1, 0, 1}.
template <class F>
KOKKOS_INLINE_FUNCTION Real SLInterpolate(const SLParams &sl, const int ndim,
                                          const Real s[3], const F &f) {
  Real w[3][3];
  int lo[3], hi[3], up[3];
  for (int d = 0; d < 3; ++d) {
    if (d < ndim) {
      SLWeights(sl.order, s[d], w[d]);
      up[d] = SLUpwindSlot(s[d]) - 1; // offset of the upwind bracketing cell
      lo[d] = -1;
      hi[d] = 1;
    } else {
      w[d][0] = 0.0;
      w[d][1] = 1.0;
      w[d][2] = 0.0;
      up[d] = 0;
      lo[d] = 0;
      hi[d] = 0;
    }
  }
  Real val = 0.0;
  for (int c = lo[2]; c <= hi[2]; ++c)
    for (int b = lo[1]; b <= hi[1]; ++b)
      for (int a = lo[0]; a <= hi[0]; ++a)
        val += w[0][a + 1] * w[1][b + 1] * w[2][c + 1] * f(a, b, c);
  if (sl.limiter) {
    // Range of the 2^d cells bracketing the departure point: offsets 0 and up[d].
    Real fmin = f(0, 0, 0), fmax = fmin;
    for (int c = 0; c <= (up[2] != 0); ++c)
      for (int b = 0; b <= (up[1] != 0); ++b)
        for (int a = 0; a <= (up[0] != 0); ++a) {
          const Real fv = f(a * up[0], b * up[1], c * up[2]);
          fmin = (fv < fmin) ? fv : fmin;
          fmax = (fv > fmax) ? fv : fmax;
        }
    val = (val < fmin) ? fmin : ((val > fmax) ? fmax : val);
  }
  return val;
}

} // namespace Kinetics

#endif // KINETICS_SEMI_LAGRANGIAN_HPP_
