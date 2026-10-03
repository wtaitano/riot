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
#ifndef KINETICS_TT_STREAM_HPP_
#define KINETICS_TT_STREAM_HPP_
// This file was made in part with generative AI.

// Semi-Lagrangian streaming of one cell along one spatial direction d in the
// tensor-train representation. The dense update (semi_lagrangian.hpp) is, per velocity
// node,
//
//   f*_i(v) = w_-(v) f_{i-1}(v) + w_0(v) f_i(v) + w_+(v) f_{i+1}(v),
//
// with weights that depend on v only through v_d (shift s = v_d h / dx). Multiplying a
// TT by a function of v_d scales the core of axis d, so each term keeps its rank, and
// the update is two block sums, each rounded (S1_DESIGN.md, S1-Q4: round after every
// elementary operation):
//
//   g  = round(w_0 f_i + w_- f_{i-1}),       ranks <= 2 rcap before rounding
//   f* = round(g + w_+ f_{i+1}).
//
// Without rounding this equals the dense update exactly. Only the unlimited weights are
// available (a limiter has no TT form).

#include <parthenon/package.hpp>

#include "kinetics/semi_lagrangian.hpp"
#include "kinetics/tt_tensor.hpp"
#include "kinetics/velocity_grid.hpp"

using namespace parthenon::package::prelude;

namespace Kinetics {
namespace TT {

// Scratch of one streamed cell: rounding at rank 2 rcap, then the intermediate g.
struct StreamScratch {
  RoundScratch round;
  TTLayout tmp;

  KOKKOS_INLINE_FUNCTION int Size() const { return round.Size() + tmp.Size(); }
};

inline StreamScratch MakeStreamScratch(const VelocityGrid &grid, const int rcap) {
  return StreamScratch{MakeRoundScratch(grid, 2 * rcap), MakeLayout(grid, rcap)};
}

// Rounding statistics accumulated over several roundings of one cell.
struct RoundTally {
  Real rel_discarded = 0.0; // sum of discarded / ||input||
  int cap_hits = 0;
  int svd_failures = 0;

  KOKKOS_INLINE_FUNCTION void Add(const RoundInfo &r) {
    if (r.norm > 0.0) rel_discarded += r.discarded / r.norm;
    cap_hits += r.cap_hit ? 1 : 0;
    svd_failures += r.svd_ok ? 0 : 1;
  }
};

// out <- SL(f) of one cell along velocity/space axis d. lo, mid, hi are the cells at
// offsets -1, 0, +1 along d; h_over_dx = h / dx_d. out must not alias the inputs.
template <class DL, class DM, class DH, class DO>
KOKKOS_INLINE_FUNCTION void
StreamCellTT(const VelocityGrid &grid, const int order, const int d, const Real h_over_dx,
             const TTRef<DL> &lo, const TTRef<DM> &mid, const TTRef<DH> &hi,
             const TTRef<DO> &out, const StreamScratch &sc, Real *work,
             const RoundParams &prm, RoundTally &tally) {
  auto weight = [&](const int slot) {
    return [&, slot](const int n) {
      Real w[3];
      SLWeights(order, grid.Node(d, n) * h_over_dx, w);
      return w[slot];
    };
  };
  const auto g = MakeOutRef(PtrData{work + sc.round.Size()}, sc.tmp);
  AddScaledInto(sc.round, work, d, mid, weight(1), lo, weight(0));
  tally.Add(Round(sc.round, work, g, prm));
  const auto one = [](const int) { return 1.0; };
  AddScaledInto(sc.round, work, d, MakeRef(g.data, g.L), one, hi, weight(2));
  tally.Add(Round(sc.round, work, out, prm));
}

} // namespace TT
} // namespace Kinetics

#endif // KINETICS_TT_STREAM_HPP_
