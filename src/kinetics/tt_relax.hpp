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
#ifndef KINETICS_TT_RELAX_HPP_
#define KINETICS_TT_RELAX_HPP_
// This file was made in part with generative AI.

// BGK relaxation of one cell in the tensor-train representation:
//
//   f <- round((1 - c) f + c M),
//
// the same step as RelaxCell (bgk.hpp). The moments of f are exact core contractions, M
// is the discrete equilibrium with those moments (rank 1), so the sum, of ranks
// (r1 + 1, r2 + 1), has the moments of f to roundoff; the rounding then changes them by
// at most its truncation error (S1_DESIGN.md, S1-Q6). One rounding per relaxation.

#include <parthenon/package.hpp>

#include "kinetics/bgk.hpp"
#include "kinetics/equilibrium.hpp"
#include "kinetics/moments.hpp"
#include "kinetics/tt_moments.hpp"
#include "kinetics/tt_tensor.hpp"
#include "kinetics/velocity_grid.hpp"

using namespace parthenon::package::prelude;

namespace Kinetics {
namespace TT {

// Scratch of one relaxation: the rounding scratch for inputs of rank rcap + 1, followed
// by the rank-1 equilibrium.
struct RelaxScratch {
  RoundScratch round;
  TTLayout eq;

  KOKKOS_INLINE_FUNCTION int Size() const { return round.Size() + eq.Size(); }
};

inline RelaxScratch MakeRelaxScratch(const VelocityGrid &grid, const int rcap) {
  return RelaxScratch{MakeRoundScratch(grid, rcap + 1), MakeLayout(grid, 1)};
}

struct TTRelaxResult {
  RelaxResult relax;
  RoundInfo round; // r1 = r2 = 0 if no rounding was done (invalid cell or c = 0)
};

// Relax f (capacity f.L.rcap) in place. work holds sc.Size() reals.
template <class Data>
KOKKOS_INLINE_FUNCTION TTRelaxResult RelaxCellTT(
    const VelocityGrid &grid, const TTRef<Data> &f, const RelaxationStep &step,
    const CollisionModel &model, const Real kb_per_m, const EquilibriumParams &eq_params,
    const RelaxScratch &sc, Real *work, const RoundParams &prm) {
  TTRelaxResult out;
  out.round = RoundInfo{0, 0, false, true, 0.0, 0.0};
  const auto target = IsotropicTarget(ComputeRawMoments(grid, f));
  Maxwellian eq;
  out.relax.eq = SolveEquilibrium(grid, target, eq_params, eq);
  if (out.relax.eq.status == EquilibriumResult::Status::invalid) {
    out.relax.nu = 0.0;
    return out; // empty or non-physical cell: leave f unchanged
  }
  out.relax.nu = model.Frequency(target.n, target.theta[0] / kb_per_m);
  const Real c = step.Fraction(out.relax.nu);
  if (c == 0.0) return out;
  const auto m = MakeOutRef(PtrData{work + sc.round.Size()}, sc.eq);
  FillMaxwellian(grid, eq, m);
  AddInto(sc.round, work, 1.0 - c, f, c, m);
  out.round = Round(sc.round, work, MakeOutRef(f.data, f.L), prm);
  return out;
}

} // namespace TT
} // namespace Kinetics

#endif // KINETICS_TT_RELAX_HPP_
