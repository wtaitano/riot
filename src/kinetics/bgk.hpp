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
#ifndef KINETICS_BGK_HPP_
#define KINETICS_BGK_HPP_
// This file was made in part with generative AI.

// BGK relaxation of one spatial cell.
//
// The BGK operator df/dt = nu (M[f] - f) conserves mass, momentum and energy, so the
// discrete equilibrium M of the cell and the collision frequency nu (a function of those
// moments only) stay fixed during a relaxation, and every relaxation step has the form
//
//   f <- f + c (M - f),
//
// with c depending only on nu h. The exact solution over h has c = 1 - exp(-nu h). An
// implicit (DIRK) stage f = f* + a h nu (M[f] - f) is solved in closed form, because
// M[f] = M[f*], and gives c = a nu h / (1 + a nu h); the collapsed explicit-implicit
// stage of the SL-DIRK2 integrator (kinetics_tasks.cpp) gives c = a nu h / (1 + b nu h).
// c > 1 (over-relaxation) is allowed and may make f slightly negative.
//
// M is the Mieussens discrete equilibrium, whose discrete moments equal those of f, so
// the update conserves the discrete invariants to roundoff whatever c is.

#include <cmath>

#include <parthenon/package.hpp>

#include "kinetics/equilibrium.hpp"
#include "kinetics/moments.hpp"
#include "kinetics/velocity_grid.hpp"

using namespace parthenon::package::prelude;

namespace Kinetics {

// Collision frequency nu(n, T). constant: nu = nu0. power_law: nu = p / mu(T) with
// p = n k_B T and mu = mu_ref (T / T_ref)^omega (hard spheres omega = 1/2, Maxwell
// molecules omega = 1).
struct CollisionModel {
  enum class Type { constant, power_law };
  Type type;
  Real nu0;
  Real mu_ref;
  Real T_ref;
  Real omega;
  Real kb; // Boltzmann constant [erg / K], for p = n k_B T

  KOKKOS_INLINE_FUNCTION Real Frequency(const Real n, const Real T) const {
    if (type == Type::constant) return nu0;
    const Real mu = mu_ref * std::pow(T / T_ref, omega);
    return n * kb * T / mu;
  }
};

// How far one relaxation step moves f toward M: exact over h, c = 1 - exp(-nu h), or
// rational, c = a nu h / (1 + b nu h).
struct RelaxationStep {
  enum class Type { exact, rational };
  Type type;
  Real h;
  Real a;
  Real b;

  KOKKOS_INLINE_FUNCTION Real Fraction(const Real nu) const {
    const Real z = nu * h;
    if (type == Type::exact) return -std::expm1(-z);
    return a * z / (1.0 + b * z);
  }
};

inline RelaxationStep ExactRelaxation(const Real h) {
  return RelaxationStep{RelaxationStep::Type::exact, h, 0.0, 0.0};
}
inline RelaxationStep RationalRelaxation(const Real h, const Real a, const Real b) {
  return RelaxationStep{RelaxationStep::Type::rational, h, a, b};
}

// Result of relaxing one cell.
struct RelaxResult {
  EquilibriumResult eq;
  Real nu;
};

// Relax the distribution of one cell by one step. `f` is a read/write accessor over the
// flat velocity index. kb_per_m = k_B / m converts theta to T.
template <class F>
KOKKOS_INLINE_FUNCTION RelaxResult RelaxCell(const VelocityGrid &grid, const F &f,
                                             const RelaxationStep &step,
                                             const CollisionModel &model,
                                             const Real kb_per_m,
                                             const EquilibriumParams &eq_params) {
  RelaxResult out;
  const auto target = IsotropicTarget(ComputeRawMoments(grid, f));
  Maxwellian eq;
  out.eq = SolveEquilibrium(grid, target, eq_params, eq);
  if (out.eq.status == EquilibriumResult::Status::invalid) {
    out.nu = 0.0;
    return out; // empty or non-physical cell: leave f unchanged
  }
  out.nu = model.Frequency(target.n, target.theta[0] / kb_per_m);
  const Real c = step.Fraction(out.nu);
  for (int iz = 0; iz < grid.nv[2]; ++iz)
    for (int iy = 0; iy < grid.nv[1]; ++iy)
      for (int ix = 0; ix < grid.nv[0]; ++ix) {
        const int n = grid.Flat(ix, iy, iz);
        const Real m = eq(grid, ix, iy, iz);
        f(n) += c * (m - f(n));
      }
  return out;
}

} // namespace Kinetics

#endif // KINETICS_BGK_HPP_
