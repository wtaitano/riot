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
#ifndef KINETICS_KINETICS_HPP_
#define KINETICS_KINETICS_HPP_
// This file was made in part with generative AI.

// Kinetics package: a neutral monatomic gas described by its velocity distribution
// function f(x, v, t) on a fixed uniform velocity grid, with a BGK collision operator.
// The dense discrete-velocity solver here is the reference for later compressed (TT)
// representations. Design: claude_sessions/kinetic_bgk/S0_DESIGN.md (dense) and
// S1_DESIGN.md (tensor train, kinetics/representation = tt).

#include <memory>
#include <string>
#include <vector>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "kinetics/bgk.hpp"
#include "kinetics/equilibrium.hpp"
#include "kinetics/velocity_grid.hpp"
#include "variables.hpp"

using namespace parthenon;
using namespace parthenon::package::prelude;

namespace Kinetics {

const std::string pkg_name = "kinetics";
const std::string input_block = "kinetics";

// Fields. As with the radiation intensity, the ncomp template argument of the flat
// velocity-space field is a placeholder; the true component count is the number of
// velocity nodes, nv1 * nv2 * nv3.
namespace fields {
VARIABLE_VECTOR(kinetics, f, false, 1); // number density in phase space, flat over v
// Tensor-train f (representation = tt): ranks and cores per cell, see tt_tensor.hpp.
VARIABLE_VECTOR(kinetics, f_tt, false, 1);
VARIABLE_SCALAR(kinetics, rho, false);
VARIABLE_VECTOR(kinetics, velocity, false, 3);
VARIABLE_SCALAR(kinetics, temperature, false);
VARIABLE_SCALAR(kinetics, pressure, false);
VARIABLE_VECTOR(kinetics, stress, false, 6); // xx, yy, zz, xy, xz, yz
VARIABLE_VECTOR(kinetics, heat_flux, false, 3);
VARIABLE_SCALAR(kinetics, eq_fallback, false); // 0 if the equilibrium solve converged
VARIABLE_VECTOR(kinetics, tt_rank, false, 2);  // representation = tt: (r1, r2)
// representation = tt, per cell over the last hydro step: sum of relative discarded
// norms of the roundings, rank-cap hits, non-converged SVDs.
VARIABLE_VECTOR(kinetics, tt_round, false, 3);
// representation = tt on a multilevel mesh, per cell over the last hydro step: the same
// statistics for the roundings of the restriction to the coarse buffer (kinetics_amr).
VARIABLE_VECTOR(kinetics, tt_amr_round, false, 3);
// Adaptive refinement: ||f - M[f]|| / ||f|| per cell at the last tagging (kinetics_amr).
VARIABLE_SCALAR(kinetics, noneq, false);
// representation = tt, tt_diag = cross, per cell at the last history output: cross
// entropy integrand sum, phi evaluations, max(q1, q2), rank-cap hit, not converged.
VARIABLE_VECTOR(kinetics, tt_cross, false, 5);
// closure_coupling: non-equilibrium stress Pi (xx, yy, zz, xy, xz, yz; traceless) and
// heat flux q (x, y, z) from f at the start (old) and end (new) of the kinetics step.
VARIABLE_VECTOR(kinetics, closure_old, false, 9);
VARIABLE_VECTOR(kinetics, closure_new, false, 9);
// closure_coupling with kinetic walls: time-integrated (m, m v, m |v|^2 / 2) flux of f
// through each wall face over the kinetics step, stored in the ghost cell next to the
// face (kinetics_walls.cpp).
VARIABLE_VECTOR(kinetics, wall_flux, false, 5);
// kinetics/lomac, per cell at the last correction: [status (0 applied, 1 skipped for
// the target or the 5 x 5 solve, 2 skipped for TT rank capacity), 1 if the correction
// made f negative somewhere].
VARIABLE_VECTOR(kinetics, lomac_stat, false, 2);
} // namespace fields

// Particle physics constants of the single species, derived from the hydro material.
struct Species {
  Real mass;     // particle mass [g]
  Real kb_per_m; // k_B / m [erg / (g K)], so theta = (k_B / m) T
};

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

// kinetics/closure_coupling: hydro fluxes take the kinetic stress and heat flux. Read
// by hydro (face velocities), the driver (step order) and the kinetics package.
inline bool ClosureCoupling(ParameterInput *pin) {
  return pin->GetOrAddBoolean("physics", "kinetics", false) &&
         pin->GetOrAddBoolean("kinetics", "closure_coupling", true,
                              "Add the kinetic stress and heat flux to the hydro fluxes");
}

// Operator-split entry point. Runs after the hydro step, or before it with
// closure_coupling (the hydro stages then interpolate the closure in time).
TaskCollection KineticsTasks(Mesh *pm, parthenon::SimTime &tm, const Real dt);

// closure_coupling (kinetics_closure.cpp). ComputeClosure: Pi and q of f on the entire
// block (the ghosts of f must be current) into closure_new. ShiftClosure: closure_old <-
// closure_new. AddClosureFluxes: adds (1 - w) old + w new to the hydro momentum and
// energy fluxes of md (w = c_s of the RK stage).
TaskStatus ComputeClosure(MeshData<Real> *md);
TaskStatus ShiftClosure(MeshData<Real> *md);
TaskStatus AddClosureFluxes(MeshData<Real> *md, const Real w);

// Hydro wall fluxes enslaved to the kinetic walls (kinetics_walls.cpp). Param
// "coupled_walls" flags the faces (ix1..ox3) that are kinetic walls with
// closure_coupling. ResetWallFlux clears kinetics.wall_flux; AccumulateWallFlux adds h
// times the flux of f in md through the wall faces normal to d (interp: dense f, sweep
// the upwind cell along the directions before d first); ApplyWallFluxes sets the hydro
// fluxes at the wall faces of md to wall_flux / dt.
bool HasCoupledWalls(const StateDescriptor *pkg);
TaskStatus ResetWallFlux(MeshData<Real> *md);
void AccumulateWallFlux(MeshData<Real> *md, const int d, const Real h, const bool interp);
TaskStatus ApplyWallFluxes(MeshData<Real> *md, const Real dt);

// kinetics/lomac (kinetics_lomac.cpp): f <- f + M P after the hydro step, so the kinetic
// moments equal the hydro ones (S3_DESIGN.md). Requires closure_coupling. LomacTasks is
// a RiotDriver::PostStepTasks entry; HistoryLomacSums gives [skipped cells, cells made
// negative, TT cells skipped for rank capacity, cells].
inline bool Lomac(ParameterInput *pin) {
  return pin->GetOrAddBoolean("physics", "kinetics", false) &&
         pin->GetOrAddBoolean("kinetics", "lomac", false,
                              "Enslave the kinetic moments to hydro after each step");
}
TaskCollection LomacTasks(Mesh *pm, parthenon::SimTime &tm, const Real dt);
std::vector<Real> HistoryLomacSums(MeshData<Real> *md);

// Time integrator of streaming + collisions (kinetics_tasks.cpp).
enum class Integrator { sl_dirk2, strang };

// One BGK relaxation step of all interior cells (bgk.cpp).
// last: the last rounding of the kinetics step; with kinetics/lomac its rank cap is
// tt_rank_max - 2, which leaves room for the rank-2 correction (S3-Q5).
TaskStatus Relax(MeshData<Real> *md, const RelaxationStep step, const bool last);
// Clear kinetics.eq_fallback (and kinetics.tt_round) (bgk.cpp); Relax only adds to them.
TaskStatus ResetFallbackFlags(MeshData<Real> *md);
// Abort if too many cells fell back to the sampled Maxwellian in the last relaxation.
TaskStatus CheckEquilibriumFallbacks(Mesh *pm);

// One semi-Lagrangian streaming step of length h, dst <- SL(src) (semi_lagrangian.cpp).
// tmp: scratch register of the multi-D tensor-train sweeps (unused otherwise).
// last: as for Relax (applies to the last directional sweep).
TaskStatus Stream(MeshData<Real> *src, MeshData<Real> *tmp, MeshData<Real> *dst,
                  const Real h, const bool last);
// Largest substep allowed by kinetics/cfl over the whole mesh (MPI-reduced).
Real MaxStreamingStep(Mesh *pm);

// Mesh refinement (kinetics_amr.cpp). RestrictTT: coarse buffer of a tensor-train f <-
// rounded average of the children, on blocks with a coarser neighbor (no-op for dense f
// or a single-level mesh). AddFExchangeTasks: RestrictTT when needed, then the ghost
// exchange of the f register md. ExchangeFGhosts: the same, executed now on base.
TaskStatus RestrictTT(MeshData<Real> *md);
// amr_prolong = linear, tt: coarse ghosts next to non-coarser neighbors <- restriction of
// the received fine ghosts (RestrictGhostsTT); fine ghosts next to coarser neighbors <-
// linear reconstruction from the coarse buffer (ProlongateTT). Inside the exchange.
TaskStatus RestrictGhostsTT(MeshData<Real> *md);
TaskStatus ProlongateTT(MeshData<Real> *md);
TaskID AddFExchangeTasks(TaskID dependency, TaskList &tl,
                         std::shared_ptr<MeshData<Real>> &md);
void ExchangeFGhosts(Mesh *pm, const bool dense_too = false);
// Adaptive refinement (kinetics_amr.cpp). ComputeNonEquilibrium: kinetics.noneq of the
// interior. CheckRefinement: the kinetic criterion (CheckRefinementMesh).
// RestrictForRemesh: TT coarse buffers of the blocks that may derefine, after tagging
// and before Parthenon's remesh (driver). AfterRemesh: if the mesh changed since the
// last call, new fine blocks (linear TT), LoMaC, ghosts and closure (driver, before the
// global time step).
TaskStatus ComputeNonEquilibrium(MeshData<Real> *md);
void CheckRefinement(MeshData<Real> *md, parthenon::ParArray1D<AmrTag> &amr_tags);
void RestrictForRemesh(Mesh *pm);
void AfterRemesh(Mesh *pm, parthenon::SimTime &tm);

// Velocity-space representation of f.
enum class Representation { dense, tt };
inline Representation GetRepresentation(const StateDescriptor *pkg) {
  return pkg->Param<std::string>("representation") == "tt" ? Representation::tt
                                                           : Representation::dense;
}

// Abort if a restart file's storage layout of f differs from this run's (kinetics_init).
void CheckRestartLayout(Mesh *pm, ParameterInput *pin, parthenon::SimTime &tm);
// UserWorkBeforeLoop: CheckRestartLayout, then (closure_coupling) the closure of the
// initial or restarted f.
void BeforeLoop(Mesh *pm, ParameterInput *pin, parthenon::SimTime &tm);

// Fill f from the hydro state (equilibrium, bi-Maxwellian or two drifting Maxwellians).
void PostInitialization(Mesh *pm, ParameterInput *pin, MeshData<Real> *md);

// Resolution diagnostics of the current f: the largest fraction of a cell's mass on
// the outermost node layer of the velocity box, and the smallest v_th / dv. Global
// (MPI-reduced). Warns or aborts according to the kinetics/edge_mass_* and
// min_vth_over_dv parameters.
struct ResolutionReport {
  Real max_edge_mass_fraction;
  Real min_vth_over_dv;
};
ResolutionReport CheckResolution(Mesh *pm, MeshData<Real> *md, const std::string &when);

// Derived moment fields for output.
void SetDerivedMomentsMesh(Mesh *pm, ParameterInput *pin, parthenon::SimTime &tm);

// History: summed invariants of the kinetic and hydro states (see kinetics_output.cpp
// for the column order) and min f.
std::vector<Real> HistorySums(MeshData<Real> *md);
Real HistoryMinF(MeshData<Real> *md);
Real HistoryFallbackCount(MeshData<Real> *md);
Real HistorySubsteps(MeshData<Real> *md);
// representation = tt: [cells, sum r1, sum r2, cells with a rank at tt_rank_max] (the
// mean ranks follow by division), and the largest rank. Per-cell ranks for percentiles
// are in the derived field kinetics.tt_rank.
std::vector<Real> HistoryRankSums(MeshData<Real> *md);
Real HistoryMaxRank(MeshData<Real> *md);
// representation = tt: kinetics.tt_round summed over the interior cells.
std::vector<Real> HistoryRoundSums(MeshData<Real> *md);
// representation = tt, multilevel mesh: kinetics.tt_amr_round summed over interior cells.
std::vector<Real> HistoryAmrRoundSums(MeshData<Real> *md);
// Adaptive refinement: largest kinetics.noneq (recomputed), and the number of blocks per
// level (relative to the root, 0 .. numlevel - 1).
Real HistoryNoneqMax(MeshData<Real> *md);
std::vector<Real> HistoryBlocksPerLevel(MeshData<Real> *md);
// tt_diag = cross: kinetics.tt_cross columns 1-4 summed over the interior cells (filled
// by HistorySums, which is enrolled before it).
std::vector<Real> HistoryCrossSums(MeshData<Real> *md);

} // namespace Kinetics

#endif // KINETICS_KINETICS_HPP_
