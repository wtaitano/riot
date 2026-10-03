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

// Operator-split task collection of the kinetics package, run after the hydro step.
//
// The hydro step dt is split into n substeps h = dt / n, with n the smallest count for
// which the fastest velocity node moves at most cfl cells in any one semi-Lagrangian
// (SL) step (cfl <= 1). Each substep is advanced by one of two integrators
// (kinetics/integrator):
//
//   sl_dirk2 (default): characteristic IMEX Runge-Kutta, explicit SL streaming with a
//     two-stage, stiffly accurate, L-stable DIRK for BGK (g = 1 - 1/sqrt 2):
//
//       f <- SL(g h) f,        f <- f + c1 (M[f] - f),  c1 = (1-g) nu h / (1 + g nu h)
//       f <- SL((1-g) h) f,    f <- f + c2 (M[f] - f),  c2 = g nu h / (1 + g nu h)
//
//     The implicit stages have closed forms because relaxation conserves the moments
//     that fix M and nu, and the explicit stage combination is folded into the first
//     relaxation (c1), so no stage values are stored. Second order in time uniformly in
//     nu h, including nu h >> 1. Its largest SL step is (1-g) h, so h may be 1/(1-g) =
//     1.41 times the single-step cap. c1 > 1 for nu h > 1/(1-2g) = 2.41: the first
//     stage over-relaxes and f may become slightly negative.
//
//   strang: SL(h/2) BGK(h) SL(h/2) with exact relaxation, f <- M + (f - M) exp(-nu h).
//     Second order for nu h <~ 1, first order for nu h >> 1. With merge_half_steps the
//     adjacent half steps of consecutive substeps are merged, giving
//
//       SL(h/2) BGK(h) SL(h) BGK(h) ... SL(h) BGK(h) SL(h/2)      (n + 1 SL steps).
//
// Without collisions (constant nu0 = 0) both integrators reduce to free streaming and
// the Strang sequence is used.
//
// Semi-Lagrangian steps are not in place: they alternate between two registers, the
// shallow k0 (same memory as base) and the separately allocated k1. Every SL step is
// its own task region, so the ghost exchange of one step completes everywhere before
// the next step starts. If the last step lands in k1 the result is copied back to k0.

#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "kinetics/bgk.hpp"
#include "kinetics/kinetics.hpp"
#include "kinetics/velocity_grid.hpp"
#include "riot_utils/riot_loops.hpp"

namespace Kinetics {

namespace {

constexpr char k0_name[] = "kinetics_k0";
constexpr char k1_name[] = "kinetics_k1";
constexpr char k2_name[] = "kinetics_k2"; // multi-D tensor-train sweeps only

// f <- f of another register (interior; ghosts are refilled by the next exchange). For
// representation = tt the whole kinetics.f_tt vector (ranks and core slots) is copied.
template <class Var>
TaskStatus CopyVar(MeshData<Real> *to, MeshData<Real> *from) {
  auto pm = to->GetParentPointer();
  static auto desc = MakePackDescriptor<Var>(pm->resolved_packages.get());
  auto vt = desc.GetPack(to);
  auto vf = desc.GetPack(from);
  if (vt.GetNBlocks() == 0) return TaskStatus::complete;
  const int ncomp = vt.GetSizeHost(0, Var());
  auto space =
      RiotFlatLoop::GetIndexSpace(IndexDomain::interior, vt.GetNBlocks(), ncomp, to);
  RiotFlatLoop::five_d(
      "Kinetics::CopyF", space,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        vt(b, Var(n), k, j, i) = vf(b, Var(n), k, j, i);
      });
  return TaskStatus::complete;
}

TaskStatus CopyF(MeshData<Real> *to, MeshData<Real> *from) {
  auto pm = to->GetParentPointer();
  if (GetRepresentation(pm->packages.Get(pkg_name).get()) == Representation::tt)
    return CopyVar<fields::f_tt>(to, from);
  return CopyVar<fields::f>(to, from);
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection Kinetics::KineticsTasks
//! \brief
TaskCollection KineticsTasks(Mesh *pm, parthenon::SimTime &tm, const Real dt) {
  using parthenon::BoundaryType;
  TaskCollection tc;
  TaskID none(0);
  const int num_partitions = pm->DefaultNumPartitions();
  auto pkg = pm->packages.Get(pkg_name);

  // Registers: k0 shares memory with base; k1 is a separate copy of f.
  const bool tt = GetRepresentation(pkg.get()) == Representation::tt;

  std::vector<std::string> names = {tt ? fields::f_tt::name() : fields::f::name(),
                                    fields::eq_fallback::name()};
  if (tt) names.push_back(fields::tt_round::name());
  auto &base = pm->mesh_data.Get();
  pm->mesh_data.AddShallow(k0_name, base, names);
  pm->mesh_data.Add(k1_name, pm->mesh_data.Get(k0_name));
  const bool sweeps = tt && pm->ndim > 1;
  if (sweeps) pm->mesh_data.Add(k2_name, pm->mesh_data.Get(k0_name));

  const auto model = pkg->Param<CollisionModel>("collision_model");
  const bool collide =
      !(model.type == CollisionModel::Type::constant && model.nu0 == 0.0);
  const bool dirk =
      collide && (pkg->Param<Integrator>("integrator") == Integrator::sl_dirk2);
  constexpr Real g = 1.0 - 0.70710678118654752440; // 1 - 1/sqrt(2)

  // Substeps. The largest SL step is h (Strang, merged) or (1 - g) h (SL-DIRK2).
  const Real hmax = MaxStreamingStep(pm) / (dirk ? (1.0 - g) : 1.0); // MPI-reduced
  const int nsub = std::max(1, static_cast<int>(std::ceil(dt / hmax)));
  const Real h = dt / nsub;
  pkg->UpdateParam("substeps", nsub);
  const bool merge = pkg->Param<bool>("merge_half_steps");

  // Sequence of (SL length, relaxation afterwards) steps.
  std::vector<std::pair<Real, std::optional<RelaxationStep>>> steps;
  const std::optional<RelaxationStep> no_relax;
  if (dirk) {
    for (int s = 0; s < nsub; ++s) {
      steps.emplace_back(g * h, RationalRelaxation(h, 1.0 - g, g));
      steps.emplace_back((1.0 - g) * h, RationalRelaxation(h, g, g));
    }
  } else {
    const auto exact = collide ? std::optional(ExactRelaxation(h)) : no_relax;
    for (int s = 0; s < nsub; ++s) {
      if (merge) {
        steps.emplace_back((s == 0) ? 0.5 * h : h, exact);
      } else {
        steps.emplace_back(0.5 * h, exact);
        steps.emplace_back(0.5 * h, no_relax);
      }
    }
    if (merge) steps.emplace_back(0.5 * h, no_relax);
  }

  // kinetics.eq_fallback (and tt_round) are OneCopy, so k0 and k1 share them. TT
  // streaming also adds to tt_round, so the reset runs for collisionless TT runs too.
  if (collide || tt) {
    TaskRegion &reset_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
      auto &k0 = pm->mesh_data.GetOrAdd(k0_name, i);
      reset_region[i].AddTask(none, ResetFallbackFlags, k0.get());
    }
  }

  // kinetics/streaming = false: the relaxations of the same sequence, in place on k0
  // (0D tests; the substeps still follow the streaming limit).
  if (!pkg->Param<bool>("streaming")) {
    for (const auto &[hs, relax] : steps) {
      if (!relax) continue;
      TaskRegion &region = tc.AddRegion(num_partitions);
      for (int i = 0; i < num_partitions; ++i) {
        auto &k0 = pm->mesh_data.GetOrAdd(k0_name, i);
        region[i].AddTask(none, Relax, k0.get(), *relax);
      }
    }
    steps.clear();
  }

  int cur = 0; // register holding the current f
  for (const auto &[hs, relax] : steps) {
    TaskRegion &region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
      auto &tl = region[i];
      auto &src = pm->mesh_data.GetOrAdd(cur == 0 ? k0_name : k1_name, i);
      auto &dst = pm->mesh_data.GetOrAdd(cur == 0 ? k1_name : k0_name, i);
      auto recv =
          tl.AddTask(none, parthenon::StartReceiveBoundBufs<BoundaryType::any>, src);
      auto bc = parthenon::AddBoundaryExchangeTasks(recv, tl, src, pm->multilevel);
      MeshData<Real> *tmp = sweeps ? pm->mesh_data.GetOrAdd(k2_name, i).get() : nullptr;
      auto stream = tl.AddTask(bc, Stream, src.get(), tmp, dst.get(), hs);
      if (relax) tl.AddTask(stream, Relax, dst.get(), *relax);
    }
    cur = 1 - cur;
  }
  if (cur == 1) {
    TaskRegion &region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
      auto &k0 = pm->mesh_data.GetOrAdd(k0_name, i);
      auto &k1 = pm->mesh_data.GetOrAdd(k1_name, i);
      region[i].AddTask(none, CopyF, k0.get(), k1.get());
    }
  }

  // Global checks. They hold MPI collectives; chain them so every rank calls them in
  // the same order whatever the task scheduler does.
  TaskRegion &check_region = tc.AddRegion(1);
  TaskID fallback_check = none;
  if (collide)
    fallback_check = check_region[0].AddTask(none, CheckEquilibriumFallbacks, pm);
  const int check_every = pkg->Param<int>("check_every");
  // ncycle is the cycle being completed, so this runs after cycles check_every, 2*...
  if (check_every > 0 && (tm.ncycle + 1) % check_every == 0) {
    check_region[0].AddTask(fallback_check, [pm, cycle = tm.ncycle + 1]() {
      CheckResolution(pm, pm->mesh_data.Get().get(), "cycle " + std::to_string(cycle));
      return TaskStatus::complete;
    });
  }
  return tc;
}

} // namespace Kinetics
