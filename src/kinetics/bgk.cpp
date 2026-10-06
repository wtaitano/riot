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

#include <iostream>
#include <string>

#include <parthenon_mpi.hpp>

#include "kinetics/bgk.hpp"
#include "kinetics/kinetics.hpp"
#include "kinetics/tt_relax.hpp"
#include "kinetics/tt_tensor.hpp"
#include "riot_utils/riot_loops.hpp"

namespace Kinetics {

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Kinetics::Relax
//! \brief One BGK relaxation step (exact or DIRK stage) of every interior cell. Cells
//! whose equilibrium solve did not converge relax toward the sampled Maxwellian and are
//! flagged in kinetics.eq_fallback (the flag is only set here, never cleared).
namespace {

// representation = tt: one team per cell (team scratch level 1 holds the rounding work
// arrays, see tt_relax.hpp); the cell is relaxed by one thread of the team.
void RelaxTT(MeshData<Real> *md, const RelaxationStep step, const bool last) {
  auto pm = md->GetParentPointer();
  static auto desc =
      MakePackDescriptor<fields::f_tt, fields::eq_fallback, fields::tt_round>(
          pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  if (v.GetNBlocks() == 0) return;

  auto pkg = pm->packages.Get(pkg_name);
  const auto grid = pkg->Param<VelocityGrid>("grid");
  const auto model = pkg->Param<CollisionModel>("collision_model");
  const auto eq_params = pkg->Param<EquilibriumParams>("eq_params");
  const Real kb_per_m = pkg->Param<Species>("species").kb_per_m;
  const auto L = pkg->Param<TT::TTLayout>("tt_layout");
  auto prm = pkg->Param<TT::RoundParams>("tt_round");
  if (last && pkg->Param<bool>("lomac")) prm.rank_max = L.rcap - 2;
  const auto sc = TT::MakeRelaxScratch(grid, L.rcap);
  const int nwork = sc.Size();
  const std::size_t scratch_bytes = parthenon::ScratchPad1D<Real>::shmem_size(nwork);
  TT::RequireTeamScratch(scratch_bytes, "relaxation");
  constexpr int scratch_level = 1;

  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Kinetics::RelaxTT", DevExecSpace(), scratch_bytes,
      scratch_level, 0, v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j,
                    const int i) {
        parthenon::ScratchPad1D<Real> work(member.team_scratch(scratch_level), nwork);
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          const auto f =
              TT::MakeRef(TT::PackCell<decltype(v), fields::f_tt>{v, b, k, j, i}, L);
          const auto res = TT::RelaxCellTT(grid, f, step, model, kb_per_m, eq_params, sc,
                                           work.data(), prm);
          if (res.relax.eq.status == EquilibriumResult::Status::fallback)
            v(b, fields::eq_fallback(), k, j, i) = 1.0;
          const Real rel =
              (res.round.norm > 0.0) ? res.round.discarded / res.round.norm : 0.0;
          v(b, fields::tt_round(0), k, j, i) += rel;
          v(b, fields::tt_round(1), k, j, i) += res.round.cap_hit ? 1.0 : 0.0;
          v(b, fields::tt_round(2), k, j, i) += res.round.svd_ok ? 0.0 : 1.0;
        });
      });
}

} // namespace

TaskStatus Relax(MeshData<Real> *md, const RelaxationStep step, const bool last) {
  auto pm = md->GetParentPointer();
  if (GetRepresentation(pm->packages.Get(pkg_name).get()) == Representation::tt) {
    RelaxTT(md, step, last);
    return TaskStatus::complete;
  }
  static auto desc =
      MakePackDescriptor<fields::f, fields::eq_fallback>(pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  if (v.GetNBlocks() == 0) return TaskStatus::complete;

  auto pkg = pm->packages.Get(pkg_name);
  const auto grid = pkg->Param<VelocityGrid>("grid");
  const auto model = pkg->Param<CollisionModel>("collision_model");
  const auto eq_params = pkg->Param<EquilibriumParams>("eq_params");
  const Real kb_per_m = pkg->Param<Species>("species").kb_per_m;

  auto space = RiotFlatLoop::GetIndexSpace(IndexDomain::interior, v.GetNBlocks(), md);
  RiotFlatLoop::four_d(
      "Kinetics::Relax", space,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto f = [&](const int n) -> Real & { return v(b, fields::f(n), k, j, i); };
        const auto res = RelaxCell(grid, f, step, model, kb_per_m, eq_params);
        // Empty or non-physical cells (Status::invalid) are left unchanged and are not
        // fallbacks.
        // The flag accumulates over the relaxations of one hydro step; KineticsTasks
        // resets it at the start of the step.
        if (res.eq.status == EquilibriumResult::Status::fallback)
          v(b, fields::eq_fallback(), k, j, i) = 1.0;
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Kinetics::CheckEquilibriumFallbacks
//! \brief Global fraction of interior cells flagged in kinetics.eq_fallback; abort above
//! kinetics/eq_fallback_abort.
TaskStatus CheckEquilibriumFallbacks(Mesh *pm) {
  auto md = pm->mesh_data.Get().get();
  static auto desc = MakePackDescriptor<fields::eq_fallback>(pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  Real counts[2] = {0.0, 0.0};
  if (v.GetNBlocks() > 0) {
    using rt = RiotFlatReduce::ReductionType<Kokkos::Sum<Real>>;
    auto space = rt::GetIndexSpace(IndexDomain::interior, v.GetNBlocks(), md);
    counts[0] = rt::four_d(
        "Kinetics::CountFallbacks", space,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
          lsum += v(b, fields::eq_fallback(), k, j, i);
        });
    counts[1] = static_cast<Real>(space.nblocks) * (space.kb.e - space.kb.s + 1) *
                (space.jb.e - space.jb.s + 1) * (space.ib.e - space.ib.s + 1);
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_PARTHENON_REAL, MPI_SUM,
                                    MPI_COMM_WORLD));
#endif
  const Real frac = (counts[1] > 0.0) ? counts[0] / counts[1] : 0.0;
  const Real abort = pm->packages.Get(pkg_name)->Param<Real>("eq_fallback_abort");
  if (frac > abort) {
    if (Globals::my_rank == 0) {
      std::cout << "kinetics: " << counts[0] << " of " << counts[1]
                << " cells fell back to the sampled Maxwellian" << std::endl;
    }
    PARTHENON_FAIL("kinetics: equilibrium fallback fraction above "
                   "kinetics/eq_fallback_abort; widen or refine the velocity box");
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Kinetics::ResetFallbackFlags
//! \brief Clear kinetics.eq_fallback (and kinetics.tt_round) before the relaxations of a
//! hydro step.
TaskStatus ResetFallbackFlags(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  static auto desc =
      MakePackDescriptor<fields::eq_fallback, fields::tt_round, fields::tt_amr_round>(
          pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  if (v.GetNBlocks() == 0) return TaskStatus::complete;
  const bool tt =
      GetRepresentation(pm->packages.Get(pkg_name).get()) == Representation::tt;
  const bool amr = tt && pm->multilevel;
  auto space = RiotFlatLoop::GetIndexSpace(IndexDomain::entire, v.GetNBlocks(), md);
  RiotFlatLoop::four_d(
      "Kinetics::ResetFallbackFlags", space,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        v(b, fields::eq_fallback(), k, j, i) = 0.0;
        if (tt) {
          for (int a = 0; a < 3; ++a)
            v(b, fields::tt_round(a), k, j, i) = 0.0;
        }
        if (amr) {
          for (int a = 0; a < 3; ++a)
            v(b, fields::tt_amr_round(a), k, j, i) = 0.0;
        }
      });
  return TaskStatus::complete;
}

} // namespace Kinetics
