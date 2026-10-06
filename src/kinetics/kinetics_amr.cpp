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

// Mesh refinement with kinetics (claude_sessions/kinetic_bgk/S4_DESIGN.md).
//
// Dense f uses Parthenon's refinement operators (piecewise-constant prolongation,
// average restriction). A tensor-train f cannot: Parthenon applies the operators slot by
// slot, and the average of TT slots is not the TT of the average. Prolongation by
// injection is a slot-wise copy, so Parthenon's piecewise-constant operator is exact;
// restriction is registered as a no-op, and RestrictTT fills the coarse buffer instead:
//
//   f_coarse = round(... round(round(w f_1 + w f_2) + w f_3) ... + w f_N),  w = 1 / N,
//
// over the N = 2^ndim children, one block sum of ranks <= 2 rcap and one rounding per
// child after the second (S4-Q11, S4-Q19). It runs before every exchange of f
// (AddFExchangeTasks), on every block with a coarser neighbor, over the whole coarse
// interior (a superset of the send regions and of the cells the coarse physical
// boundary conditions read).

#include <memory>

#include "kinetics/kinetics.hpp"
#include "kinetics/tt_stream.hpp"
#include "kinetics/tt_tensor.hpp"
#include "kinetics/velocity_grid.hpp"

namespace Kinetics {

namespace {

constexpr char amr_name[] = "kinetics_amr";

bool RestrictsTT(const Mesh *pm) {
  return pm->multilevel &&
         GetRepresentation(pm->packages.Get(pkg_name).get()) == Representation::tt;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Kinetics::RestrictTT
//! \brief representation = tt on a multilevel mesh: coarse buffer of kinetics.f_tt <-
//! average of the children, as a rounded TT, on the blocks with a coarser neighbor.
//! Rounding statistics go to kinetics.tt_amr_round of the first child.
TaskStatus RestrictTT(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  if (!RestrictsTT(pm)) return TaskStatus::complete;
  auto pkg = pm->packages.Get(pkg_name);
  const auto grid = pkg->Param<VelocityGrid>("grid");
  const auto L = pkg->Param<TT::TTLayout>("tt_layout");
  const auto prm = pkg->Param<TT::RoundParams>("tt_round");
  const int ndim = pm->ndim;
  const int nchild = 1 << ndim;
  const Real w = 1.0 / nchild;
  const auto sc = TT::MakeRoundScratch(grid, 2 * L.rcap);
  const int nwork = sc.Size() + L.Size(); // rounding, then the running sum
  const std::size_t scratch_bytes = parthenon::ScratchPad1D<Real>::shmem_size(nwork);
  TT::RequireTeamScratch(scratch_bytes, "restriction");
  constexpr int scratch_level = 1;

  static auto desc_fine =
      MakePackDescriptor<fields::f_tt, fields::tt_amr_round>(pm->resolved_packages.get());
  static auto desc_coarse = MakePackDescriptor<fields::f_tt>(
      pm->resolved_packages.get(), {}, {parthenon::PDOpt::Coarse});
  for (int b = 0; b < md->NumBlocks(); ++b) {
    auto mbd = md->GetBlockData(b);
    auto pmb = mbd->GetBlockPointer();
    if (!pmb->HasCoarserNeighbors()) continue;
    auto vf = desc_fine.GetPack(mbd.get());
    auto vc = desc_coarse.GetPack(mbd.get());
    const auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    const auto cib = pmb->c_cellbounds.GetBoundsI(IndexDomain::interior);
    const auto cjb = pmb->c_cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto ckb = pmb->c_cellbounds.GetBoundsK(IndexDomain::interior);
    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "Kinetics::RestrictTT", DevExecSpace(), scratch_bytes,
        scratch_level, ckb.s, ckb.e, cjb.s, cjb.e, cib.s, cib.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int ck, const int cj,
                      const int ci) {
          parthenon::ScratchPad1D<Real> work(member.team_scratch(scratch_level), nwork);
          Kokkos::single(Kokkos::PerTeam(member), [&]() {
            using PF = TT::PackCell<decltype(vf), fields::f_tt>;
            using PC = TT::PackCell<decltype(vc), fields::f_tt>;
            // First child (lower corner) of the coarse cell.
            const int i0 = (ci - cib.s) * 2 + ib.s;
            const int j0 = (ndim > 1) ? (cj - cjb.s) * 2 + jb.s : jb.s;
            const int k0 = (ndim > 2) ? (ck - ckb.s) * 2 + kb.s : kb.s;
            auto child = [&](const int c) {
              return TT::MakeRef(
                  PF{vf, 0, k0 + ((c >> 2) & 1), j0 + ((c >> 1) & 1), i0 + (c & 1)}, L);
            };
            const auto out = TT::MakeOutRef(PC{vc, 0, ck, cj, ci}, L);
            const auto acc = TT::MakeOutRef(TT::PtrData{work.data() + sc.Size()}, L);
            TT::RoundTally tally;
            TT::AddInto(sc, work.data(), w, child(0), w, child(1));
            if (nchild == 2) {
              tally.Add(TT::Round(sc, work.data(), out, prm));
            } else {
              tally.Add(TT::Round(sc, work.data(), acc, prm));
              for (int c = 2; c < nchild; ++c) {
                TT::AddInto(sc, work.data(), 1.0, TT::MakeRef(acc.data, L), w, child(c));
                if (c + 1 < nchild) {
                  tally.Add(TT::Round(sc, work.data(), acc, prm));
                } else {
                  tally.Add(TT::Round(sc, work.data(), out, prm));
                }
              }
            }
            vf(0, fields::tt_amr_round(0), k0, j0, i0) += tally.rel_discarded;
            vf(0, fields::tt_amr_round(1), k0, j0, i0) += tally.cap_hits;
            vf(0, fields::tt_amr_round(2), k0, j0, i0) += tally.svd_failures;
          });
        });
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskID Kinetics::AddFExchangeTasks
//! \brief Ghost exchange of f on md (a register holding f and, for TT, tt_amr_round):
//! RestrictTT on multilevel TT meshes, then Parthenon's boundary exchange.
TaskID AddFExchangeTasks(TaskID dependency, TaskList &tl,
                         std::shared_ptr<MeshData<Real>> &md) {
  using parthenon::BoundaryType;
  auto pm = md->GetParentPointer();
  auto recv =
      tl.AddTask(dependency, parthenon::StartReceiveBoundBufs<BoundaryType::any>, md);
  TaskID ready = recv;
  if (RestrictsTT(pm)) ready = recv | tl.AddTask(dependency, RestrictTT, md.get());
  return parthenon::AddBoundaryExchangeTasks(ready, tl, md, pm->multilevel);
}

//----------------------------------------------------------------------------------------
//! \fn  void Kinetics::ExchangeFGhosts
//! \brief Restrict and exchange f now (multilevel TT only). Parthenon's own exchanges
//! during initialization and restart run with the no-op TT restriction, so the coarse
//! neighbors of fine blocks hold stale ghosts until this runs.
void ExchangeFGhosts(Mesh *pm) {
  if (!RestrictsTT(pm)) return;
  auto &base = pm->mesh_data.Get();
  pm->mesh_data.AddShallow(amr_name, base,
                           {fields::f_tt::name(), fields::tt_amr_round::name()});
  TaskCollection tc;
  TaskID none(0);
  const int num_partitions = pm->DefaultNumPartitions();
  TaskRegion &region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    auto &md = pm->mesh_data.GetOrAdd(amr_name, i);
    AddFExchangeTasks(none, region[i], md);
  }
  PARTHENON_REQUIRE(tc.Execute() == TaskListStatus::complete,
                    "kinetics: ghost exchange of f failed");
}

} // namespace Kinetics
