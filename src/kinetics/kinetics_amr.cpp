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
//
// amr_prolong = linear (S4-Q16, Q21): unlimited linear slopes need coarse values next
// to the coarse cell, so the TT prolongation is also a no-op in Parthenon. Inside the
// exchange, RestrictGhostsTT fills the coarse ghosts next to same-level and finer
// neighbors (Parthenon's SetBounds restriction, for dense f), and ProlongateTT fills the
// fine ghosts next to coarser neighbors from the coarse buffer, each as a rounded TT sum.

#include <array>
#include <memory>
#include <set>
#include <string>

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

namespace {

// Index box of cells (inclusive bounds).
struct Box {
  IndexRange k, j, i;
};

// Rounding statistics into kinetics.tt_amr_round of interior cell (k, j, i) (several
// teams may add to the same cell).
template <class Pack>
KOKKOS_INLINE_FUNCTION void AddTally(const Pack &v, const int k, const int j, const int i,
                                     const TT::RoundTally &t) {
  Kokkos::atomic_add(&v(0, fields::tt_amr_round(0), k, j, i), t.rel_discarded);
  Kokkos::atomic_add(&v(0, fields::tt_amr_round(1), k, j, i),
                     static_cast<Real>(t.cap_hits));
  Kokkos::atomic_add(&v(0, fields::tt_amr_round(2), k, j, i),
                     static_cast<Real>(t.svd_failures));
}

KOKKOS_INLINE_FUNCTION int Clamp(const int a, const IndexRange &r) {
  return (a < r.s) ? r.s : ((a > r.e) ? r.e : a);
}

// Coarse buffer of f_tt on the coarse cells of box cb <- rounded average of their
// children in the fine data (sequential pairwise sums, S4-Q19). The children of cb must
// hold current values.
void RestrictBox(MeshBlockData<Real> *mbd, const Box &cb) {
  auto pmb = mbd->GetBlockPointer();
  auto pm = pmb->pmy_mesh;
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
  auto vf = desc_fine.GetPack(mbd);
  auto vc = desc_coarse.GetPack(mbd);
  const auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto cib = pmb->c_cellbounds.GetBoundsI(IndexDomain::interior);
  const auto cjb = pmb->c_cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto ckb = pmb->c_cellbounds.GetBoundsK(IndexDomain::interior);
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Kinetics::RestrictTT", DevExecSpace(), scratch_bytes,
      scratch_level, cb.k.s, cb.k.e, cb.j.s, cb.j.e, cb.i.s, cb.i.e,
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
          AddTally(vf, Clamp(k0, kb), Clamp(j0, jb), Clamp(i0, ib), tally);
        });
      });
}

// Range of the cells along one direction of the region next to a neighbor at offset o:
// the interior range r for o = 0, else the n cells beyond r on that side.
IndexRange Side(const int o, const IndexRange &r, const int n) {
  if (o < 0) return IndexRange{r.s - n, r.s - 1};
  if (o > 0) return IndexRange{r.e + 1, r.e + n};
  return r;
}

bool ProlongsLinearTT(const Mesh *pm) {
  return RestrictsTT(pm) &&
         pm->packages.Get(pkg_name)->Param<std::string>("amr_prolong") == "linear";
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Kinetics::RestrictTT
//! \brief representation = tt on a multilevel mesh: coarse buffer of kinetics.f_tt <-
//! average of the children, as a rounded TT, over the coarse interior of the blocks with
//! a coarser neighbor.
TaskStatus RestrictTT(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  if (!RestrictsTT(pm)) return TaskStatus::complete;
  for (int b = 0; b < md->NumBlocks(); ++b) {
    auto mbd = md->GetBlockData(b);
    auto pmb = mbd->GetBlockPointer();
    if (!pmb->HasCoarserNeighbors()) continue;
    const auto &cb = pmb->c_cellbounds;
    RestrictBox(mbd.get(), Box{cb.GetBoundsK(IndexDomain::interior),
                               cb.GetBoundsJ(IndexDomain::interior),
                               cb.GetBoundsI(IndexDomain::interior)});
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Kinetics::RestrictGhostsTT
//! \brief amr_prolong = linear, tt: on blocks with a coarser neighbor, the coarse ghost
//! cells next to the other (same-level or finer) neighbors, nghost / 2 layers deep <-
//! rounded average of their fine ghost children (received). These are slope-stencil cells
//! of the prolongation; Parthenon restricts them for dense f in SetBounds. Runs after
//! SetBounds.
TaskStatus RestrictGhostsTT(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  if (!ProlongsLinearTT(pm)) return TaskStatus::complete;
  const int ndim = pm->ndim;
  for (int b = 0; b < md->NumBlocks(); ++b) {
    auto mbd = md->GetBlockData(b);
    auto pmb = mbd->GetBlockPointer();
    if (!pmb->HasCoarserNeighbors()) continue;
    const auto &cb = pmb->c_cellbounds;
    const IndexRange r[3] = {cb.GetBoundsI(IndexDomain::interior),
                             cb.GetBoundsJ(IndexDomain::interior),
                             cb.GetBoundsK(IndexDomain::interior)};
    // nghost / 2 coarse layers, as Parthenon's SetBounds restriction; once per offset
    // (several finer neighbors share one).
    const int depth = Globals::nghost / 2;
    std::set<std::array<int, 3>> done;
    for (const auto &nb : pmb->GetNeighbors()) {
      if (nb.loc.level() < pmb->loc.level()) continue; // coarser: received
      if (nb.offsets.IsCell()) continue;
      const std::array<int, 3> o = nb.offsets;
      if (!done.insert(o).second) continue;
      IndexRange s[3];
      for (int d = 0; d < 3; ++d)
        s[d] = (d < ndim) ? Side(o[d], r[d], depth) : r[d];
      RestrictBox(mbd.get(), Box{s[2], s[1], s[0]});
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Kinetics::ProlongateTT
//! \brief amr_prolong = linear, tt: fine ghost cells next to coarser neighbors <-
//! unlimited linear reconstruction from the coarse buffer, as Parthenon's
//! ProlongateSharedLinear on a uniform mesh: a child on the low (high) side of its coarse
//! cell C along d uses the slope toward C - e_d (C + e_d), so
//!   f = (1 - ndim / 4) f_C + (1/4) sum_d f_{C +- e_d},
//! summed pairwise and rounded after every addition.
TaskStatus ProlongateTT(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  if (!ProlongsLinearTT(pm)) return TaskStatus::complete;
  auto pkg = pm->packages.Get(pkg_name);
  const auto grid = pkg->Param<VelocityGrid>("grid");
  const auto L = pkg->Param<TT::TTLayout>("tt_layout");
  const auto prm = pkg->Param<TT::RoundParams>("tt_round");
  const int ndim = pm->ndim;
  const Real a0 = 1.0 - 0.25 * ndim;
  const auto sc = TT::MakeRoundScratch(grid, 2 * L.rcap);
  const int nwork = sc.Size() + L.Size();
  const std::size_t scratch_bytes = parthenon::ScratchPad1D<Real>::shmem_size(nwork);
  TT::RequireTeamScratch(scratch_bytes, "prolongation");
  constexpr int scratch_level = 1;
  const int ng = Globals::nghost;

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
    const IndexRange r[3] = {ib, jb, kb};
    for (const auto &nb : pmb->GetNeighbors()) {
      if (!(nb.loc.level() < pmb->loc.level())) continue;
      // As Parthenon (CalcIndices): along a direction tangent to the neighbor, the
      // region extends over the ghosts on the side away from the sibling block.
      IndexRange s[3];
      for (int d = 0; d < 3; ++d) {
        s[d] = (d < ndim) ? Side(static_cast<int>(nb.offsets[d]), r[d], ng) : r[d];
        if (d < ndim && static_cast<int>(nb.offsets[d]) == 0) {
          if (pmb->loc.l(d) % 2 == 1) {
            s[d].s -= ng;
          } else {
            s[d].e += ng;
          }
        }
      }
      parthenon::par_for_outer(
          DEFAULT_OUTER_LOOP_PATTERN, "Kinetics::ProlongateTT", DevExecSpace(),
          scratch_bytes, scratch_level, s[2].s, s[2].e, s[1].s, s[1].e, s[0].s, s[0].e,
          KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j,
                        const int i) {
            parthenon::ScratchPad1D<Real> work(member.team_scratch(scratch_level), nwork);
            Kokkos::single(Kokkos::PerTeam(member), [&]() {
              using PF = TT::PackCell<decltype(vf), fields::f_tt>;
              using PC = TT::PackCell<decltype(vc), fields::f_tt>;
              // Coarse cell of (k, j, i) and the side of the child in it (floor division
              // also for the ghosts, i < ib.s).
              const int fi[3] = {i - ib.s, j - jb.s, k - kb.s};
              int c[3] = {cib.s, cjb.s, ckb.s}, step[3] = {0, 0, 0};
              for (int d = 0; d < ndim; ++d) {
                const int q = (fi[d] >= 0) ? fi[d] / 2 : -((1 - fi[d]) / 2);
                c[d] += q;
                step[d] = (fi[d] - 2 * q == 0) ? -1 : 1;
              }
              auto coarse = [&](const int d) {
                int e[3] = {c[0], c[1], c[2]};
                if (d >= 0) e[d] += step[d];
                return TT::MakeRef(PC{vc, 0, e[2], e[1], e[0]}, L);
              };
              const auto out = TT::MakeOutRef(PF{vf, 0, k, j, i}, L);
              const auto acc = TT::MakeOutRef(TT::PtrData{work.data() + sc.Size()}, L);
              TT::RoundTally tally;
              TT::AddInto(sc, work.data(), a0, coarse(-1), 0.25, coarse(0));
              if (ndim == 1) {
                tally.Add(TT::Round(sc, work.data(), out, prm));
              } else {
                tally.Add(TT::Round(sc, work.data(), acc, prm));
                for (int d = 1; d < ndim; ++d) {
                  TT::AddInto(sc, work.data(), 1.0, TT::MakeRef(acc.data, L), 0.25,
                              coarse(d));
                  if (d + 1 < ndim) {
                    tally.Add(TT::Round(sc, work.data(), acc, prm));
                  } else {
                    tally.Add(TT::Round(sc, work.data(), out, prm));
                  }
                }
              }
              AddTally(vf, Clamp(k, kb), Clamp(j, jb), Clamp(i, ib), tally);
            });
          });
    }
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
  if (!ProlongsLinearTT(pm))
    return parthenon::AddBoundaryExchangeTasks(ready, tl, md, pm->multilevel);
  // Linear TT prolongation: Parthenon's TT operators are no-ops; restrict the slope
  // stencil before the coarse boundary conditions, prolongate before the fine ones.
  auto bcs = [](TaskID id, TaskList *tl, std::shared_ptr<MeshData<Real>> md,
                const bool coarse) {
    auto pre = tl->AddTask(id, coarse ? RestrictGhostsTT : ProlongateTT, md.get());
    return tl->AddTask(pre, parthenon::ApplyBoundaryConditionsOnCoarseOrFineMD, md,
                       coarse);
  };
  return parthenon::AddBoundaryExchangeTasks(ready, tl, md, pm->multilevel, bcs);
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
