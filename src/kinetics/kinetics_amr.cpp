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
#include <unordered_set>
#include <vector>

#include "kinetics/equilibrium.hpp"
#include "kinetics/kinetics.hpp"
#include "kinetics/moments.hpp"
#include "kinetics/tt_moments.hpp"
#include "kinetics/tt_stream.hpp"
#include "kinetics/tt_tensor.hpp"
#include "kinetics/velocity_grid.hpp"
#include "riot_utils/riot_loops.hpp"

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

namespace {

// Fine f_tt on the cells of box fb <- unlimited linear reconstruction from the coarse
// buffer, as Parthenon's ProlongateSharedLinear on a uniform mesh: a child on the low
// (high) side of its coarse cell C along d uses the slope toward C - e_d (C + e_d), so
//   f = (1 - ndim / 4) f_C + (1/4) sum_d f_{C +- e_d},
// summed pairwise and rounded after every addition. The coarse cells read must be set.
void ProlongBox(MeshBlockData<Real> *mbd, const Box &fb) {
  auto pmb = mbd->GetBlockPointer();
  auto pm = pmb->pmy_mesh;
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
      DEFAULT_OUTER_LOOP_PATTERN, "Kinetics::ProlongateTT", DevExecSpace(), scratch_bytes,
      scratch_level, fb.k.s, fb.k.e, fb.j.s, fb.j.e, fb.i.s, fb.i.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j, const int i) {
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

} // namespace

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Kinetics::ProlongateTT
//! \brief amr_prolong = linear, tt: fine ghost cells next to coarser neighbors <-
//! unlimited linear reconstruction from the coarse buffer (ProlongBox).
TaskStatus ProlongateTT(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  if (!ProlongsLinearTT(pm)) return TaskStatus::complete;
  const int ndim = pm->ndim;
  const int ng = Globals::nghost;
  for (int b = 0; b < md->NumBlocks(); ++b) {
    auto mbd = md->GetBlockData(b);
    auto pmb = mbd->GetBlockPointer();
    if (!pmb->HasCoarserNeighbors()) continue;
    const IndexRange r[3] = {pmb->cellbounds.GetBoundsI(IndexDomain::interior),
                             pmb->cellbounds.GetBoundsJ(IndexDomain::interior),
                             pmb->cellbounds.GetBoundsK(IndexDomain::interior)};
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
      ProlongBox(mbd.get(), Box{s[2], s[1], s[0]});
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
//! \brief Restrict and exchange f now: multilevel TT (Parthenon's own exchanges during
//! initialization, restart and remesh run with the no-op TT operators, so ghosts next to
//! other levels are stale until this runs), or any multilevel f with dense_too.
void ExchangeFGhosts(Mesh *pm, const bool dense_too) {
  if (!RestrictsTT(pm) && !(dense_too && pm->multilevel)) return;
  auto &base = pm->mesh_data.Get();
  const bool tt =
      GetRepresentation(pm->packages.Get(pkg_name).get()) == Representation::tt;
  if (tt) {
    pm->mesh_data.AddShallow(amr_name, base,
                             {fields::f_tt::name(), fields::tt_amr_round::name()});
  } else {
    pm->mesh_data.AddShallow(amr_name, base, {fields::f::name()});
  }
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

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Kinetics::ComputeNonEquilibrium
//! \brief kinetics.noneq <- ||f - M[f]||_2 / ||f||_2 on the interior, M the discrete
//! equilibrium of the cell (S4-Q14); 0 for an empty cell. TT: exact core contractions,
//! ||f - M||^2 = ||f||^2 - 2 <f, M> + ||M||^2 (M rank 1), so values below ~1e-8 are
//! roundoff.
TaskStatus ComputeNonEquilibrium(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  static auto desc = MakePackDescriptor<fields::f, fields::f_tt, fields::noneq>(
      pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  if (v.GetNBlocks() == 0) return TaskStatus::complete;
  auto pkg = pm->packages.Get(pkg_name);
  const auto grid = pkg->Param<VelocityGrid>("grid");
  const auto eq_params = pkg->Param<EquilibriumParams>("eq_params");
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  if (GetRepresentation(pkg.get()) == Representation::dense) {
    auto space = RiotFlatLoop::GetIndexSpace(IndexDomain::interior, v.GetNBlocks(), md);
    RiotFlatLoop::four_d(
        "Kinetics::NonEquilibrium", space,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          auto f = [&](const int n) -> Real & { return v(b, fields::f(n), k, j, i); };
          Maxwellian eq;
          const auto res = SolveEquilibrium(
              grid, IsotropicTarget(ComputeRawMoments(grid, f)), eq_params, eq);
          Real ff = 0.0, dd = 0.0;
          if (res.status != EquilibriumResult::Status::invalid) {
            for (int iz = 0; iz < grid.nv[2]; ++iz)
              for (int iy = 0; iy < grid.nv[1]; ++iy)
                for (int ix = 0; ix < grid.nv[0]; ++ix) {
                  const Real fv = f(grid.Flat(ix, iy, iz));
                  const Real dv = fv - eq(grid, ix, iy, iz);
                  ff += fv * fv;
                  dd += dv * dv;
                }
          }
          v(b, fields::noneq(), k, j, i) = (ff > 0.0) ? std::sqrt(dd / ff) : 0.0;
        });
    return TaskStatus::complete;
  }
  const auto L = pkg->Param<TT::TTLayout>("tt_layout");
  const auto Lm = TT::MakeLayout(grid, 1);
  const int nwork = Lm.Size() + TT::DotWorkSize(L, L);
  const std::size_t scratch_bytes = parthenon::ScratchPad1D<Real>::shmem_size(nwork);
  TT::RequireTeamScratch(scratch_bytes, "the non-equilibrium criterion");
  constexpr int scratch_level = 1;
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Kinetics::NonEquilibriumTT", DevExecSpace(),
      scratch_bytes, scratch_level, 0, v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j,
                    const int i) {
        parthenon::ScratchPad1D<Real> work(member.team_scratch(scratch_level), nwork);
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          const auto f =
              TT::MakeRef(TT::PackCell<decltype(v), fields::f_tt>{v, b, k, j, i}, L);
          Maxwellian eq;
          const auto res = SolveEquilibrium(
              grid, IsotropicTarget(TT::ComputeRawMoments(grid, f)), eq_params, eq);
          Real ratio = 0.0;
          const Real ff = TT::Dot(f, f, work.data() + Lm.Size());
          if (res.status != EquilibriumResult::Status::invalid && ff > 0.0) {
            const auto m = TT::MakeOutRef(TT::PtrData{work.data()}, Lm);
            TT::FillMaxwellian(grid, eq, m);
            const auto mr = TT::MakeRef(m.data, Lm);
            const Real fm = TT::Dot(f, mr, work.data() + Lm.Size());
            const Real mm = TT::Dot(mr, mr, work.data() + Lm.Size());
            const Real dd = ff - 2.0 * fm + mm;
            ratio = std::sqrt((dd > 0.0 ? dd : 0.0) / ff);
          }
          v(b, fields::noneq(), k, j, i) = ratio;
        });
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Kinetics::CheckRefinement
//! \brief Kinetic refinement criterion (S4-Q14, Q23): per block, the largest
//! kinetics.noneq; refine above kinetics/amr_noneq_refine, derefine below
//! kinetics/amr_noneq_derefine, else same. Combined with the other criteria by
//! Parthenon (max over criteria: refine if any, derefine only if all).
void CheckRefinement(MeshData<Real> *md, parthenon::ParArray1D<AmrTag> &amr_tags) {
  auto pm = md->GetParentPointer();
  ComputeNonEquilibrium(md);
  static auto desc = MakePackDescriptor<fields::noneq>(pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  const int nb = v.GetNBlocks();
  if (nb == 0) return;
  auto pkg = pm->packages.Get(pkg_name);
  const Real refine = pkg->Param<Real>("amr_noneq_refine");
  const Real derefine = pkg->Param<Real>("amr_noneq_derefine");
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Kinetics::CheckRefinement", DevExecSpace(), 0, 0, 0,
      nb - 1, KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
        Real vmax = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(member, (kb.e - kb.s + 1) * (jb.e - jb.s + 1) *
                                                (ib.e - ib.s + 1)),
            [&](const int n, Real &lmax) {
              const int ni = ib.e - ib.s + 1, nj = jb.e - jb.s + 1;
              const int i = ib.s + n % ni;
              const int j = jb.s + (n / ni) % nj;
              const int k = kb.s + n / (ni * nj);
              const Real x = v(b, fields::noneq(), k, j, i);
              lmax = (x > lmax) ? x : lmax;
            },
            Kokkos::Max<Real>(vmax));
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          AmrTag tag = AmrTag::same;
          if (vmax > refine) tag = AmrTag::refine;
          if (vmax < derefine) tag = AmrTag::derefine;
          if (static_cast<int>(tag) > static_cast<int>(amr_tags(b))) amr_tags(b) = tag;
        });
      });
}

//----------------------------------------------------------------------------------------
//! \fn  void Kinetics::RestrictForRemesh
//! \brief Adaptive mesh, after tagging and before the remesh. amr_prolong = linear: f
//! ghosts exchanged (the parent ghosts feed the new fine blocks' slopes). TT: coarse
//! buffer of every block that may derefine (derefinement count at the threshold) <- TT
//! restriction of its interior (S4-Q20). Parthenon's remesh sends the coarse buffers of
//! the blocks that do derefine (a subset) with its no-op TT restriction.
void RestrictForRemesh(Mesh *pm) {
  if (!pm->adaptive) return;
  // Linear prolongation of new fine blocks reads the parent's ghosts: make them current
  // (the last semi-Lagrangian step updates the interior only), for dense and TT alike.
  if (pm->packages.Get(pkg_name)->Param<std::string>("amr_prolong") == "linear")
    ExchangeFGhosts(pm, true);
  if (!RestrictsTT(pm)) return;
  const int threshold = pm->packages.Get(pkg_name)->Param<int>("amr_derefine_count");
  for (auto &pmb : pm->block_list) {
    if (pmb->pmr == nullptr || pmb->pmr->DerefinementCount() < threshold) continue;
    auto &mbd = pmb->meshblock_data.Get();
    const auto &cb = pmb->c_cellbounds;
    RestrictBox(mbd.get(), Box{cb.GetBoundsK(IndexDomain::interior),
                               cb.GetBoundsJ(IndexDomain::interior),
                               cb.GetBoundsI(IndexDomain::interior)});
  }
}

namespace {

// representation = tt: f_tt of every interior cell rounded in place to ranks <= rank_max
// (tt_eps of the run), rounding statistics into kinetics.tt_amr_round.
void RoundToRank(MeshData<Real> *md, const int rank_max) {
  auto pm = md->GetParentPointer();
  static auto desc =
      MakePackDescriptor<fields::f_tt, fields::tt_amr_round>(pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  if (v.GetNBlocks() == 0) return;
  auto pkg = pm->packages.Get(pkg_name);
  const auto grid = pkg->Param<VelocityGrid>("grid");
  const auto L = pkg->Param<TT::TTLayout>("tt_layout");
  auto prm = pkg->Param<TT::RoundParams>("tt_round");
  prm.rank_max = rank_max;
  const auto sc = TT::MakeRoundScratch(grid, L.rcap);
  const int nwork = sc.Size();
  const std::size_t scratch_bytes = parthenon::ScratchPad1D<Real>::shmem_size(nwork);
  TT::RequireTeamScratch(scratch_bytes, "rounding");
  constexpr int scratch_level = 1;
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Kinetics::RoundToRank", DevExecSpace(), scratch_bytes,
      scratch_level, 0, v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j,
                    const int i) {
        parthenon::ScratchPad1D<Real> work(member.team_scratch(scratch_level), nwork);
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          const auto f =
              TT::MakeRef(TT::PackCell<decltype(v), fields::f_tt>{v, b, k, j, i}, L);
          TT::CopyTT(f, TT::InputRef(sc, work.data()));
          TT::RoundTally tally;
          tally.Add(TT::Round(sc, work.data(), TT::MakeOutRef(f.data, L), prm));
          v(b, fields::tt_amr_round(0), k, j, i) += tally.rel_discarded;
          v(b, fields::tt_amr_round(1), k, j, i) += tally.cap_hits;
          v(b, fields::tt_amr_round(2), k, j, i) += tally.svd_failures;
        });
      });
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn  void Kinetics::AfterRemesh
//! \brief Called before every global time step, i.e. after each remesh / load-balancing
//! pass (S4-Q12, Q13, Q22). If the mesh changed (Parthenon's global flag):
//!   * amr_prolong = linear, tt: the interior of each new fine block (its coarse buffer
//!     holds the parent region, copied by Parthenon) <- linear reconstruction;
//!   * lomac: f enslaved to the remeshed hydro moments on every block, ghosts exchanged,
//!     closure recomputed (LomacTasks); a TT f is first rounded to tt_rank_max - 2;
//!   * otherwise TT ghosts exchanged (RestrictTT), and with closure_coupling the closure
//!     of every block recomputed from f (the closure fields are not remeshed).
//! The first call only records the mesh.
void AfterRemesh(Mesh *pm, parthenon::SimTime &tm) {
  if (!pm->multilevel) return;
  auto pkg = pm->packages.Get(pkg_name);
  const auto previous = pkg->Param<std::vector<LogicalLocation>>("amr_locs");
  const bool first = !pkg->Param<bool>("amr_started");
  pkg->UpdateParam("amr_started", true);
  pkg->UpdateParam("amr_locs", pm->GetLocList());
  if (first || !pm->modified) return;

  if (ProlongsLinearTT(pm)) {
    const std::unordered_set<LogicalLocation> old(previous.begin(), previous.end());
    for (auto &pmb : pm->block_list) {
      if (old.count(pmb->loc) || !old.count(pmb->loc.GetParent())) continue;
      auto &mbd = pmb->meshblock_data.Get();
      const auto &cb = pmb->cellbounds;
      ProlongBox(mbd.get(), Box{cb.GetBoundsK(IndexDomain::interior),
                                cb.GetBoundsJ(IndexDomain::interior),
                                cb.GetBoundsI(IndexDomain::interior)});
    }
  }
  // A load-balancing pass moves blocks without changing the mesh: f is unchanged, only
  // the closure (not moved) has to be recomputed.
  const bool remeshed = (previous != pm->GetLocList());
  if (remeshed && pkg->Param<bool>("lomac")) {
    // Leave rank room for the rank-2 correction, as the last rounding of a step (S3-Q5).
    if (RestrictsTT(pm)) {
      const int rmax = pkg->Param<TT::TTLayout>("tt_layout").rcap - 2;
      for (int i = 0; i < pm->DefaultNumPartitions(); ++i)
        RoundToRank(pm->mesh_data.GetOrAdd("base", i).get(), rmax);
    }
    PARTHENON_REQUIRE(LomacTasks(pm, tm, tm.dt).Execute() == TaskListStatus::complete,
                      "kinetics: LoMaC after remesh failed");
    return;
  }
  ExchangeFGhosts(pm);
  if (pkg->Param<bool>("closure_coupling")) {
    for (int i = 0; i < pm->DefaultNumPartitions(); ++i)
      ComputeClosure(pm->mesh_data.GetOrAdd("base", i).get());
  }
}

} // namespace Kinetics
