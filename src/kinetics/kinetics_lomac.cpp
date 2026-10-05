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

// kinetics/lomac (S3_DESIGN.md): after the hydro step, the kinetic moments of every
// interior cell are enslaved to the hydro state, f <- f + M P (lomac.hpp). Runs in
// RiotDriver::PostStepTasks, after OperatorSplitTasks, so f matches the final hydro
// state of the step. Per cell, kinetics.lomac_stat holds [status (0 applied, 1 skipped:
// invalid target or singular 5 x 5, 2 skipped: TT rank capacity), 1 if M P < -f at
// some node]. A global fraction of skipped cells above kinetics/lomac_skip_abort aborts.

#include <iostream>
#include <string>
#include <vector>

#include <parthenon_mpi.hpp>

#include "kinetics/kinetics.hpp"
#include "kinetics/kinetics_cell.hpp"
#include "kinetics/lomac.hpp"
#include "kinetics/moments.hpp"
#include "kinetics/tt_tensor.hpp"
#include "riot_utils/riot_loops.hpp"
#include "variables.hpp"

namespace Kinetics {

namespace {

constexpr char lomac_name[] = "kinetics_lomac";

// Target moments (number units) from the hydro conserved state of the cell.
template <class Pack>
KOKKOS_INLINE_FUNCTION LomacTarget HydroTarget(const Pack &v, const Real inv_m,
                                               const int b, const int k, const int j,
                                               const int i) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  Real rho = 0.0;
  for (int mm = 0; mm < v.GetSize(b, ccmat::rho()); ++mm)
    rho += v(b, ccmat::rho(mm), k, j, i);
  LomacTarget t;
  t.n = rho * inv_m;
  for (int d = 0; d < 3; ++d)
    t.nu[d] = v(b, ccbulk::momentum(d), k, j, i) * inv_m;
  t.nvv = 2.0 * v(b, ccbulk::total_material_energy(), k, j, i) * inv_m;
  return t;
}

KOKKOS_INLINE_FUNCTION Real StatusCode(const LomacResult::Status s) {
  return (s == LomacResult::Status::applied) ? 0.0 : 1.0;
}

TaskStatus LomacCorrect(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  auto pm = md->GetParentPointer();
  static auto desc =
      MakePackDescriptor<fields::f, fields::f_tt, fields::lomac_stat, ccmat::rho,
                         ccbulk::momentum, ccbulk::total_material_energy>(
          pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  if (v.GetNBlocks() == 0) return TaskStatus::complete;
  auto pkg = pm->packages.Get(pkg_name);
  const auto grid = pkg->Param<VelocityGrid>("grid");
  const Real inv_m = 1.0 / pkg->Param<Species>("species").mass;
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  if (GetRepresentation(pkg.get()) == Representation::dense) {
    auto space = RiotFlatLoop::GetIndexSpace(IndexDomain::interior, v.GetNBlocks(), md);
    RiotFlatLoop::four_d(
        "Kinetics::Lomac", space,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          auto f = [&](const int n) -> Real & { return v(b, fields::f(n), k, j, i); };
          const auto r = LomacSolve(grid, ComputeRawMoments(grid, f),
                                    HydroTarget(v, inv_m, b, k, j, i));
          v(b, fields::lomac_stat(0), k, j, i) = StatusCode(r.status);
          v(b, fields::lomac_stat(1), k, j, i) = 0.0;
          if (r.status != LomacResult::Status::applied) return;
          bool neg = false;
          for (int iz = 0; iz < grid.nv[2]; ++iz)
            for (int iy = 0; iy < grid.nv[1]; ++iy)
              for (int ix = 0; ix < grid.nv[0]; ++ix) {
                const int n = grid.Flat(ix, iy, iz);
                const Real mp = LomacCorrection(grid, r, ix, iy, iz);
                neg = neg || (mp < -f(n));
                f(n) += mp;
              }
          v(b, fields::lomac_stat(1), k, j, i) = neg ? 1.0 : 0.0;
        });
    return TaskStatus::complete;
  }

  // tensor train: one team per cell, scratch for the block sum (capacity rcap)
  const auto L = pkg->Param<TT::TTLayout>("tt_layout");
  const int nwork = L.Size();
  const std::size_t scratch_bytes = parthenon::ScratchPad1D<Real>::shmem_size(nwork);
  TT::RequireTeamScratch(scratch_bytes, "the LoMaC correction");
  constexpr int scratch_level = 1;
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Kinetics::LomacTT", DevExecSpace(), scratch_bytes,
      scratch_level, 0, v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j,
                    const int i) {
        parthenon::ScratchPad1D<Real> work(member.team_scratch(scratch_level), nwork);
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          const auto f =
              TT::MakeRef(TT::PackCell<decltype(v), fields::f_tt>{v, b, k, j, i}, L);
          const auto r = LomacSolve(grid, TT::ComputeRawMoments(grid, f),
                                    HydroTarget(v, inv_m, b, k, j, i));
          v(b, fields::lomac_stat(0), k, j, i) = StatusCode(r.status);
          v(b, fields::lomac_stat(1), k, j, i) = 0.0;
          if (r.status != LomacResult::Status::applied) return;
          if (!TT::LomacApplyTT(grid, r, f, work.data())) {
            v(b, fields::lomac_stat(0), k, j, i) = 2.0;
            return;
          }
          // M P < -f(old) <=> f(new) < 0 where the old f was >= 0; count new negatives
          bool neg = false;
          TT::ForEachNode(TT::MakeRef(f.data, L),
                          [&](const int, const int, const int, const Real fv) {
                            neg = neg || (fv < 0.0);
                          });
          v(b, fields::lomac_stat(1), k, j, i) = neg ? 1.0 : 0.0;
        });
      });
  return TaskStatus::complete;
}

// Global fraction of skipped cells; abort above kinetics/lomac_skip_abort.
TaskStatus CheckLomacSkips(Mesh *pm) {
  auto md = pm->mesh_data.Get().get();
  const auto counts = HistoryLomacSums(md);
  Real c[2] = {counts[0], counts[3]};
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(
      MPI_Allreduce(MPI_IN_PLACE, c, 2, MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
#endif
  const Real frac = (c[1] > 0.0) ? c[0] / c[1] : 0.0;
  const Real abort = pm->packages.Get(pkg_name)->Param<Real>("lomac_skip_abort");
  if (frac > abort) {
    if (Globals::my_rank == 0)
      std::cout << "kinetics: LoMaC skipped " << c[0] << " of " << c[1] << " cells"
                << std::endl;
    PARTHENON_FAIL(
        "kinetics: LoMaC skipped-cell fraction above kinetics/lomac_skip_abort");
  }
  return TaskStatus::complete;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn  std::vector<Real> Kinetics::HistoryLomacSums
//! \brief kinetics/lomac, over the interior cells at the last correction: [skipped
//! cells, cells with M P < -f (dense) or a new negative node (TT), TT cells skipped for
//! rank capacity, cells].
std::vector<Real> HistoryLomacSums(MeshData<Real> *md) {
  constexpr int NSUM = 4;
  auto pm = md->GetParentPointer();
  static auto desc = MakePackDescriptor<fields::lomac_stat>(pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  std::vector<Real> sums(NSUM, 0.0);
  if (v.GetNBlocks() == 0) return sums;
  using rt =
      RiotFlatReduce::ReductionType<RiotUtils::GlobalSum<Real, Kokkos::HostSpace, NSUM>>;
  auto space = rt::GetIndexSpace(IndexDomain::interior, v.GetNBlocks(), md);
  const auto result = rt::four_d(
      "Kinetics::HistoryLomacSums", space,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    RiotUtils::array_type<Real, NSUM> &acc) {
        const Real s = v(b, fields::lomac_stat(0), k, j, i);
        acc.my_array[0] += (s > 0.0) ? 1.0 : 0.0;
        acc.my_array[1] += v(b, fields::lomac_stat(1), k, j, i);
        acc.my_array[2] += (s == 2.0) ? 1.0 : 0.0;
        acc.my_array[3] += 1.0;
      });
  Kokkos::fence();
  for (int a = 0; a < NSUM; ++a)
    sums[a] = result.my_array[a];
  return sums;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection Kinetics::LomacTasks
//! \brief Post-step task list of kinetics/lomac: correct f on base, refresh its ghosts
//! and recompute the closure from the corrected f (the next step's closure_old, S3-Q4),
//! then the skip check.
TaskCollection LomacTasks(Mesh *pm, parthenon::SimTime &tm, const Real dt) {
  using parthenon::BoundaryType;
  TaskCollection tc;
  TaskID none(0);
  const int num_partitions = pm->DefaultNumPartitions();
  const bool tt =
      GetRepresentation(pm->packages.Get(pkg_name).get()) == Representation::tt;
  auto &base = pm->mesh_data.Get();
  // f only, sharing base's memory, for the ghost exchange
  pm->mesh_data.AddShallow(lomac_name, base,
                           {tt ? fields::f_tt::name() : fields::f::name()});
  TaskRegion &region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    auto &tl = region[i];
    auto &mb = pm->mesh_data.GetOrAdd("base", i);
    auto &mf = pm->mesh_data.GetOrAdd(lomac_name, i);
    auto corr = tl.AddTask(none, LomacCorrect, mb.get());
    auto recv = tl.AddTask(corr, parthenon::StartReceiveBoundBufs<BoundaryType::any>, mf);
    auto bc = parthenon::AddBoundaryExchangeTasks(recv, tl, mf, pm->multilevel);
    tl.AddTask(bc, ComputeClosure, mb.get());
  }
  TaskRegion &check = tc.AddRegion(1);
  check[0].AddTask(none, CheckLomacSkips, pm);
  return tc;
}

} // namespace Kinetics
