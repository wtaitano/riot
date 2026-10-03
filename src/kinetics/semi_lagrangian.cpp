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

#include <algorithm>
#include <limits>

#include <parthenon_mpi.hpp>

#include "kinetics/kinetics.hpp"
#include "kinetics/semi_lagrangian.hpp"
#include "kinetics/tt_stream.hpp"
#include "kinetics/tt_tensor.hpp"
#include "kinetics/velocity_grid.hpp"
#include "riot_utils/riot_loops.hpp"

namespace Kinetics {

namespace {

// representation = tt: one directional sweep, dst <- SL_d(src) on the cells in
// [kb, jb, ib], one cell per team (level-1 team scratch for the two roundings of
// tt_stream.hpp). Rounding statistics are added for interior cells only.
void SweepTT(MeshData<Real> *src, MeshData<Real> *dst, const int d, const Real h,
             const IndexRange kb, const IndexRange jb, const IndexRange ib) {
  auto pm = dst->GetParentPointer();
  static auto desc =
      MakePackDescriptor<fields::f_tt, fields::tt_round>(pm->resolved_packages.get());
  auto vs = desc.GetPack(src);
  auto vd = desc.GetPack(dst);
  if (vd.GetNBlocks() == 0) return;

  auto pkg = pm->packages.Get(pkg_name);
  const auto grid = pkg->Param<VelocityGrid>("grid");
  const int order = pkg->Param<SLParams>("sl_params").order;
  const auto L = pkg->Param<TT::TTLayout>("tt_layout");
  const auto prm = pkg->Param<TT::RoundParams>("tt_round");
  const auto sc = TT::MakeStreamScratch(grid, L.rcap);
  const int nwork = sc.Size();
  const std::size_t scratch_bytes = parthenon::ScratchPad1D<Real>::shmem_size(nwork);
  TT::RequireTeamScratch(scratch_bytes, "streaming");
  constexpr int scratch_level = 1;
  const int di = (d == 0), dj = (d == 1), dk = (d == 2);
  const auto ii = dst->GetBoundsI(IndexDomain::interior);
  const auto ji = dst->GetBoundsJ(IndexDomain::interior);
  const auto ki = dst->GetBoundsK(IndexDomain::interior);

  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Kinetics::StreamTT", DevExecSpace(), scratch_bytes,
      scratch_level, 0, vd.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j,
                    const int i) {
        parthenon::ScratchPad1D<Real> work(member.team_scratch(scratch_level), nwork);
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          using PC = TT::PackCell<decltype(vs), fields::f_tt>;
          const Real h_over_dx = h / vd.GetCoordinates(b).Dx(d + 1);
          TT::RoundTally tally;
          TT::StreamCellTT(grid, order, d, h_over_dx,
                           TT::MakeRef(PC{vs, b, k - dk, j - dj, i - di}, L),
                           TT::MakeRef(PC{vs, b, k, j, i}, L),
                           TT::MakeRef(PC{vs, b, k + dk, j + dj, i + di}, L),
                           TT::MakeOutRef(PC{vd, b, k, j, i}, L), sc, work.data(), prm,
                           tally);
          const bool interior = (k >= ki.s && k <= ki.e && j >= ji.s && j <= ji.e &&
                                 i >= ii.s && i <= ii.e);
          if (interior) {
            vd(b, fields::tt_round(0), k, j, i) += tally.rel_discarded;
            vd(b, fields::tt_round(1), k, j, i) += tally.cap_hits;
            vd(b, fields::tt_round(2), k, j, i) += tally.svd_failures;
          }
        });
      });
}

// representation = tt: dimension-by-dimension sweeps x1, x2, x3 (S1_DESIGN.md, S1-Q5).
// Sweep d covers the interior along the swept directions and one ghost layer along the
// directions still to sweep, so the next sweep has its stencil; before rounding this is
// exactly the dense single-pass tensor-product update, with one ghost exchange per step.
// Registers: 1D src -> dst; 2D src -> tmp -> dst; 3D src -> dst -> tmp -> dst.
// In 3D the first sweep also writes one ghost layer of dst (k0, i.e. the base data),
// which stays half swept until the next boundary exchange; nothing reads base ghosts in
// between (outputs use the interior).
void StreamTT(MeshData<Real> *src, MeshData<Real> *tmp, MeshData<Real> *dst,
              const Real h) {
  const int ndim = dst->GetParentPointer()->ndim;
  PARTHENON_REQUIRE(ndim == 1 || tmp != nullptr,
                    "kinetics: multi-D tensor-train streaming needs a scratch register");
  // reg[d] = {input, output} of sweep d; the last sweep always writes dst.
  MeshData<Real> *reg[3][2] = {{src, dst}, {nullptr, nullptr}, {nullptr, nullptr}};
  if (ndim == 2) {
    reg[0][1] = tmp;
    reg[1][0] = tmp;
    reg[1][1] = dst;
  } else if (ndim == 3) {
    reg[1][0] = dst;
    reg[1][1] = tmp;
    reg[2][0] = tmp;
    reg[2][1] = dst;
  }
  for (int d = 0; d < ndim; ++d) {
    IndexRange r[3] = {dst->GetBoundsI(IndexDomain::interior),
                       dst->GetBoundsJ(IndexDomain::interior),
                       dst->GetBoundsK(IndexDomain::interior)};
    for (int e = d + 1; e < ndim; ++e) {
      r[e].s -= 1;
      r[e].e += 1;
    }
    SweepTT(reg[d][0], reg[d][1], d, h, r[2], r[1], r[0]);
  }
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Kinetics::Stream
//! \brief One semi-Lagrangian step of length h: dst(interior) <- SL(src). tmp is a third
//! register used by the multi-D tensor-train sweeps (may be null otherwise). src must
//! have valid ghost cells (one layer is enough under the |s| <= 1 cap).
TaskStatus Stream(MeshData<Real> *src, MeshData<Real> *tmp, MeshData<Real> *dst,
                  const Real h) {
  auto pm = dst->GetParentPointer();
  if (GetRepresentation(pm->packages.Get(pkg_name).get()) == Representation::tt) {
    StreamTT(src, tmp, dst, h);
    return TaskStatus::complete;
  }
  static auto desc = MakePackDescriptor<fields::f>(pm->resolved_packages.get());
  auto vs = desc.GetPack(src);
  auto vd = desc.GetPack(dst);
  if (vd.GetNBlocks() == 0) return TaskStatus::complete;

  auto pkg = pm->packages.Get(pkg_name);
  const auto grid = pkg->Param<VelocityGrid>("grid");
  const auto sl = pkg->Param<SLParams>("sl_params");
  const int ndim = pm->ndim;

  auto space = RiotFlatLoop::GetIndexSpace(IndexDomain::interior, vd.GetNBlocks(),
                                           grid.Size(), dst);
  RiotFlatLoop::five_d(
      "Kinetics::Stream", space,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        int idx[3];
        grid.Unflatten(n, idx[0], idx[1], idx[2]);
        const auto &coords = vd.GetCoordinates(b);
        Real shift[3] = {0.0, 0.0, 0.0};
        for (int d = 0; d < ndim; ++d)
          shift[d] = grid.Node(d, idx[d]) * h / coords.Dx(d + 1);
        const Real val = SLInterpolate(sl, ndim, shift, [&](int a, int bb, int c) {
          return vs(b, fields::f(n), k + c, j + bb, i + a);
        });
        vd(b, fields::f(n), k, j, i) = val;
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Kinetics::MaxStreamingStep
//! \brief Largest substep with |v_d| h / dx_d <= cfl on every active axis, over the whole
//! mesh (MPI-reduced).
Real MaxStreamingStep(Mesh *pm) {
  auto pkg = pm->packages.Get(pkg_name);
  const auto grid = pkg->Param<VelocityGrid>("grid");
  const Real cfl = pkg->Param<Real>("cfl");
  Real hmax = std::numeric_limits<Real>::max();
  for (auto &pmb : pm->block_list) {
    for (int d = 0; d < pm->ndim; ++d) {
      const Real vmax = grid.MaxNodeSpeed(d);
      if (vmax > 0.0) hmax = std::min(hmax, cfl * pmb->coords.Dx(d + 1) / vmax);
    }
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(
      MPI_Allreduce(MPI_IN_PLACE, &hmax, 1, MPI_PARTHENON_REAL, MPI_MIN, MPI_COMM_WORLD));
#endif
  return hmax;
}

} // namespace Kinetics
