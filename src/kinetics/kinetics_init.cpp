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

// Initialization of f from the hydro state, and the velocity-resolution checks.
//
// Parthenon calls package PostInitialization after the problem generator, the derived
// fill and the boundary exchange (external/parthenon/src/mesh/mesh.cpp,
// Mesh::Initialize), so the bulk density, velocity and temperature are valid on the whole
// block, ghosts included. f is filled on the entire domain and Parthenon exchanges
// boundaries again afterwards.

#include <iostream>
#include <sstream>
#include <string>

#include <parthenon_mpi.hpp>

#include "kinetics/equilibrium.hpp"
#include "kinetics/kinetics.hpp"
#include "kinetics/kinetics_cell.hpp"
#include "kinetics/moments.hpp"
#include "kinetics/velocity_grid.hpp"
#include "riot_utils/riot_loops.hpp"
#include "variables.hpp"

namespace Kinetics {

//----------------------------------------------------------------------------------------
//! \fn  void Kinetics::PostInitialization
//! \brief Fill f from the hydro state: the discrete equilibrium, a bi-Maxwellian, or two
//! Maxwellians drifting apart along init_axis (counter-streaming, rank 2 in TT).
//! two_maxwellian: halves of density n/2 at u -+ delta e_axis, delta = init_drift
//! sqrt(theta), and theta' = theta - delta^2 / 3, so that n, u and the energy equal the
//! hydro state. Each half is a discrete equilibrium, so the sum has those moments
//! exactly on the grid.
void PostInitialization(Mesh *pm, ParameterInput *pin, MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  // No early return on ranks without blocks: they must still join the collectives below.
  auto pkg = pm->packages.Get(pkg_name);
  const auto grid = pkg->Param<VelocityGrid>("grid");
  const auto species = pkg->Param<Species>("species");
  const auto eq_params = pkg->Param<EquilibriumParams>("eq_params");
  const std::string init = pkg->Param<std::string>("init");
  const bool bimaxwellian = (init == "bimaxwellian");
  const bool two = (init == "two_maxwellian");
  const Real drift = pkg->Param<Real>("init_drift");
  const auto kind = GetCellKind(pkg.get());
  const Real T_ratio = pkg->Param<Real>("init_T_ratio");
  const int par_axis = pkg->Param<int>("init_axis") - 1;

  static auto desc =
      MakePackDescriptor<ccbulk::rho, ccbulk::velocity, ccbulk::temperature, fields::f,
                         fields::f_tt, fields::eq_fallback>(pm->resolved_packages.get());
  auto v = desc.GetPack(md);

  // theta_par / theta and theta_perp / theta at fixed total energy (mean of the three
  // per-axis temperatures equals the hydro temperature).
  const Real par_scale = 3.0 * T_ratio / (T_ratio + 2.0);
  const Real perp_scale = 3.0 / (T_ratio + 2.0);

  using rt = RiotFlatReduce::ReductionType<Kokkos::Sum<Real>>;
  auto space = rt::GetIndexSpace(IndexDomain::entire, v.GetNBlocks(), md);
  Real nfallback = 0.0, ncells = 0.0; // Real: cell counts may exceed int
  if (v.GetNBlocks() > 0) {
    nfallback = rt::four_d(
        "Kinetics::PostInitialization", space,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &nfail) {
          const Real theta = species.kb_per_m * v(b, ccbulk::temperature(), k, j, i);
          EquilibriumTarget target;
          target.n = v(b, ccbulk::rho(), k, j, i) / species.mass;
          for (int d = 0; d < 3; ++d) {
            target.u[d] = v(b, ccbulk::velocity(d), k, j, i);
            target.theta[d] = theta * (bimaxwellian ? perp_scale : 1.0);
          }
          Maxwellian eq[2];
          bool failed = false;
          if (bimaxwellian) {
            target.theta[par_axis] = theta * par_scale;
            const auto res = SolveAnisotropicEquilibrium(grid, target, eq_params, eq[0]);
            failed = res.status == EquilibriumResult::Status::fallback;
          } else if (two) {
            const Real delta = drift * std::sqrt(theta);
            target.n *= 0.5;
            for (int d = 0; d < 3; ++d)
              target.theta[d] = theta - delta * delta / 3.0;
            for (int c = 0; c < 2; ++c) {
              EquilibriumTarget half = target;
              half.u[par_axis] += (c == 0) ? -delta : delta;
              const auto res = SolveEquilibrium(grid, half, eq_params, eq[c]);
              failed = failed || (res.status == EquilibriumResult::Status::fallback);
            }
          } else {
            const auto res = SolveEquilibrium(grid, target, eq_params, eq[0]);
            failed = res.status == EquilibriumResult::Status::fallback;
          }
          WithCell(kind, v, b, k, j, i,
                   [&](const auto &cell) { cell.Fill(eq, two ? 2 : 1); });
          v(b, fields::eq_fallback(), k, j, i) = failed ? 1.0 : 0.0;
          nfail += failed ? 1.0 : 0.0;
        });
    ncells = static_cast<Real>(space.nblocks) * (space.kb.e - space.kb.s + 1) *
             (space.jb.e - space.jb.s + 1) * (space.ib.e - space.ib.s + 1);
  }
  Real counts[2] = {nfallback, ncells};
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_PARTHENON_REAL, MPI_SUM,
                                    MPI_COMM_WORLD));
#endif
  const Real frac = (counts[1] > 0.0) ? counts[0] / counts[1] : 0.0;
  if (Globals::my_rank == 0) {
    std::cout << "kinetics: initialized f (" << init
              << "); equilibrium fallbacks: " << counts[0] << " of " << counts[1]
              << " cells (ghosts included)" << std::endl;
  }
  PARTHENON_REQUIRE(frac <= pkg->Param<Real>("eq_fallback_abort"),
                    "kinetics: too many cells fell back to the sampled Maxwellian at "
                    "initialization; widen or refine the velocity box");

  CheckResolution(pm, md, "initialization");
}

//----------------------------------------------------------------------------------------
//! \fn  ResolutionReport Kinetics::CheckResolution
//! \brief Mass on the outermost velocity-node layer and thermal resolution, globally.
ResolutionReport CheckResolution(Mesh *pm, MeshData<Real> *md, const std::string &when) {
  auto pkg = pm->packages.Get(pkg_name);
  const auto grid = pkg->Param<VelocityGrid>("grid");

  static auto desc =
      MakePackDescriptor<fields::f, fields::f_tt>(pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  const auto kind = GetCellKind(pkg.get());

  ResolutionReport report{0.0, std::numeric_limits<Real>::max()};
  if (v.GetNBlocks() > 0) {
    // An axis with fewer than 3 nodes has no interior to compare against; skip it.
    const bool edge_axis[3] = {grid.nv[0] >= 3, grid.nv[1] >= 3, grid.nv[2] >= 3};
    using rmax = RiotFlatReduce::ReductionType<Kokkos::Max<Real>>;
    using rmin = RiotFlatReduce::ReductionType<Kokkos::Min<Real>>;
    auto space = rmax::GetIndexSpace(IndexDomain::interior, v.GetNBlocks(), md);
    report.max_edge_mass_fraction = rmax::four_d(
        "Kinetics::EdgeMass", space,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmax) {
          Real total = 0.0, edge = 0.0;
          WithCell(kind, v, b, k, j, i, [&](const auto &cell) {
            cell.ForEach([&](const int ix, const int iy, const int iz, const Real fv) {
              const int idx[3] = {ix, iy, iz};
              bool on_edge = false;
              for (int d = 0; d < 3; ++d)
                on_edge = on_edge ||
                          (edge_axis[d] && (idx[d] == 0 || idx[d] == grid.nv[d] - 1));
              total += std::abs(fv);
              edge += on_edge ? std::abs(fv) : 0.0;
            });
          });
          lmax = std::max(lmax, total > 0.0 ? edge / total : 0.0);
        });
    report.min_vth_over_dv = rmin::four_d(
        "Kinetics::ThermalResolution", space,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmin) {
          RawMoments m;
          WithCell(kind, v, b, k, j, i, [&](const auto &cell) { m = cell.Raw(); });
          const auto t = IsotropicTarget(m);
          const Real vth = std::sqrt(std::max(t.theta[0], 0.0));
          for (int d = 0; d < 3; ++d)
            lmin = std::min(lmin, vth / grid.dv[d]);
        });
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &report.max_edge_mass_fraction, 1,
                                    MPI_PARTHENON_REAL, MPI_MAX, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &report.min_vth_over_dv, 1,
                                    MPI_PARTHENON_REAL, MPI_MIN, MPI_COMM_WORLD));
#endif

  const Real warn = pkg->Param<Real>("edge_mass_warn");
  const Real abort = pkg->Param<Real>("edge_mass_abort");
  const Real min_res = pkg->Param<Real>("min_vth_over_dv");
  if (Globals::my_rank == 0) {
    std::stringstream msg;
    msg << "kinetics (" << when << "): max mass fraction on the velocity-box edge "
        << report.max_edge_mass_fraction << ", min v_th/dv " << report.min_vth_over_dv;
    if (report.max_edge_mass_fraction > warn)
      msg << "\n  WARNING: velocity box edge mass above kinetics/edge_mass_warn = "
          << warn;
    if (report.min_vth_over_dv < min_res)
      msg << "\n  WARNING: thermal speed under-resolved, v_th/dv below "
             "kinetics/min_vth_over_dv = "
          << min_res;
    std::cout << msg.str() << std::endl;
  }
  PARTHENON_REQUIRE(report.max_edge_mass_fraction <= abort,
                    "kinetics: mass on the velocity-box edge above "
                    "kinetics/edge_mass_abort; widen the velocity box");
  return report;
}

} // namespace Kinetics
