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

// Derived moment fields and history output of the kinetics package.

#include <cmath>
#include <limits>
#include <vector>

#include "kinetics/kinetics.hpp"
#include "kinetics/kinetics_cell.hpp"
#include "kinetics/moments.hpp"
#include "kinetics/velocity_grid.hpp"
#include "riot_utils/riot_loops.hpp"
#include "riot_utils/riot_utils.hpp"
#include "variables.hpp"

namespace Kinetics {

//----------------------------------------------------------------------------------------
//! \fn  void Kinetics::SetDerivedMomentsMesh
//! \brief Mass density, velocity, temperature, pressure, stress and heat flux from f.
void SetDerivedMomentsMesh(Mesh *pm, ParameterInput *pin, parthenon::SimTime &tm) {
  auto md = pm->mesh_data.Get().get();
  static auto desc =
      MakePackDescriptor<fields::f, fields::f_tt, fields::rho, fields::velocity,
                         fields::temperature, fields::pressure, fields::stress,
                         fields::heat_flux, fields::tt_rank>(pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  if (v.GetNBlocks() == 0) return;

  auto pkg = pm->packages.Get(pkg_name);
  const auto species = pkg->Param<Species>("species");
  const auto kind = GetCellKind(pkg.get());

  auto space = RiotFlatLoop::GetIndexSpace(IndexDomain::interior, v.GetNBlocks(), md);
  RiotFlatLoop::four_d(
      "Kinetics::SetDerivedMoments", space,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        RawMoments raw;
        EquilibriumTarget t;
        CentralMoments c;
        WithCell(kind, v, b, k, j, i, [&](const auto &cell) {
          raw = cell.Raw();
          t = IsotropicTarget(raw);
          c = cell.Central(t.u);
        });
        if (kind.tt) {
          v(b, fields::tt_rank(0), k, j, i) = v(b, fields::f_tt(0), k, j, i);
          v(b, fields::tt_rank(1), k, j, i) = v(b, fields::f_tt(1), k, j, i);
        }
        const Real m = species.mass;
        v(b, fields::rho(), k, j, i) = m * t.n;
        for (int d = 0; d < 3; ++d) {
          v(b, fields::velocity(d), k, j, i) = t.u[d];
          v(b, fields::heat_flux(d), k, j, i) = m * c.heat[d];
        }
        v(b, fields::temperature(), k, j, i) = t.theta[0] / species.kb_per_m;
        v(b, fields::pressure(), k, j, i) = m * t.n * t.theta[0];
        for (int a = 0; a < 6; ++a)
          v(b, fields::stress(a), k, j, i) = m * c.stress[a];
      });
}

//----------------------------------------------------------------------------------------
//! \fn  std::vector<Real> Kinetics::HistorySums
//! \brief Volume-integrated invariants. Columns (suffix _n of "kinetics_sums"):
//!   0 kinetic mass, 1-3 kinetic momentum, 4 kinetic energy,
//!   5 entropy H = sum f (ln f - 1) w dV over f > 0, 6 negative mass (m sum f w dV, f<0),
//!   7 hydro mass, 8-10 hydro momentum, 11 hydro total energy.
std::vector<Real> HistorySums(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  constexpr int NSUM = 12;
  auto pm = md->GetParentPointer();
  static auto desc =
      MakePackDescriptor<fields::f, fields::f_tt, ccmat::rho, ccbulk::momentum,
                         ccbulk::total_material_energy>(pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  std::vector<Real> sums(NSUM, 0.0);
  if (v.GetNBlocks() == 0) return sums;

  auto pkg = pm->packages.Get(pkg_name);
  const auto grid = pkg->Param<VelocityGrid>("grid");
  const Real m = pkg->Param<Species>("species").mass;
  const Real w = grid.Weight();
  const auto kind = GetCellKind(pkg.get());

  using rt =
      RiotFlatReduce::ReductionType<RiotUtils::GlobalSum<Real, Kokkos::HostSpace, NSUM>>;
  auto space = rt::GetIndexSpace(IndexDomain::interior, v.GetNBlocks(), md);
  const auto result = rt::four_d(
      "Kinetics::HistorySums", space,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    RiotUtils::array_type<Real, NSUM> &acc) {
        const Real dV = v.GetCoordinates(b).CellVolume(k, j, i);
        // Dense: one pass over the nodes. TT: invariants by exact core contractions,
        // entropy and negative mass node by node (decompression, tt_diag = exact).
        Real s[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        if (!kind.tt) {
          for (int iz = 0; iz < grid.nv[2]; ++iz) {
            const Real vz = grid.Node(2, iz);
            for (int iy = 0; iy < grid.nv[1]; ++iy) {
              const Real vy = grid.Node(1, iy);
              for (int ix = 0; ix < grid.nv[0]; ++ix) {
                const Real vx = grid.Node(0, ix);
                const Real fv = v(b, fields::f(grid.Flat(ix, iy, iz)), k, j, i);
                s[0] += fv;
                s[1] += vx * fv;
                s[2] += vy * fv;
                s[3] += vz * fv;
                s[4] += 0.5 * (vx * vx + vy * vy + vz * vz) * fv;
                s[5] += (fv > 0.0) ? fv * (std::log(fv) - 1.0) : 0.0;
                s[6] += (fv < 0.0) ? fv : 0.0;
              }
            }
          }
        } else {
          const TTCell<decltype(v)> cell{v, b, k, j, i, kind.grid, kind.L};
          const auto raw = cell.Raw();
          s[0] = raw.n / w;
          for (int d = 0; d < 3; ++d)
            s[1 + d] = raw.nu[d] / w;
          s[4] = 0.5 * (raw.nvv[0] + raw.nvv[1] + raw.nvv[2]) / w;
          cell.ForEach([&](const int, const int, const int, const Real fv) {
            s[5] += (fv > 0.0) ? fv * (std::log(fv) - 1.0) : 0.0;
            s[6] += (fv < 0.0) ? fv : 0.0;
          });
        }
        for (int a = 0; a < 5; ++a)
          acc.my_array[a] += m * s[a] * w * dV;
        acc.my_array[5] += s[5] * w * dV;
        acc.my_array[6] += m * s[6] * w * dV;
        Real rho = 0.0;
        for (int mm = 0; mm < v.GetSize(b, ccmat::rho()); ++mm)
          rho += v(b, ccmat::rho(mm), k, j, i);
        acc.my_array[7] += rho * dV;
        for (int d = 0; d < 3; ++d)
          acc.my_array[8 + d] += v(b, ccbulk::momentum(d), k, j, i) * dV;
        acc.my_array[11] += v(b, ccbulk::total_material_energy(), k, j, i) * dV;
      });
  Kokkos::fence();
  for (int a = 0; a < NSUM; ++a)
    sums[a] = result.my_array[a];
  return sums;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Kinetics::HistoryMinF
//! \brief Smallest value of f over all interior cells and velocity nodes.
Real HistoryMinF(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  static auto desc =
      MakePackDescriptor<fields::f, fields::f_tt>(pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  if (v.GetNBlocks() == 0) return std::numeric_limits<Real>::max();
  auto pkg = pm->packages.Get(pkg_name);
  const auto kind = GetCellKind(pkg.get());

  if (!kind.tt) {
    const int nv = kind.grid.Size();
    using rt = RiotFlatReduce::ReductionType<Kokkos::Min<Real>>;
    auto space = rt::GetIndexSpace(IndexDomain::interior, v.GetNBlocks(), nv, md);
    return rt::five_d(
        "Kinetics::HistoryMinF", space,
        KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i,
                      Real &lmin) {
          lmin = std::min(lmin, v(b, fields::f(n), k, j, i));
        });
  }
  using rt = RiotFlatReduce::ReductionType<Kokkos::Min<Real>>;
  auto space = rt::GetIndexSpace(IndexDomain::interior, v.GetNBlocks(), md);
  return rt::four_d(
      "Kinetics::HistoryMinF", space,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmin) {
        WithCell(kind, v, b, k, j, i, [&](const auto &cell) {
          cell.ForEach([&](const int, const int, const int, const Real fv) {
            lmin = std::min(lmin, fv);
          });
        });
      });
}

//----------------------------------------------------------------------------------------
//! \fn  Real Kinetics::HistoryFallbackCount
//! \brief Interior cells whose last equilibrium solve fell back to the sampled
//! Maxwellian.
Real HistoryFallbackCount(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  static auto desc = MakePackDescriptor<fields::eq_fallback>(pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  if (v.GetNBlocks() == 0) return 0.0;
  using rt = RiotFlatReduce::ReductionType<Kokkos::Sum<Real>>;
  auto space = rt::GetIndexSpace(IndexDomain::interior, v.GetNBlocks(), md);
  return rt::four_d(
      "Kinetics::HistoryFallbackCount", space,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        lsum += v(b, fields::eq_fallback(), k, j, i);
      });
}

//----------------------------------------------------------------------------------------
//! \fn  std::vector<Real> Kinetics::HistoryRankSums
//! \brief representation = tt: interior cell count, sum of r1, sum of r2, and the number
//! of cells with a rank equal to tt_rank_max (at the cap, which includes cells that are
//! exact at that rank; rounding cap hits proper are reported once f evolves).
std::vector<Real> HistoryRankSums(MeshData<Real> *md) {
  constexpr int NSUM = 4;
  auto pm = md->GetParentPointer();
  static auto desc = MakePackDescriptor<fields::f_tt>(pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  std::vector<Real> sums(NSUM, 0.0);
  if (v.GetNBlocks() == 0) return sums;
  const int rcap = pm->packages.Get(pkg_name)->Param<TT::TTLayout>("tt_layout").rcap;
  using rt =
      RiotFlatReduce::ReductionType<RiotUtils::GlobalSum<Real, Kokkos::HostSpace, NSUM>>;
  auto space = rt::GetIndexSpace(IndexDomain::interior, v.GetNBlocks(), md);
  const auto result = rt::four_d(
      "Kinetics::HistoryRankSums", space,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    RiotUtils::array_type<Real, NSUM> &acc) {
        const Real r1 = v(b, fields::f_tt(0), k, j, i);
        const Real r2 = v(b, fields::f_tt(1), k, j, i);
        acc.my_array[0] += 1.0;
        acc.my_array[1] += r1;
        acc.my_array[2] += r2;
        acc.my_array[3] += (r1 >= rcap || r2 >= rcap) ? 1.0 : 0.0;
      });
  Kokkos::fence();
  for (int a = 0; a < NSUM; ++a)
    sums[a] = result.my_array[a];
  return sums;
}

//----------------------------------------------------------------------------------------
//! \fn  std::vector<Real> Kinetics::HistoryRoundSums
//! \brief representation = tt, over the last hydro step and all interior cells: sum of
//! relative discarded norms of the roundings, rank-cap hits, non-converged SVDs.
std::vector<Real> HistoryRoundSums(MeshData<Real> *md) {
  constexpr int NSUM = 3;
  auto pm = md->GetParentPointer();
  static auto desc = MakePackDescriptor<fields::tt_round>(pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  std::vector<Real> sums(NSUM, 0.0);
  if (v.GetNBlocks() == 0) return sums;
  using rt =
      RiotFlatReduce::ReductionType<RiotUtils::GlobalSum<Real, Kokkos::HostSpace, NSUM>>;
  auto space = rt::GetIndexSpace(IndexDomain::interior, v.GetNBlocks(), md);
  const auto result = rt::four_d(
      "Kinetics::HistoryRoundSums", space,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    RiotUtils::array_type<Real, NSUM> &acc) {
        for (int a = 0; a < NSUM; ++a)
          acc.my_array[a] += v(b, fields::tt_round(a), k, j, i);
      });
  Kokkos::fence();
  for (int a = 0; a < NSUM; ++a)
    sums[a] = result.my_array[a];
  return sums;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Kinetics::HistoryMaxRank
//! \brief representation = tt: largest of r1, r2 over the interior cells.
Real HistoryMaxRank(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  static auto desc = MakePackDescriptor<fields::f_tt>(pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  if (v.GetNBlocks() == 0) return 0.0;
  using rt = RiotFlatReduce::ReductionType<Kokkos::Max<Real>>;
  auto space = rt::GetIndexSpace(IndexDomain::interior, v.GetNBlocks(), md);
  return rt::four_d(
      "Kinetics::HistoryMaxRank", space,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmax) {
        lmax = std::max(lmax, std::max(v(b, fields::f_tt(0), k, j, i),
                                       v(b, fields::f_tt(1), k, j, i)));
      });
}

//----------------------------------------------------------------------------------------
//! \fn  Real Kinetics::HistorySubsteps
//! \brief Kinetic substeps taken in the last hydro step.
Real HistorySubsteps(MeshData<Real> *md) {
  return md->GetParentPointer()->packages.Get(pkg_name)->Param<int>("substeps");
}

} // namespace Kinetics
