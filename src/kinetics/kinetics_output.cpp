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
      MakePackDescriptor<fields::f, fields::rho, fields::velocity, fields::temperature,
                         fields::pressure, fields::stress, fields::heat_flux>(
          pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  if (v.GetNBlocks() == 0) return;

  auto pkg = pm->packages.Get(pkg_name);
  const auto grid = pkg->Param<VelocityGrid>("grid");
  const auto species = pkg->Param<Species>("species");

  auto space = RiotFlatLoop::GetIndexSpace(IndexDomain::interior, v.GetNBlocks(), md);
  RiotFlatLoop::four_d(
      "Kinetics::SetDerivedMoments", space,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto f = [&](const int n) { return v(b, fields::f(n), k, j, i); };
        const auto raw = ComputeRawMoments(grid, f);
        const auto t = IsotropicTarget(raw);
        const auto c = ComputeCentralMoments(grid, f, t.u);
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
      MakePackDescriptor<fields::f, ccmat::rho, ccbulk::momentum,
                         ccbulk::total_material_energy>(pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  std::vector<Real> sums(NSUM, 0.0);
  if (v.GetNBlocks() == 0) return sums;

  auto pkg = pm->packages.Get(pkg_name);
  const auto grid = pkg->Param<VelocityGrid>("grid");
  const Real m = pkg->Param<Species>("species").mass;
  const Real w = grid.Weight();

  using rt =
      RiotFlatReduce::ReductionType<RiotUtils::GlobalSum<Real, Kokkos::HostSpace, NSUM>>;
  auto space = rt::GetIndexSpace(IndexDomain::interior, v.GetNBlocks(), md);
  const auto result = rt::four_d(
      "Kinetics::HistorySums", space,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    RiotUtils::array_type<Real, NSUM> &acc) {
        const Real dV = v.GetCoordinates(b).CellVolume(k, j, i);
        Real s[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
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
  static auto desc = MakePackDescriptor<fields::f>(pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  if (v.GetNBlocks() == 0) return std::numeric_limits<Real>::max();
  const int nv = pm->packages.Get(pkg_name)->Param<VelocityGrid>("grid").Size();

  using rt = RiotFlatReduce::ReductionType<Kokkos::Min<Real>>;
  auto space = rt::GetIndexSpace(IndexDomain::interior, v.GetNBlocks(), nv, md);
  return rt::five_d(
      "Kinetics::HistoryMinF", space,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i,
                    Real &lmin) { lmin = std::min(lmin, v(b, fields::f(n), k, j, i)); });
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
//! \fn  Real Kinetics::HistorySubsteps
//! \brief Kinetic substeps taken in the last hydro step.
Real HistorySubsteps(MeshData<Real> *md) {
  return md->GetParentPointer()->packages.Get(pkg_name)->Param<int>("substeps");
}

} // namespace Kinetics
