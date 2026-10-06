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

// Closure coupling (claude_sessions/kinetic_bgk/CLOSURE_DESIGN.md): the hydro fluxes take
// the non-equilibrium stress Pi = P - p I and the heat flux q of f, so that hydro solves
// the moment equations of the BGK model,
//
//   F_mom_i += Pi_id,    F_E += Pi_id u_i + q_d    (face normal d, Cartesian),
//
// with u the hydro face velocity. The coupling is one-way: f does not see hydro. With
// closure_coupling the kinetics step runs before hydro; closure_old holds the closure of
// f at t^n and closure_new that at t^{n+1}, and RK stage s uses
// (1 - c_s) old + c_s new. Face values average the two adjacent cells.

#include "kinetics/kinetics.hpp"
#include "kinetics/kinetics_cell.hpp"
#include "kinetics/moments.hpp"
#include "riot_utils/riot_loops.hpp"
#include "variables.hpp"

namespace Kinetics {

namespace {

// Component of the symmetric Pi (xx, yy, zz, xy, xz, yz) for the index pair (a, d).
KOKKOS_FORCEINLINE_FUNCTION int Sym(const int a, const int d) {
  if (a == d) return a;
  const int s = a + d; // 1: xy, 2: xz, 3: yz
  return 2 + s;
}

template <parthenon::CoordinateDirection DIR, class Pack>
void AddClosureFluxesDir(MeshData<Real> *md, const Pack &v, const Real w) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using TE = parthenon::TopologicalElement;
  constexpr TE te = (DIR == X1DIR) ? TE::F1 : ((DIR == X2DIR) ? TE::F2 : TE::F3);
  constexpr int d = static_cast<int>(DIR) - 1;
  constexpr int di = (d == 0), dj = (d == 1), dk = (d == 2);
  auto space = RiotFlatLoop::GetIndexSpace(IndexDomain::interior, v.GetNBlocks(), md, te);
  RiotFlatLoop::four_d(
      "Kinetics::AddClosureFluxes", space,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        // closure at the face: time interpolation, then the average of the two cells
        const auto face = [&](const int n) {
          const Real lo =
              (1.0 - w) * v(b, fields::closure_old(n), k - dk, j - dj, i - di) +
              w * v(b, fields::closure_new(n), k - dk, j - dj, i - di);
          const Real hi = (1.0 - w) * v(b, fields::closure_old(n), k, j, i) +
                          w * v(b, fields::closure_new(n), k, j, i);
          return 0.5 * (lo + hi);
        };
        Real fe = face(6 + d);
        for (int a = 0; a < 3; ++a) {
          const Real pi = face(Sym(a, d));
          v.flux(b, DIR, ccbulk::momentum(a), k, j, i) += pi;
          fe += pi * v(b, ccbulk::face_velocity(3 * d + a), k, j, i);
        }
        v.flux(b, DIR, ccbulk::total_material_energy(), k, j, i) += fe;
      });
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Kinetics::ComputeClosure
//! \brief closure_new <- (Pi, q) of f on the entire block. Pi = m sum c c f w - p I,
//! p = m n theta with theta from the energy, so Pi is traceless.
TaskStatus ComputeClosure(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  static auto desc = MakePackDescriptor<fields::f, fields::f_tt, fields::closure_new>(
      pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  if (v.GetNBlocks() == 0) return TaskStatus::complete;
  auto pkg = pm->packages.Get(pkg_name);
  const Real m = pkg->Param<Species>("species").mass;
  const auto kind = GetCellKind(pkg.get());
  auto space = RiotFlatLoop::GetIndexSpace(IndexDomain::entire, v.GetNBlocks(), md);
  RiotFlatLoop::four_d(
      "Kinetics::ComputeClosure", space,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        EquilibriumTarget t;
        CentralMoments c;
        WithCell(kind, v, b, k, j, i, [&](const auto &cell) {
          t = IsotropicTarget(cell.Raw());
          c = cell.Central(t.u);
        });
        const Real p = m * t.n * t.theta[0];
        for (int a = 0; a < 6; ++a)
          v(b, fields::closure_new(a), k, j, i) = m * c.stress[a] - ((a < 3) ? p : 0.0);
        for (int d = 0; d < 3; ++d)
          v(b, fields::closure_new(6 + d), k, j, i) = m * c.heat[d];
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Kinetics::ShiftClosure
//! \brief closure_old <- closure_new on the entire block (start of a kinetics step).
TaskStatus ShiftClosure(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  static auto desc = MakePackDescriptor<fields::closure_old, fields::closure_new>(
      pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  if (v.GetNBlocks() == 0) return TaskStatus::complete;
  auto space = RiotFlatLoop::GetIndexSpace(IndexDomain::entire, v.GetNBlocks(), 9, md);
  RiotFlatLoop::five_d(
      "Kinetics::ShiftClosure", space,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        v(b, fields::closure_old(n), k, j, i) = v(b, fields::closure_new(n), k, j, i);
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Kinetics::AddClosureFluxes
//! \brief Add the closure, (1 - w) old + w new, to the hydro momentum and energy fluxes.
//! Runs after Hydro::CalculateFluxes (which stores the face velocities) and before the
//! flux correction.
TaskStatus AddClosureFluxes(MeshData<Real> *md, const Real w) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  auto pm = md->GetParentPointer();
  static auto desc =
      MakePackDescriptor<ccbulk::momentum, ccbulk::total_material_energy,
                         ccbulk::face_velocity, fields::closure_old, fields::closure_new>(
          pm->resolved_packages.get(), {}, {parthenon::PDOpt::WithFluxes});
  auto v = desc.GetPack(md);
  if (v.GetNBlocks() == 0) return TaskStatus::complete;
  AddClosureFluxesDir<X1DIR>(md, v, w);
  if (pm->ndim > 1) AddClosureFluxesDir<X2DIR>(md, v, w);
  if (pm->ndim > 2) AddClosureFluxesDir<X3DIR>(md, v, w);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Kinetics::BeforeLoop
//! \brief Restart layout check, then the closure of the initial (or restarted) f, which
//! the first step shifts to closure_old. The ghosts of f are current here:
//! Mesh::Initialize exchanges boundaries after the problem generator and after reading a
//! restart.
void BeforeLoop(Mesh *pm, ParameterInput *pin, parthenon::SimTime &tm) {
  CheckRestartLayout(pm, pin, tm);
  ExchangeFGhosts(pm);
  if (!pm->packages.Get(pkg_name)->Param<bool>("closure_coupling")) return;
  for (int i = 0; i < pm->DefaultNumPartitions(); ++i)
    ComputeClosure(pm->mesh_data.GetOrAdd("base", i).get());
}

} // namespace Kinetics
