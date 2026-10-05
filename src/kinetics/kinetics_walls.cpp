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

// Hydro wall fluxes enslaved to the kinetic walls (claude_sessions/kinetic_bgk/
// S3_DESIGN.md, S3-Q9/Q14). With closure_coupling, a kinetic specular or diffuse wall
// sits on a hydro reflecting face (EnrollKineticBCs). Each semi-Lagrangian step of length
// h adds h times the kinetic moment flux of {m, m v, m |v|^2 / 2} through the wall faces
// to kinetics.wall_flux; hydro then uses wall_flux / dt as its mass, momentum and energy
// flux at those faces in every RK stage (the stage weights sum to dt), so the hydro wall
// exchange over a step equals the discrete kinetic one.
//
// Linear SL (sl_order = 1) with |s| <= 1 is first-order upwind in flux form. The
// tensor-product step equals the sweeps x1, x2, x3 applied in order, so the flux through
// a d-face is the upwind value of f after the sweeps e < d: the dense step interpolates
// f along e < d here, the tensor-train sweeps pass their intermediate register.
// wall_flux of a face lives in the ghost cell next to it (index s - 1 inner, e + 1 outer
// along d), so the faces of different directions never share a cell.

#include <array>

#include "kinetics/kinetics.hpp"
#include "kinetics/semi_lagrangian.hpp"
#include "kinetics/tt_tensor.hpp"
#include "kinetics/velocity_grid.hpp"
#include "riot_utils/riot_loops.hpp"
#include "variables.hpp"

namespace Kinetics {

namespace {

// Half-range moment sum_v v_d [sigma v_d > 0] v_a^p t(v) of a tensor-train cell (no
// quadrature weight), by one exact core contraction.
template <class Ref>
KOKKOS_INLINE_FUNCTION Real HalfMoment(const VelocityGrid &grid, const Ref &t,
                                       const int d, const Real sigma, const int a,
                                       const int p) {
  const auto axis = [=](const int e) {
    return [=](const int n) {
      const Real v = grid.Node(e, n);
      Real w = 1.0;
      if (e == d) w = (sigma * v > 0.0) ? v : 0.0;
      if (e == a) w *= (p == 1) ? v : ((p == 2) ? v * v : 1.0);
      return w;
    };
  };
  return TT::Contract(t, axis(0), axis(1), axis(2));
}

// Index ranges of the storage cells of the wall faces of side `inner` along d: the
// ghost layer next to the face along d, the interior along the other directions.
void WallCells(MeshData<Real> *md, const int d, const bool inner, IndexRange r[3]) {
  r[0] = md->GetBoundsI(IndexDomain::interior);
  r[1] = md->GetBoundsJ(IndexDomain::interior);
  r[2] = md->GetBoundsK(IndexDomain::interior);
  const int g = inner ? r[d].s - 1 : r[d].e + 1;
  r[d].s = g;
  r[d].e = g;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn  bool Kinetics::HasCoupledWalls
//! \brief Some face is a kinetic wall with closure_coupling (its hydro flux is enslaved).
bool HasCoupledWalls(const StateDescriptor *pkg) {
  const auto walls = pkg->Param<std::array<bool, 6>>("coupled_walls");
  for (const bool w : walls)
    if (w) return true;
  return false;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Kinetics::ResetWallFlux
//! \brief kinetics.wall_flux <- 0 (start of a kinetics step).
TaskStatus ResetWallFlux(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  static auto desc = MakePackDescriptor<fields::wall_flux>(pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  if (v.GetNBlocks() == 0) return TaskStatus::complete;
  auto space = RiotFlatLoop::GetIndexSpace(IndexDomain::entire, v.GetNBlocks(), 5, md);
  RiotFlatLoop::five_d(
      "Kinetics::ResetWallFlux", space,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        v(b, fields::wall_flux(n), k, j, i) = 0.0;
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Kinetics::AccumulateWallFlux
//! \brief wall_flux += h (m, m v, m |v|^2 / 2) fluxes of f in md through the coupled
//! wall faces normal to d. interp: dense f, interpolate the upwind cell along the
//! directions e < d (linear SL weights); otherwise f in md is already swept along them.
//! The ghosts of f in md along d must be current.
void AccumulateWallFlux(MeshData<Real> *md, const int d, const Real h,
                        const bool interp) {
  auto pm = md->GetParentPointer();
  auto pkg = pm->packages.Get(pkg_name);
  const auto walls = pkg->Param<std::array<bool, 6>>("coupled_walls");
  if (!walls[2 * d] && !walls[2 * d + 1]) return;
  static auto desc = MakePackDescriptor<fields::f, fields::f_tt, fields::wall_flux>(
      pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  if (v.GetNBlocks() == 0) return;
  const auto grid = pkg->Param<VelocityGrid>("grid");
  const Real m = pkg->Param<Species>("species").mass;
  const bool tt = GetRepresentation(pkg.get()) == Representation::tt;
  TT::TTLayout L{{0, 0, 0}, 0};
  if (tt) L = pkg->Param<TT::TTLayout>("tt_layout");
  const int di = (d == 0), dj = (d == 1), dk = (d == 2);
  const Real scale = h * m * grid.Weight();

  for (const bool inner : {true, false}) {
    if (!walls[2 * d + (inner ? 0 : 1)]) continue;
    IndexRange r[3];
    WallCells(md, d, inner, r);
    const int off = inner ? -1 : 1;
    // lower and upper cell of the face, as offsets from the storage cell along d
    const int lo = inner ? 0 : -1, up = inner ? 1 : 0;
    auto space = RiotFlatLoop::GetIndexSpace(v.GetNBlocks(), r[2], r[1], r[0]);
    RiotFlatLoop::four_d(
        "Kinetics::AccumulateWallFlux", space,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          if (!v.IsPhysicalBoundary(b, dk * off, dj * off, di * off)) return;
          Real F[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
          if (tt) {
            using PC = TT::PackCell<decltype(v), fields::f_tt>;
            for (int side = 0; side < 2; ++side) {
              const int c = side ? up : lo;
              const Real sigma = side ? -1.0 : 1.0; // nodes leaving that cell
              const auto t = TT::MakeRef(PC{v, b, k + dk * c, j + dj * c, i + di * c}, L);
              F[0] += HalfMoment(grid, t, d, sigma, 0, 0);
              for (int a = 0; a < 3; ++a) {
                F[1 + a] += HalfMoment(grid, t, d, sigma, a, 1);
                F[4] += 0.5 * HalfMoment(grid, t, d, sigma, a, 2);
              }
            }
          } else {
            const auto &coords = v.GetCoordinates(b);
            for (int n = 0; n < grid.Size(); ++n) {
              int idx[3];
              grid.Unflatten(n, idx[0], idx[1], idx[2]);
              const Real vel[3] = {grid.Node(0, idx[0]), grid.Node(1, idx[1]),
                                   grid.Node(2, idx[2])};
              if (vel[d] == 0.0) continue;
              const int c = (vel[d] > 0.0) ? lo : up;
              // Upwind cell, swept along the directions e < d (x1 offset p, x2 offset q)
              // with the linear SL weights; a single point when not interpolating.
              Real w[2][3] = {{0.0, 1.0, 0.0}, {0.0, 1.0, 0.0}};
              for (int e = 0; e < d && interp; ++e)
                SLWeights(1, vel[e] * h / coords.Dx(e + 1), w[e]);
              const int np = (interp && d > 0) ? 1 : 0, nq = (interp && d > 1) ? 1 : 0;
              Real fu = 0.0;
              for (int q = -nq; q <= nq; ++q)
                for (int p = -np; p <= np; ++p)
                  fu += w[0][p + 1] * w[1][q + 1] *
                        v(b, fields::f(n), k + dk * c, j + dj * c + q, i + di * c + p);
              const Real fl = vel[d] * fu;
              F[0] += fl;
              for (int a = 0; a < 3; ++a)
                F[1 + a] += vel[a] * fl;
              F[4] += 0.5 * (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]) * fl;
            }
          }
          for (int n = 0; n < 5; ++n)
            v(b, fields::wall_flux(n), k, j, i) += scale * F[n];
        });
  }
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Kinetics::ApplyWallFluxes
//! \brief At the coupled wall faces, replace the hydro mass, momentum and energy fluxes
//! (Riemann + closure) by the kinetic ones, wall_flux / dt. Runs after
//! AddClosureFluxes and before the flux correction.
TaskStatus ApplyWallFluxes(MeshData<Real> *md, const Real dt) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  auto pm = md->GetParentPointer();
  auto pkg = pm->packages.Get(pkg_name);
  if (!HasCoupledWalls(pkg.get())) return TaskStatus::complete;
  const auto walls = pkg->Param<std::array<bool, 6>>("coupled_walls");
  static auto desc = MakePackDescriptor<ccmat::rho, ccbulk::momentum,
                                        ccbulk::total_material_energy, fields::wall_flux>(
      pm->resolved_packages.get(), {}, {parthenon::PDOpt::WithFluxes});
  auto v = desc.GetPack(md);
  if (v.GetNBlocks() == 0) return TaskStatus::complete;
  const Real inv_dt = 1.0 / dt;
  for (int d = 0; d < pm->ndim; ++d) {
    for (const bool inner : {true, false}) {
      if (!walls[2 * d + (inner ? 0 : 1)]) continue;
      IndexRange r[3];
      WallCells(md, d, inner, r);
      const int di = (d == 0), dj = (d == 1), dk = (d == 2);
      const int off = inner ? -1 : 1;
      const int face = inner ? 1 : 0; // face index along d = storage index + face
      const int dir = d + 1;
      auto space = RiotFlatLoop::GetIndexSpace(v.GetNBlocks(), r[2], r[1], r[0]);
      RiotFlatLoop::four_d(
          "Kinetics::ApplyWallFluxes", space,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            if (!v.IsPhysicalBoundary(b, dk * off, dj * off, di * off)) return;
            const int kf = k + dk * face, jf = j + dj * face, if_ = i + di * face;
            v.flux(b, dir, ccmat::rho(0), kf, jf, if_) =
                inv_dt * v(b, fields::wall_flux(0), k, j, i);
            for (int a = 0; a < 3; ++a)
              v.flux(b, dir, ccbulk::momentum(a), kf, jf, if_) =
                  inv_dt * v(b, fields::wall_flux(1 + a), k, j, i);
            v.flux(b, dir, ccbulk::total_material_energy(), kf, jf, if_) =
                inv_dt * v(b, fields::wall_flux(4), k, j, i);
          });
    }
  }
  return TaskStatus::complete;
}

} // namespace Kinetics
