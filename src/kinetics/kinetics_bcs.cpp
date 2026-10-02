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

#include <array>
#include <cmath>
#include <string>

#include <bvals/boundary_conditions_generic.hpp>
#include <loop_abstraction/loop_abstraction.hpp>

#include "kinetics/equilibrium.hpp"
#include "kinetics/kinetics.hpp"
#include "kinetics/kinetics_bcs.hpp"
#include "kinetics/velocity_grid.hpp"

namespace Kinetics {

namespace {

using parthenon::CoordinateDirection;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;
using Side = parthenon::BoundaryFunction::BCSide;

// Wall state of a diffuse face: separable equilibrium with unit density, normalized
// later per ghost cell by the flux balance.
struct WallState {
  Maxwellian eq;     // discrete equilibrium with n = 1 at (T_w, u_w)
  Real influx_per_n; // incoming number flux of eq through the wall, per unit n_w
};

template <CoordinateDirection DIR, Side SIDE>
constexpr IndexDomain GhostDomain() {
  constexpr bool inner = (SIDE == Side::Inner);
  if constexpr (DIR == X1DIR) {
    return inner ? IndexDomain::inner_x1 : IndexDomain::outer_x1;
  } else if constexpr (DIR == X2DIR) {
    return inner ? IndexDomain::inner_x2 : IndexDomain::outer_x2;
  } else {
    return inner ? IndexDomain::inner_x3 : IndexDomain::outer_x3;
  }
}

template <CoordinateDirection DIR>
IndexRange NormalInterior(const parthenon::IndexShape &bounds) {
  if constexpr (DIR == X1DIR) {
    return bounds.GetBoundsI(IndexDomain::interior);
  } else if constexpr (DIR == X2DIR) {
    return bounds.GetBoundsJ(IndexDomain::interior);
  } else {
    return bounds.GetBoundsK(IndexDomain::interior);
  }
}

// Replace the wall-normal index of (k, j, i) by n.
template <CoordinateDirection DIR>
KOKKOS_FORCEINLINE_FUNCTION void WithNormal(const int k, const int j, const int i,
                                            const int n, int &kr, int &jr, int &ir) {
  kr = (DIR == X3DIR) ? n : k;
  jr = (DIR == X2DIR) ? n : j;
  ir = (DIR == X1DIR) ? n : i;
}

template <CoordinateDirection DIR>
KOKKOS_FORCEINLINE_FUNCTION int NormalIndex(const int k, const int j, const int i) {
  return (DIR == X3DIR) ? k : ((DIR == X2DIR) ? j : i);
}

template <CoordinateDirection DIR, Side SIDE, KineticBC TYPE>
void KineticBCImpl(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  auto pmb = mbd->GetBlockPointer();
  constexpr bool inner = (SIDE == Side::Inner);
  constexpr int d = static_cast<int>(DIR) - 1; // velocity axis normal to the wall
  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto range = NormalInterior<DIR>(bounds);
  const int ref = inner ? range.s : range.e;
  // Mirror of ghost index g through the wall: ref - (g - ref) +- 1.
  const int mirror_sum = 2 * ref + (inner ? -1 : 1);

  auto pkg = pmb->packages.Get(pkg_name);
  const auto grid = pkg->Param<VelocityGrid>("grid");
  const int nv = grid.Size();
  WallState wall{};
  if constexpr (TYPE == KineticBC::diffuse) {
    const int face = 2 * d + (inner ? 0 : 1);
    wall = pkg->Param<std::array<WallState, 6>>("bc_wall_states")[face];
  }

  std::set<parthenon::PDOpt> opts;
  if (coarse) opts.insert(parthenon::PDOpt::Coarse);
  auto desc = MakePackDescriptor<fields::f>(mbd.get(), {Metadata::FillGhost}, opts);
  auto v = desc.GetPack(mbd.get());
  if (v.GetMaxNumberOfVars() == 0) return;

  constexpr IndexDomain domain = GhostDomain<DIR, SIDE>();
  pmb->par_for_bndry(
      "Kinetics::BC", IndexRange{0, 0}, domain, parthenon::TopologicalElement::CC, coarse,
      false, KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
        // nvcc fixes the capture set before resolving if constexpr; force the captures.
        parthenon::loop_abstraction::impl::ForceCapture(v, grid, nv, mirror_sum, wall);
        int kr, jr, ir;
        WithNormal<DIR>(k, j, i, ref, kr, jr, ir);
        if constexpr (TYPE == KineticBC::outflow) {
          for (int n = 0; n < nv; ++n)
            v(0, fields::f(n), k, j, i) = v(0, fields::f(n), kr, jr, ir);
        } else if constexpr (TYPE == KineticBC::specular) {
          int km, jm, im;
          WithNormal<DIR>(k, j, i, mirror_sum - NormalIndex<DIR>(k, j, i), km, jm, im);
          for (int n = 0; n < nv; ++n) {
            int idx[3];
            grid.Unflatten(n, idx[0], idx[1], idx[2]);
            idx[d] = grid.nv[d] - 1 - idx[d]; // reflected node, v_d -> -v_d
            v(0, fields::f(n), k, j, i) =
                v(0, fields::f(grid.Flat(idx[0], idx[1], idx[2])), km, jm, im);
          }
        } else { // diffuse
          // Outgoing number flux of the boundary cell through the wall (velocity
          // component pointing out of the domain), per unit area.
          Real outflux = 0.0;
          for (int n = 0; n < nv; ++n) {
            int idx[3];
            grid.Unflatten(n, idx[0], idx[1], idx[2]);
            const Real vn = grid.Node(d, idx[d]);
            // Nodes at |v_n| < dv/4 (only the zero node of an odd grid) carry no flux and
            // are treated as leaving, i.e. copied from the boundary cell.
            const bool leaving =
                inner ? (vn < 0.25 * grid.dv[d]) : (vn > -0.25 * grid.dv[d]);
            if (leaving) outflux += std::abs(vn) * v(0, fields::f(n), kr, jr, ir);
          }
          outflux *= grid.Weight();
          const Real n_w = (wall.influx_per_n > 0.0) ? outflux / wall.influx_per_n : 0.0;
          for (int n = 0; n < nv; ++n) {
            int idx[3];
            grid.Unflatten(n, idx[0], idx[1], idx[2]);
            const Real vn = grid.Node(d, idx[d]);
            const bool entering =
                inner ? (vn > 0.25 * grid.dv[d]) : (vn < -0.25 * grid.dv[d]);
            v(0, fields::f(n), k, j, i) =
                entering ? n_w * wall.eq(grid, idx[0], idx[1], idx[2])
                         : v(0, fields::f(n), kr, jr, ir);
          }
        }
      });
}

template <CoordinateDirection DIR, Side SIDE>
parthenon::BValFunc MakeBC(const KineticBC type) {
  switch (type) {
  case KineticBC::outflow:
    return KineticBCImpl<DIR, SIDE, KineticBC::outflow>;
  case KineticBC::specular:
    return KineticBCImpl<DIR, SIDE, KineticBC::specular>;
  case KineticBC::diffuse:
    return KineticBCImpl<DIR, SIDE, KineticBC::diffuse>;
  default:
    PARTHENON_FAIL("kinetics: no boundary function for a periodic face");
  }
  return nullptr;
}

parthenon::BValFunc MakeBC(const int face, const KineticBC type) {
  switch (face) {
  case 0:
    return MakeBC<X1DIR, Side::Inner>(type);
  case 1:
    return MakeBC<X1DIR, Side::Outer>(type);
  case 2:
    return MakeBC<X2DIR, Side::Inner>(type);
  case 3:
    return MakeBC<X2DIR, Side::Outer>(type);
  case 4:
    return MakeBC<X3DIR, Side::Inner>(type);
  default:
    return MakeBC<X3DIR, Side::Outer>(type);
  }
}

// Discrete equilibrium (n = 1) at the wall state, and its incoming flux through the
// face. Host side: the solve is KOKKOS_INLINE_FUNCTION and runs fine on the host.
WallState MakeWallState(const VelocityGrid &grid, const EquilibriumParams &eq_params,
                        const Real theta_w, const Real u_w[3], const int d,
                        const bool inner) {
  WallState w;
  EquilibriumTarget t{1.0, {u_w[0], u_w[1], u_w[2]}, {theta_w, theta_w, theta_w}};
  const auto res = SolveEquilibrium(grid, t, eq_params, w.eq);
  PARTHENON_REQUIRE(res.status == EquilibriumResult::Status::converged,
                    "kinetics: wall equilibrium not resolved by the velocity grid");
  w.influx_per_n = 0.0;
  for (int n = 0; n < grid.Size(); ++n) {
    int idx[3];
    grid.Unflatten(n, idx[0], idx[1], idx[2]);
    const Real vn = grid.Node(d, idx[d]);
    const bool entering = inner ? (vn > 0.25 * grid.dv[d]) : (vn < -0.25 * grid.dv[d]);
    if (entering) w.influx_per_n += std::abs(vn) * w.eq(grid, idx[0], idx[1], idx[2]);
  }
  w.influx_per_n *= grid.Weight();
  return w;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn  void Kinetics::EnrollKineticBCs
void EnrollKineticBCs(StateDescriptor *pkg, ParameterInput *pin) {
  const std::array<std::string, 6> faces = {"ix1", "ox1", "ix2", "ox2", "ix3", "ox3"};
  const auto grid = pkg->Param<VelocityGrid>("grid");
  const auto eq_params = pkg->Param<EquilibriumParams>("eq_params");
  const Real kb_per_m = pkg->Param<Species>("species").kb_per_m;
  // Same rule as Parthenon's Mesh (mesh.cpp): nx3 > 1 makes the mesh 3D.
  const int nx2 = pin->GetInteger("parthenon/mesh", "nx2");
  const int nx3 = pin->GetInteger("parthenon/mesh", "nx3");
  const int ndim = (nx3 > 1) ? 3 : ((nx2 > 1) ? 2 : 1);

  std::array<WallState, 6> walls{};
  for (int f = 0; f < 6; ++f) {
    const int d = f / 2;
    const bool inner = (f % 2 == 0);
    const std::string mesh_bc = pin->GetString("parthenon/mesh", faces[f] + "_bc");
    const std::string def = (mesh_bc == "periodic") ? "periodic" : "outflow";
    const std::string name =
        pin->GetOrAddString(input_block, faces[f] + "_bc", def,
                            {"periodic", "outflow", "specular", "diffuse"},
                            "Kinetic boundary condition of this face");
    PARTHENON_REQUIRE((name == "periodic") == (mesh_bc == "periodic"),
                      "kinetics/" + faces[f] +
                          "_bc must be periodic exactly when the mesh face is");
    if (name == "periodic" || d >= ndim) continue;

    KineticBC type = KineticBC::outflow;
    if (name == "specular") {
      type = KineticBC::specular;
      const Real lo = grid.vmin[d], hi = grid.vmin[d] + grid.nv[d] * grid.dv[d];
      PARTHENON_REQUIRE(std::abs(lo + hi) <= 1.0e-12 * (hi - lo),
                        "kinetics: a specular wall needs a velocity box symmetric "
                        "about 0 along the wall-normal axis");
    } else if (name == "diffuse") {
      type = KineticBC::diffuse;
      const Real T_w = pin->GetReal(input_block, faces[f] + "_wall_T");
      PARTHENON_REQUIRE(T_w > 0.0, "kinetics: wall temperature must be positive");
      Real u_w[3];
      for (int a = 0; a < 3; ++a)
        u_w[a] =
            pin->GetOrAddReal(input_block, faces[f] + "_wall_u" + std::to_string(a + 1),
                              0.0, "Wall velocity component");
      PARTHENON_REQUIRE(u_w[d] == 0.0, "kinetics: a wall cannot move along its normal");
      walls[f] = MakeWallState(grid, eq_params, kb_per_m * T_w, u_w, d, inner);
    }
    pkg->UserBoundaryFunctions[f].push_back(MakeBC(f, type));
  }
  pkg->AddParam("bc_wall_states", walls);
}

} // namespace Kinetics
