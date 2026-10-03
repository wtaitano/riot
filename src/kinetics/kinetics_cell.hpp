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
#ifndef KINETICS_KINETICS_CELL_HPP_
#define KINETICS_KINETICS_CELL_HPP_
// This file was made in part with generative AI.

// Read access to f of one spatial cell, the same for both representations, so that the
// diagnostics and the initialization are written once:
//
//   cell.Raw(), cell.Central(u)   moments (exact core contractions for TT),
//   cell.ForEach(func)            func(ix, iy, iz, value) on every node (decompression
//                                 for TT: the `exact` nonlinear-diagnostics path),
//   cell.Fill(eqs, count)         f = sum of count separable equilibria (exact in TT).
//
// A pack holding both kinetics.f and kinetics.f_tt can be used: only the field of the
// active representation is registered, and only the matching cell type touches it.

#include <parthenon/package.hpp>

#include "kinetics/equilibrium.hpp"
#include "kinetics/kinetics.hpp"
#include "kinetics/moments.hpp"
#include "kinetics/tt_moments.hpp"
#include "kinetics/tt_tensor.hpp"
#include "kinetics/velocity_grid.hpp"

namespace Kinetics {

template <class Pack>
struct DenseCell {
  const Pack &v;
  int b, k, j, i;
  VelocityGrid grid;

  KOKKOS_INLINE_FUNCTION Real &F(const int n) const {
    return v(b, fields::f(n), k, j, i);
  }
  KOKKOS_INLINE_FUNCTION RawMoments Raw() const {
    return ComputeRawMoments(grid, [&](const int n) { return F(n); });
  }
  KOKKOS_INLINE_FUNCTION CentralMoments Central(const Real u[3]) const {
    return ComputeCentralMoments(grid, [&](const int n) { return F(n); }, u);
  }
  template <class Func>
  KOKKOS_INLINE_FUNCTION void ForEach(const Func &func) const {
    for (int iz = 0; iz < grid.nv[2]; ++iz)
      for (int iy = 0; iy < grid.nv[1]; ++iy)
        for (int ix = 0; ix < grid.nv[0]; ++ix)
          func(ix, iy, iz, F(grid.Flat(ix, iy, iz)));
  }
  KOKKOS_INLINE_FUNCTION void Fill(const Maxwellian *eqs, const int count) const {
    for (int iz = 0; iz < grid.nv[2]; ++iz)
      for (int iy = 0; iy < grid.nv[1]; ++iy)
        for (int ix = 0; ix < grid.nv[0]; ++ix) {
          Real val = 0.0;
          for (int c = 0; c < count; ++c)
            val += eqs[c](grid, ix, iy, iz);
          F(grid.Flat(ix, iy, iz)) = val;
        }
  }
};

template <class Pack>
struct TTCell {
  using Data = TT::PackCell<Pack, fields::f_tt>;
  const Pack &v;
  int b, k, j, i;
  VelocityGrid grid;
  TT::TTLayout L;

  KOKKOS_INLINE_FUNCTION Data D() const { return Data{v, b, k, j, i}; }
  KOKKOS_INLINE_FUNCTION TT::TTRef<Data> Ref() const { return TT::MakeRef(D(), L); }
  KOKKOS_INLINE_FUNCTION RawMoments Raw() const {
    return TT::ComputeRawMoments(grid, Ref());
  }
  KOKKOS_INLINE_FUNCTION CentralMoments Central(const Real u[3]) const {
    return TT::ComputeCentralMoments(grid, Ref(), u);
  }
  template <class Func>
  KOKKOS_INLINE_FUNCTION void ForEach(const Func &func) const {
    TT::ForEachNode(Ref(), func);
  }
  KOKKOS_INLINE_FUNCTION void Fill(const Maxwellian *eqs, const int count) const {
    TT::FillMaxwellians(grid, eqs, count, TT::MakeOutRef(D(), L));
  }
};

// Representation of f, and the TT layout (all zero for dense), as kernels need them.
struct CellKind {
  bool tt;
  VelocityGrid grid;
  TT::TTLayout L;
};

inline CellKind GetCellKind(const StateDescriptor *pkg) {
  CellKind c{GetRepresentation(pkg) == Representation::tt,
             pkg->Param<VelocityGrid>("grid"), TT::TTLayout{{0, 0, 0}, 0}};
  if (c.tt) c.L = pkg->Param<TT::TTLayout>("tt_layout");
  return c;
}

// Call body(cell) with the cell of the active representation. body is a generic lambda;
// both instantiations are compiled, one runs.
template <class Pack, class Body>
KOKKOS_INLINE_FUNCTION void WithCell(const CellKind &kind, const Pack &v, const int b,
                                     const int k, const int j, const int i,
                                     const Body &body) {
  if (kind.tt) {
    body(TTCell<Pack>{v, b, k, j, i, kind.grid, kind.L});
  } else {
    body(DenseCell<Pack>{v, b, k, j, i, kind.grid});
  }
}

} // namespace Kinetics

#endif // KINETICS_KINETICS_CELL_HPP_
