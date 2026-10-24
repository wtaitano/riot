//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//
// This file was made in part with generative AI.
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

#include <vector>

#include <Kokkos_Core.hpp>

#include <tensors/tt_operations.hpp>
#include <tensors/tt_pack.hpp>

#include "tt_smoke.hpp"

namespace ckm {

using parthenon::Real;
using parthenon::tensor::TensorPack;
using parthenon::tensor::TensorTrain;

void FillTrain(TensorTrain &train, Real value) {
  std::vector<TensorTrain *> trains{&train};
  parthenon::tensor::TensorTrainHostPack host_pack(trains);
  auto pack = host_pack.MakeDevicePack();
  parthenon::tensor::SetTTPackToValue(pack, value);
  Kokkos::fence();
}

Real SumTrainEntries(const TensorTrain &train) {
  Real total = 0;
  for (int c = 0; c < train.NCores(); ++c) {
    const auto &core = train(c);
    for (int l = 0; l < core.LR(); ++l) {
      for (int r = 0; r < core.RR(); ++r) {
        auto fiber_h =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), core.GetFiber(l, r));
        for (int d = 0; d < core.DD(); ++d) {
          total += fiber_h(d);
        }
      }
    }
  }
  return total;
}

Real TTSmoke(const std::vector<int> &phys_dims, const std::vector<int> &ranks,
             Real value) {
  TensorTrain train(phys_dims, ranks);
  FillTrain(train, value);
  return SumTrainEntries(train);
}

} // namespace ckm
