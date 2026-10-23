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
#ifndef CKM_TT_SMOKE_HPP_
#define CKM_TT_SMOKE_HPP_

#include <vector>

#include <basic_types.hpp>
#include <tensors/tt_types.hpp>

namespace ckm {

//! \brief Fill every core of `train` with `value` on device.
void FillTrain(parthenon::tensor::TensorTrain &train, parthenon::Real value);

//! \brief Sum of every entry in every core of `train`, copied back to host.
parthenon::Real SumTrainEntries(const parthenon::tensor::TensorTrain &train);

//! \brief Build a closed train with the given physical dims and internal ranks,
//! fill it with `value`, and return the host-side sum of its core entries. The
//! expected result is value * sum_c (LR_c * DD_c * RR_c). Used as a link and
//! runtime check of the tensor-train library.
parthenon::Real TTSmoke(const std::vector<int> &phys_dims, const std::vector<int> &ranks,
                        parthenon::Real value);

} // namespace ckm

#endif // CKM_TT_SMOKE_HPP_
