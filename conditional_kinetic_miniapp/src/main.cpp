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

#include <cmath>
#include <iostream>
#include <memory>

#include <globals.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <tensors/tt_container.hpp>
#include <tensors/tt_field_metadata.hpp>

#include "tt_smoke.hpp"

using namespace parthenon::driver::prelude;
using parthenon::Metadata;
using parthenon::Packages_t;
using parthenon::Real;
using parthenon::StateDescriptor;
using parthenon::TTFieldMetadata;

namespace ckm {

constexpr char kPackageName[] = "ckm";
constexpr char kTTField[] = "f";

//----------------------------------------------------------------------------------------
//! \brief Skeleton package: one tensor-train field over (space, ckm/n1, ckm/n2).
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>(kPackageName);
  const int n1 = pin->GetOrAddInteger("ckm", "n1", 8);
  const int n2 = pin->GetOrAddInteger("ckm", "n2", 8);
  pkg->AddTTField(kTTField, TTFieldMetadata({n1, n2}, Metadata({Metadata::Cell,
                                                                Metadata::Independent})));
  return pkg;
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  packages.Add(Initialize(pin.get()));
  return packages;
}

//----------------------------------------------------------------------------------------
//! \brief Skeleton driver. No evolution: fills the TT field on every block and checks
//! the result, proving the tensor-train library is linked and runs on this mesh.
class CkmDriver : public Driver {
 public:
  CkmDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : Driver(pin, app_in, pm) {}

  DriverStatus Execute() override {
    PreExecute();
    const Real value = pinput->GetOrAddReal("ckm", "fill_value", 1.0);
    int nfail = 0;
    for (auto &pmb : pmesh->block_list) {
      auto &train = pmb->tt_block_data.Get()->Get(kTTField)->train();
      FillTrain(train, value);
      Real expected = 0;
      for (int c = 0; c < train.NCores(); ++c) {
        expected += value * train(c).LR() * train(c).DD() * train(c).RR();
      }
      const Real got = SumTrainEntries(train);
      if (std::abs(got - expected) > 1e-6 * std::abs(expected)) ++nfail;
    }
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(
        MPI_Allreduce(MPI_IN_PLACE, &nfail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
#endif
    if (parthenon::Globals::my_rank == 0) {
      std::cout << "ckm: tensor-train check on " << pmesh->nbtotal
                << " blocks: " << (nfail == 0 ? "PASS" : "FAIL") << std::endl;
    }
    const auto status = nfail == 0 ? DriverStatus::complete : DriverStatus::failed;
    PostExecute(status);
    return status;
  }
};

} // namespace ckm

//----------------------------------------------------------------------------------------
//! \fn  int main
int main(int argc, char *argv[]) {
  ParthenonManager pman;
  pman.app_input->ProcessPackages = ckm::ProcessPackages;

  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  pman.ParthenonInitPackagesAndMesh();
  DriverStatus driver_status;
  {
    // Scoped so the driver is destroyed before Finalize
    ckm::CkmDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());
    driver_status = driver.Execute();
  }
  pman.ParthenonFinalize();

  return driver_status == DriverStatus::complete ? 0 : 1;
}
