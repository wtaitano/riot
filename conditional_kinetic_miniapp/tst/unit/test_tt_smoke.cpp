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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "tt_smoke.hpp"

using Catch::Approx;

TEST_CASE("Tensor-train library is linked and fills cores on device", "[ckm][tt]") {
  GIVEN("A closed three-core train with dims {4, 5, 6} and ranks {2, 3}") {
    const std::vector<int> dims{4, 5, 6};
    const std::vector<int> ranks{2, 3};
    // Core shapes are 1x4x2, 2x5x3 and 3x6x1.
    const double nentries = 1 * 4 * 2 + 2 * 5 * 3 + 3 * 6 * 1;

    WHEN("Every core entry is set to 0.5") {
      const double total = ckm::TTSmoke(dims, ranks, 0.5);
      THEN("The entries sum to 0.5 times the entry count") {
        REQUIRE(total == Approx(0.5 * nentries));
      }
    }
  }
}
