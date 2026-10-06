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

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <utils/constants.hpp>

#include "kinetics/bgk.hpp"
#include "kinetics/kinetics.hpp"
#include "kinetics/kinetics_bcs.hpp"
#include "kinetics/semi_lagrangian.hpp"
#include "kinetics/tt_cross.hpp"
#include "kinetics/tt_tensor.hpp"
#include "materials/materials.hpp"

namespace Kinetics {

namespace {

// Require the single hydro material to be a monatomic ideal gas and derive the
// particle mass from it: p = (Gamma - 1) rho Cv T = n k_B T gives k_B / m = (Gamma-1) Cv.
Species GasFromMaterial(ParameterInput *pin) {
  using pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;
  PARTHENON_REQUIRE(Materials::CountMaterials(pin) == 1,
                    "kinetics: exactly one material (material0) is required");
  const std::string mat = "material0";
  PARTHENON_REQUIRE(pin->GetOrAddInteger(mat, "nphase", 1) == 1,
                    "kinetics: the material must have a single phase");
  const std::string eos_block =
      pin->DoesParameterExist(mat, "eos") ? pin->GetString(mat, "eos") : mat;
  PARTHENON_REQUIRE(pin->GetString(eos_block, "eos_type") == "IdealGas",
                    "kinetics: the material EOS must be IdealGas");
  const Real gamma = pin->GetReal(eos_block, "Gamma");
  const Real cv = pin->GetReal(eos_block, "Cv");
  PARTHENON_REQUIRE(std::abs(gamma - 5.0 / 3.0) < 1.0e-12,
                    "kinetics: a monatomic gas needs Gamma = 5/3");
  PARTHENON_REQUIRE(cv > 0.0, "kinetics: Cv must be positive");
  Species s;
  s.kb_per_m = (gamma - 1.0) * cv;
  s.mass = pc::kb / s.kb_per_m;
  return s;
}

// Refinement operator that does nothing: Parthenon must not combine tensor-train slots
// (kinetics_amr.cpp fills the coarse buffer and, for linear prolongation, the fine
// ghosts instead).
struct RefinementNoOp {
  static constexpr bool OperationRequired(parthenon::TopologicalElement,
                                          parthenon::TopologicalElement) {
    return false;
  }
  template <int DIM, parthenon::TopologicalElement EL = parthenon::TopologicalElement::CC,
            parthenon::TopologicalElement CEL = parthenon::TopologicalElement::CC,
            class... Args>
  KOKKOS_FORCEINLINE_FUNCTION static void Do(Args &&...) {}
};

} // namespace

//----------------------------------------------------------------------------------------
//! \fn  std::shared_ptr<StateDescriptor> Kinetics::Initialize
//! \brief Kinetics package: parameters, velocity grid and fields.
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>(pkg_name);
  Params &params = pkg->AllParams();

  const std::string representation =
      pin->GetOrAddString(input_block, "representation", "dense", {"dense", "tt"},
                          "Velocity-space representation of f: dense or tensor train");
  params.Add("representation", representation);
  const bool tt = (representation == "tt");

  // Gas
  const Species species = GasFromMaterial(pin);
  params.Add("species", species);

  // Velocity grid (CGS, cm/s)
  int nv[3];
  Real vmin[3], vmax[3];
  for (int d = 0; d < 3; ++d) {
    const std::string ax = std::to_string(d + 1);
    nv[d] = pin->GetInteger(input_block, "nv" + ax);
    vmin[d] = pin->GetReal(input_block, "v" + ax + "min");
    vmax[d] = pin->GetReal(input_block, "v" + ax + "max");
  }
  const VelocityGrid grid = MakeVelocityGrid(nv, vmin, vmax);
  params.Add("grid", grid);
  // Storage layout of f: {representation (0 dense, 1 tt), nv1, nv2, nv3, tt_rank_max}.
  // "f_layout" is this run's; "f_layout_file" is overwritten from the restart file on a
  // restart (Mutability::Restart) and checked against it before f is used.
  const int rcap_in = tt ? pin->GetOrAddInteger(input_block, "tt_rank_max", 16,
                                                "Largest TT rank (sets the storage per "
                                                "cell)")
                         : 0;
  const std::vector<int> layout = {tt ? 1 : 0, nv[0], nv[1], nv[2], rcap_in};
  params.Add("f_layout", layout);
  // On a restart, start from {-1} so that a file without the parameter (written before
  // it existed) fails the check instead of passing silently.
  params.Add("f_layout_file", Globals::is_restart ? std::vector<int>{-1} : layout,
             Params::Mutability::Restart);

  // Tensor-train representation (S1_DESIGN.md)
  if (tt) {
    TT::RoundParams round;
    round.eps = pin->GetOrAddReal(input_block, "tt_eps", 1.0e-8,
                                  "Relative TT rounding tolerance per rounding; 0 = "
                                  "fixed rank tt_rank_max");
    PARTHENON_REQUIRE(round.eps >= 0.0 && round.eps < 1.0,
                      "kinetics: tt_eps must be in [0, 1)");
    params.Add("tt_round", round);
    const int rcap = rcap_in;
    PARTHENON_REQUIRE(rcap >= 2 && rcap <= TT::kMaxRank,
                      "kinetics: tt_rank_max must be in [2, " +
                          std::to_string(TT::kMaxRank) + "]");
    params.Add("tt_layout", TT::MakeLayout(grid, rcap));
    const std::string diag =
        pin->GetOrAddString(input_block, "tt_diag", "cross", {"exact", "cross"},
                            "Nonlinear diagnostics of a TT f: exact (decompress) or "
                            "cross (DEIM cross approximation, estimates)");
    TT::CrossParams cross;
    cross.eps = pin->GetOrAddReal(input_block, "tt_cross_eps", cross.eps,
                                  "Cross: relative max residual on the trial set");
    cross.rank_max = pin->GetOrAddInteger(input_block, "tt_cross_rank_max",
                                          cross.rank_max, "Cross: cap on the ranks of g");
    cross.trial_factor =
        pin->GetOrAddInteger(input_block, "tt_cross_trial_factor", cross.trial_factor,
                             "Cross: trial nodes = factor * max TT rank * max nv");
    const Real cross_delta = pin->GetOrAddReal(
        input_block, "tt_cross_delta", 1.0e-12,
        "Cross entropy: regularization delta relative to max |f| (sampled)");
    PARTHENON_REQUIRE(cross.eps > 0.0 && cross.eps < 1.0,
                      "kinetics: tt_cross_eps must be in (0, 1)");
    PARTHENON_REQUIRE(cross.rank_max >= 1 && cross.rank_max <= TT::kMaxRank,
                      "kinetics: tt_cross_rank_max must be in [1, " +
                          std::to_string(TT::kMaxRank) + "]");
    PARTHENON_REQUIRE(cross.trial_factor >= 1 && cross.trial_factor <= 64,
                      "kinetics: tt_cross_trial_factor must be in [1, 64]");
    PARTHENON_REQUIRE(cross_delta > 0.0 && cross_delta < 1.0,
                      "kinetics: tt_cross_delta must be in (0, 1)");
    params.Add("tt_cross", cross);
    params.Add("tt_cross_delta", cross_delta);
    params.Add("tt_diag", diag);
  }

  // Mesh refinement (S4_DESIGN.md)
  const std::string refinement =
      pin->GetOrAddString("parthenon/mesh", "refinement", "none");
  const bool multilevel = (refinement != "none");
  const std::string amr_prolong = pin->GetOrAddString(
      input_block, "amr_prolong", "constant", {"constant", "linear"},
      "Prolongation of f to fine ghosts and new fine cells: copy of the parent "
      "(constant) or unlimited linear slopes (linear)");
  params.Add("amr_prolong", amr_prolong);
  // Adaptive refinement: kinetic criterion and remesh bookkeeping (kinetics_amr.cpp).
  const bool adaptive = (refinement == "adaptive");
  params.Add("amr_noneq_refine",
             pin->GetOrAddReal(input_block, "amr_noneq_refine", 1.0e-2,
                               "Refine a block where ||f - M[f]|| / ||f|| exceeds this"));
  params.Add("amr_noneq_derefine",
             pin->GetOrAddReal(input_block, "amr_noneq_derefine", 1.0e-3,
                               "Derefine a block where ||f - M[f]|| / ||f|| is below "
                               "this everywhere"));
  PARTHENON_REQUIRE(params.Get<Real>("amr_noneq_derefine") <=
                        params.Get<Real>("amr_noneq_refine"),
                    "kinetics: amr_noneq_derefine must not exceed amr_noneq_refine");
  params.Add("amr_derefine_count",
             pin->GetOrAddInteger("parthenon/mesh", "derefine_count", 10));
  params.Add("amr_started", false, Params::Mutability::Mutable);
  params.Add("amr_locs", std::vector<parthenon::LogicalLocation>{},
             Params::Mutability::Mutable);
  PARTHENON_REQUIRE(!multilevel || parthenon::IsCoord<parthenon::UniformCartesian>(),
                    "kinetics: mesh refinement needs Cartesian coordinates");

  // Discrete equilibrium solve
  EquilibriumParams eq;
  eq.tol = pin->GetOrAddReal(input_block, "eq_tol", 1.0e-13,
                             "Relative moment tolerance of the equilibrium solve");
  eq.max_iter = pin->GetOrAddInteger(input_block, "eq_max_iter", 20,
                                     "Newton iteration limit of the equilibrium solve");
  params.Add("eq_params", eq);
  params.Add("eq_fallback_abort",
             pin->GetOrAddReal(input_block, "eq_fallback_abort", 1.0e-3,
                               "Abort if a larger fraction of cells falls back to the "
                               "sampled Maxwellian"));

  // Velocity-space resolution checks
  params.Add("edge_mass_warn",
             pin->GetOrAddReal(input_block, "edge_mass_warn", 1.0e-10,
                               "Warn if a cell has more of its mass on the outermost "
                               "node layer of the velocity box"));
  params.Add("edge_mass_abort",
             pin->GetOrAddReal(input_block, "edge_mass_abort", 1.0e-6,
                               "Abort if a cell has more of its mass on the outermost "
                               "node layer of the velocity box"));
  params.Add("min_vth_over_dv",
             pin->GetOrAddReal(input_block, "min_vth_over_dv", 1.5,
                               "Warn if the thermal speed is resolved by fewer nodes"));
  params.Add("check_every",
             pin->GetOrAddInteger(input_block, "check_every", 100,
                                  "Cycles between velocity-resolution checks"));

  // Collision frequency
  using pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;
  CollisionModel model;
  const std::string nu_model =
      pin->GetOrAddString(input_block, "nu_model", "constant", {"constant", "power_law"},
                          "BGK collision frequency model");
  model.type = (nu_model == "constant") ? CollisionModel::Type::constant
                                        : CollisionModel::Type::power_law;
  model.kb = pc::kb;
  model.nu0 = 0.0;
  model.mu_ref = 1.0;
  model.T_ref = 1.0;
  model.omega = 0.5;
  if (model.type == CollisionModel::Type::constant) {
    model.nu0 = pin->GetReal(input_block, "nu0", "Constant collision frequency [1/s]");
    PARTHENON_REQUIRE(model.nu0 >= 0.0, "kinetics: nu0 must be non-negative");
  } else {
    model.mu_ref = pin->GetReal(input_block, "mu_ref", "Viscosity at T_ref [g/(cm s)]");
    model.T_ref = pin->GetReal(input_block, "T_ref", "Reference temperature [K]");
    model.omega =
        pin->GetOrAddReal(input_block, "omega", 0.5, "Viscosity exponent, mu ~ T^omega");
    PARTHENON_REQUIRE(model.mu_ref > 0.0 && model.T_ref > 0.0,
                      "kinetics: mu_ref and T_ref must be positive");
  }
  params.Add("collision_model", model);

  // Streaming
  SLParams sl;
  sl.order = pin->GetOrAddInteger(input_block, "sl_order", 1,
                                  "Semi-Lagrangian interpolation order (1 or 2)");
  PARTHENON_REQUIRE(sl.order == 1 || sl.order == 2, "kinetics: sl_order must be 1 or 2");
  const std::string limiter = pin->GetOrAddString(
      input_block, "sl_limiter", "minmax", {"minmax", "none"},
      "Clip interpolated values to the bracketing cells (minmax) or not (none)");
  sl.limiter = (limiter == "minmax") && (sl.order == 2); // linear SL is monotone
  params.Add("sl_params", sl);
  if (multilevel && sl.order == 2 && amr_prolong == "constant" && Globals::my_rank == 0)
    std::cout << "kinetics: WARNING: sl_order = 2 with amr_prolong = constant is first "
                 "order at fine-coarse interfaces; use amr_prolong = linear"
              << std::endl;
  const Real cfl = pin->GetOrAddReal(input_block, "cfl", 1.0,
                                     "Max cells moved per kinetic substep (<= 1)");
  PARTHENON_REQUIRE(cfl > 0.0 && cfl <= 1.0, "kinetics: cfl must be in (0, 1]");
  PARTHENON_REQUIRE(!(tt && sl.limiter),
                    "kinetics: representation = tt needs sl_limiter = none (a pointwise "
                    "limiter has no tensor-train form) or sl_order = 1");
  params.Add("cfl", cfl);
  const std::string integrator = pin->GetOrAddString(
      input_block, "integrator", "sl_dirk2", {"sl_dirk2", "strang"},
      "Streaming + collision integrator: characteristic IMEX-RK (sl_dirk2) or Strang");
  params.Add("integrator",
             (integrator == "strang") ? Integrator::strang : Integrator::sl_dirk2);
  const bool streaming =
      pin->GetOrAddBoolean(input_block, "streaming", true,
                           "Stream f in space; false keeps only the collisions (0D "
                           "relaxation tests), with the same substeps");
  params.Add("streaming", streaming);
  params.Add("merge_half_steps",
             pin->GetOrAddBoolean(input_block, "merge_half_steps", true,
                                  "strang: merge adjacent half steps of substeps"));
  params.Add("substeps", 0, Params::Mutability::Mutable);

  // Closure coupling to hydro (CLOSURE_DESIGN.md)
  const bool closure = ClosureCoupling(pin);
  params.Add("closure_coupling", closure);
  if (closure) {
    PARTHENON_REQUIRE(parthenon::IsCoord<parthenon::UniformCartesian>(),
                      "kinetics: closure_coupling needs Cartesian coordinates");
    // closure_old carries over from the previous step; it is recomputed from f after a
    // remesh (AfterRemesh).
  }
  // LoMaC: kinetic moments enslaved to hydro (S3_DESIGN.md, S3-Q8/Q16)
  const bool lomac = Lomac(pin);
  params.Add("lomac", lomac);
  PARTHENON_REQUIRE(!lomac || closure,
                    "kinetics: lomac = true needs closure_coupling = true");
  PARTHENON_REQUIRE(!lomac || pin->GetOrAddBoolean("physics", "hydro", true),
                    "kinetics: lomac = true needs hydro");
  params.Add("lomac_skip_abort",
             pin->GetOrAddReal(input_block, "lomac_skip_abort", 1.0e-3,
                               "Abort if a larger fraction of cells skips the LoMaC "
                               "correction"));
  if (lomac && tt)
    PARTHENON_REQUIRE(rcap_in >= 3,
                      "kinetics: lomac with representation = tt needs tt_rank_max >= 3");
  PARTHENON_REQUIRE(Globals::nghost >= 1, "kinetics: needs at least one ghost cell");

  // Initialization from the hydro state
  const std::string init = pin->GetOrAddString(
      input_block, "init", "equilibrium",
      {"equilibrium", "bimaxwellian", "two_maxwellian"}, "Initial distribution");
  params.Add("init", init);
  const Real init_drift = pin->GetOrAddReal(
      input_block, "init_drift", 1.0,
      "two_maxwellian: drift of each half along init_axis, in units of sqrt(k_B T / m)");
  PARTHENON_REQUIRE(init_drift >= 0.0 && init_drift * init_drift < 3.0,
                    "kinetics: init_drift must be in [0, sqrt(3)) so that the two "
                    "Maxwellians keep a positive temperature");
  params.Add("init_drift", init_drift);
  const Real init_T_ratio = pin->GetOrAddReal(
      input_block, "init_T_ratio", 1.0, "bimaxwellian: T_parallel / T_perpendicular");
  const int init_axis = pin->GetOrAddInteger(input_block, "init_axis", 1,
                                             "bimaxwellian: parallel axis (1, 2 or 3)");
  PARTHENON_REQUIRE(init_T_ratio > 0.0, "kinetics: init_T_ratio must be positive");
  PARTHENON_REQUIRE(init_axis >= 1 && init_axis <= 3, "kinetics: init_axis must be 1-3");
  params.Add("init_T_ratio", init_T_ratio);
  params.Add("init_axis", init_axis);

  // Fields
  auto MetadataKinetics = pkg->GetMetadataFlag();
  auto MetadataOperatorSplit = Metadata::GetOrAddFlag(riot::metadata::OperatorSplit);
  const int ncomp_f = tt ? params.Get<TT::TTLayout>("tt_layout").Size() : grid.Size();
  Metadata mf({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
               Metadata::Restart, MetadataKinetics, MetadataOperatorSplit},
              std::vector<int>({ncomp_f}));
  // Fine ghosts and new fine cells copy the parent cell (S4-Q7; Parthenon's default
  // minmod slope is a nonlinear limiter). The slot-wise copy is exact for TT too; TT
  // restriction is RestrictTT (kinetics_amr.cpp).
  {
    using namespace parthenon::refinement_ops;
    if (tt && amr_prolong == "linear") {
      mf.RegisterRefinementOps<RefinementNoOp, RefinementNoOp>();
    } else if (tt) {
      mf.RegisterRefinementOps<ProlongatePiecewiseConstant, RefinementNoOp>();
    } else if (amr_prolong == "linear") {
      mf.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
    } else {
      mf.RegisterRefinementOps<ProlongatePiecewiseConstant, RestrictAverage>();
    }
  }
  if (tt) {
    pkg->AddField<fields::f_tt>(mf);
  } else {
    pkg->AddField<fields::f>(mf);
  }

  Metadata ms({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, MetadataKinetics,
               MetadataOperatorSplit});
  pkg->AddField<fields::rho>(ms);
  pkg->AddField<fields::temperature>(ms);
  pkg->AddField<fields::pressure>(ms);
  pkg->AddField<fields::eq_fallback>(ms);
  Metadata m3({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, MetadataKinetics,
               MetadataOperatorSplit},
              std::vector<int>({3}));
  pkg->AddField<fields::velocity>(m3);
  pkg->AddField<fields::heat_flux>(m3);
  Metadata m6({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, MetadataKinetics,
               MetadataOperatorSplit},
              std::vector<int>({6}));
  pkg->AddField<fields::stress>(m6);
  if (adaptive) {
    pkg->AddField<fields::noneq>(ms);
    pkg->CheckRefinementMesh = CheckRefinement;
  }
  if (closure) {
    // No OperatorSplit flag: the hydro stage registers (u0, u1) read the closure.
    // OneCopy, so they share base's memory.
    Metadata m9({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, MetadataKinetics},
                std::vector<int>({9}));
    pkg->AddField<fields::closure_old>(m9);
    pkg->AddField<fields::closure_new>(m9);
    Metadata m5w({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, MetadataKinetics},
                 std::vector<int>({5}));
    pkg->AddField<fields::wall_flux>(m5w);
  }
  if (lomac) {
    Metadata m2l({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, MetadataKinetics,
                  MetadataOperatorSplit},
                 std::vector<int>({2}));
    pkg->AddField<fields::lomac_stat>(m2l);
  }
  if (tt) {
    Metadata m2({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, MetadataKinetics,
                 MetadataOperatorSplit},
                std::vector<int>({2}));
    pkg->AddField<fields::tt_rank>(m2);
    Metadata m3r({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, MetadataKinetics,
                  MetadataOperatorSplit},
                 std::vector<int>({3}));
    pkg->AddField<fields::tt_round>(m3r);
    pkg->AddField<fields::tt_amr_round>(m3r);
    if (params.Get<std::string>("tt_diag") == "cross") {
      Metadata m5({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, MetadataKinetics,
                   MetadataOperatorSplit},
                  std::vector<int>({5}));
      pkg->AddField<fields::tt_cross>(m5);
    }
  }

  // Boundary conditions (needs grid, eq_params and species)
  EnrollKineticBCs(pkg.get(), pin);

  // Hooks
  pkg->PostInitializationMesh = PostInitialization;
  pkg->UserWorkBeforeOutputMesh = SetDerivedMomentsMesh;
  pkg->UserWorkBeforeLoopMesh = BeforeLoop;

  // History
  using parthenon::UserHistoryOperation;
  parthenon::HstVec_list hst_vecs = {};
  hst_vecs.emplace_back(parthenon::HistoryOutputVec(UserHistoryOperation::sum,
                                                    HistorySums, "kinetics_sums"));
  if (tt) {
    hst_vecs.emplace_back(parthenon::HistoryOutputVec(
        UserHistoryOperation::sum, HistoryRankSums, "kinetics_tt_ranks"));
    hst_vecs.emplace_back(parthenon::HistoryOutputVec(
        UserHistoryOperation::sum, HistoryRoundSums, "kinetics_tt_round"));
    if (multilevel)
      hst_vecs.emplace_back(parthenon::HistoryOutputVec(
          UserHistoryOperation::sum, HistoryAmrRoundSums, "kinetics_tt_amr_round"));
    if (params.Get<std::string>("tt_diag") == "cross")
      hst_vecs.emplace_back(parthenon::HistoryOutputVec(
          UserHistoryOperation::sum, HistoryCrossSums, "kinetics_tt_cross"));
  }
  if (lomac)
    hst_vecs.emplace_back(parthenon::HistoryOutputVec(
        UserHistoryOperation::sum, HistoryLomacSums, "kinetics_lomac"));
  if (multilevel)
    hst_vecs.emplace_back(parthenon::HistoryOutputVec(
        UserHistoryOperation::sum, HistoryBlocksPerLevel, "kinetics_blocks_per_level"));
  pkg->AddParam<>(parthenon::hist_vec_param_key, hst_vecs);
  parthenon::HstVar_list hst_vars = {};
  hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::min,
                                                    HistoryMinF, "kinetics_min_f"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      UserHistoryOperation::sum, HistoryFallbackCount, "kinetics_eq_fallbacks"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      UserHistoryOperation::max, HistorySubsteps, "kinetics_substeps"));
  if (tt) {
    hst_vars.emplace_back(parthenon::HistoryOutputVar(
        UserHistoryOperation::max, HistoryMaxRank, "kinetics_tt_max_rank"));
  }
  if (adaptive)
    hst_vars.emplace_back(parthenon::HistoryOutputVar(
        UserHistoryOperation::max, HistoryNoneqMax, "kinetics_noneq_max"));
  pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

  if (Globals::my_rank == 0) {
    std::stringstream msg;
    msg << "kinetics: " << (tt ? "tensor-train" : "dense") << " f on " << nv[0] << " x "
        << nv[1] << " x " << nv[2] << " velocity nodes";
    if (tt) {
      msg << " (rank <= " << params.Get<TT::TTLayout>("tt_layout").rcap << ", eps "
          << params.Get<TT::RoundParams>("tt_round").eps << ", " << ncomp_f
          << " reals per cell vs " << grid.Size() << " dense)";
    }
    msg << "; particle mass m = " << species.mass << " g, k_B/m = " << species.kb_per_m
        << " erg/(g K)" << std::endl;
    for (int d = 0; d < 3; ++d) {
      msg << "  v" << d + 1 << " in [" << vmin[d] << ", " << vmax[d]
          << "], dv = " << grid.dv[d] << std::endl;
    }
    std::cout << msg.str();
  }

  return pkg;
}

} // namespace Kinetics
