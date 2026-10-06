# Kinetics S4: static and adaptive mesh refinement for kinetics (dense and tensor-train f)

> **Filename note:** no PR number existed when this file was written. Rename it to
> `plan_histories/<PR-number>.md` when the PR is opened.

> This file was made in part with generative AI.

S4 lets the kinetics package run on Parthenon's block-structured meshes with real-space
refinement, both static (`parthenon/mesh/refinement = static`) and adaptive (`adaptive`,
remesh plus load balancing). It works with dense and tensor-train (TT) f, uncoupled or
with closure coupling and LoMaC.

Scope: steps 1-8 of the S4 plan on `taitano/vlasov`, on top of S3 (`f75c1de`, after the
rebase). Steps 1-4 are committed: `720cb33` (dense SMR), `ea86f04` (TT SMR), `b54b4af`
(2D SMR regression), `040a22e` (linear-slope prolongation), `80a6a57` (AMR remesh),
`f49c04d` (MPI/restart, Sod AMR and 2D blast regressions), and the commit adding this
file (docs). The design, decision log and implementation log are in
`claude_sessions/kinetic_bgk/S4_DESIGN.md`; decision numbers S4-Qn below refer to it.
The measured numbers come from its implementation log (steps 1-5) and from
`claude_sessions/kinetic_bgk/s4/RESULTS.md` (all steps, including 6-8).

## Motivation

Before S4, a TT run refused any mesh refinement. A dense run had no guard, but it would
have used Parthenon's default ops on f: minmod prolongation (a nonlinear limiter) and
average restriction. That path was never verified, and the closure coupling refused
adaptive meshes. Kinetic effects are local (shocks, blast fronts), so refining only where
f is far from equilibrium is the natural next step for the real-space grid (S4-Q1, Q2).
The TT layout makes this hard: a TT cell is a packed set of cores, and every Parthenon
refinement op other than a copy is invalid on it slot by slot.

Out of scope: per-level time subcycling (S4-Q9), conservative f at fine-coarse faces
(S4-Q10), coupled kinetic walls on refined meshes (S4-Q5), verification in 3D (S4-Q3),
limited-slope prolongation (S4-Q7), GPU runs.

## Design decisions

- **Dimensions and representations (S4-Q3, Q4).** 1D and 2D are verified; the code also
  allows 3D. Dense f is the oracle: TT at `tt_eps` 1e-14 must follow dense on the same
  mesh and the same remesh sequence.
- **Prolongation (S4-Q7, Q16, Q21).** `kinetics/amr_prolong = constant | linear`, default
  `constant`. Constant is piecewise-constant injection: exact in TT, no rank growth,
  positive, conservative. Linear uses unlimited slopes (Parthenon
  `ProlongateSharedLinear` for dense, a kinetics kernel for TT). A limited slope was
  rejected for now: in TT it needs cross approximation, which is nonlinear.
- **TT restriction (S4-Q11, Q19, Q20).** The mean of the 2^d child TTs is built by
  sequential pairwise add and round (input rank 2r), with the run's `tt_eps` and rank
  cap. It never forms a dense Nv³ sum. Parthenon's restriction for `f_tt` is a no-op;
  kinetics fills `coarse_s` itself before each boundary send and before each remesh. Dense
  keeps `RestrictAverage`. No edits under `external/`.
- **Time step (S4-Q9).** One global SL substep, set by the finest level (the existing
  MPI-min over block dx). There is no per-level subcycling, as in hydro.
- **Conservation (S4-Q10).** SL is nodal, not flux form, so f moments leak at fine-coarse
  faces. This is accepted and measured. Hydro is flux-corrected (closure fluxes are added
  before Parthenon's flux correction), and LoMaC restores the f moments.
- **Remesh (S4-Q12, Q13, Q22).** On new blocks, hydro (minmod) and f (injection) may
  disagree; this is accepted. After every remesh, `closure_old` is recomputed from f, and
  with `lomac = true` LoMaC runs on all blocks.
- **Tagging (S4-Q6, Q14, Q15, Q23).** Hydro criterion OR kinetic criterion: Parthenon
  refines if any criterion asks and derefines only if all agree. The kinetic criterion is
  ‖f − M[f]‖₂ / ‖f‖₂, with M the discrete Maxwellian of BGK. It is exact in TT through
  inner products (M has rank 1). Inputs `amr_noneq_refine` (1e-2) and
  `amr_noneq_derefine` (1e-3). The hydro criterion uses Parthenon's
  `<parthenon/refinement0>`, with no new code.
- **Guards (S4-Q24).** Refused: coupled kinetic walls with refinement, and non-Cartesian
  coordinates with multilevel. Warning: `sl_order = 2` with `amr_prolong = constant`. The
  TT refinement ban and the closure + adaptive ban are removed.
- **Verification plan (S4-Q8, Q17, Q18, Q27).** Interface order bars: injection ≥ 0.9,
  linear + `sl_order 2` ≥ 1.8, TT vs dense ≤ 1e-10. Plus MPI-2 runs, an SMR restart with
  bitwise f, a 1D Sod AMR regression, and a 2D blast showcase. The showcase gate is
  relative L1 ≤ 1e-2 against uniform fine; "≥ 2× fewer cells and lower wall time" is
  reported, not gated.
- **History (S4-Q25), order (S4-Q26).** History columns for the TT round tally, max
  kinetic criterion and blocks per level. Eight steps, each committed separately.

## Proposed changes

### Kinetics package (`src/kinetics/`)

- `kinetics_amr.cpp` (new):
  - `RestrictTT` / `RestrictBox`: the TT average of the children goes into the coarse
    buffer of blocks with a coarser neighbor (pairwise add + round). Tallies go to
    `kinetics.tt_amr_round`.
  - `RestrictGhostsTT`: restricts the coarse ghosts next to same-level or finer neighbors
    (nghost/2 layers, once per offset), which feed the linear slopes. Parthenon's
    SetBounds restriction does this for dense f.
  - `ProlongateTT` / `ProlongBox`: TT linear prolongation of fine ghosts from `coarse_s`,
    f = (1 − ndim/4) f_C + ¼ Σ_d f_{C±e_d}, rounded pairwise. The region is extended
    tangentially by nghost, as in Parthenon's CalcIndices.
  - `AddFExchangeTasks`: RestrictTT | StartReceive → exchange, with the TT BC hooks. It
    replaces the three f-exchange sites (two in `kinetics_tasks.cpp`, one in
    `kinetics_lomac.cpp`). `ExchangeFGhosts` redoes the exchange after Parthenon's
    init/restart exchanges, which ran with the no-op restriction.
  - `ComputeNonEquilibrium` (field `kinetics.noneq`; dense node sums, TT via `TT::Dot`)
    and `CheckRefinement` (registered as `CheckRefinementMesh`; block max against the
    thresholds).
  - `RestrictForRemesh`: with `linear`, a ghost exchange. With TT, `RestrictBox` on
    blocks whose derefinement count has reached `derefine_count` (a superset of the
    blocks that actually derefine).
  - `AfterRemesh`: runs only if `pm->modified`. With linear TT, `ProlongBox` on new fine
    blocks. With LoMaC, `RoundToRank` to `tt_rank_max − 2`, then `LomacTasks`.
    Otherwise `ExchangeFGhosts` + `ComputeClosure`.
- `kinetics.cpp`: registers refinement ops per representation and `amr_prolong`:
  dense `ProlongatePiecewiseConstant` or `ProlongateSharedLinear` with
  `RestrictAverage`; TT `ProlongatePiecewiseConstant` + `RefinementNoOp`, or
  `RefinementNoOp` for both when linear. Also adds the params `amr_prolong`, `amr_noneq_refine`, `amr_noneq_derefine` and
  `amr_derefine_count`, the field `noneq`, and the guards and warning above. The TT and
  closure adaptive bans are removed.
- `kinetics.hpp`: fields `tt_amr_round` (3) and `noneq`, and declarations.
- `tt_tensor.hpp`: `TT::Dot` (Frobenius inner product by core contraction,
  O(n rA rB (rA + rB))) and `DotWorkSize`.
- `kinetics_output.cpp`: history `kinetics_tt_amr_round_0..2`, `kinetics_noneq_max`,
  `kinetics_blocks_per_level_*`.
- `kinetics_bcs.cpp`: refuses coupled kinetic walls on a refined mesh.
- `bgk.cpp`: resets `tt_amr_round` with the other per-step flags.
- `kinetics_closure.cpp` (`BeforeLoop`): calls `ExchangeFGhosts`.
- `kinetics_lomac.cpp`, `kinetics_tasks.cpp`: use `AddFExchangeTasks`. `tt_amr_round` is
  added to the shallow containers.

### Driver and build

- `riot_driver.{hpp,cpp}`: `Kinetics::RestrictForRemesh` runs after `RiotPostStepTasks`
  (after tagging, before Parthenon's remesh). A new `SetGlobalTimeStep` override calls
  `Kinetics::AfterRemesh`; Parthenon calls it after every load-balancing/AMR pass. New
  member `do_kinetics`.
- `src/CMakeLists.txt`: adds `kinetics_amr.cpp`.

### Tests

- `tst/unit/test_kinetics_tt.cpp`: new case "T1: TT inner product equals the dense sum of
  products". The ranks differ, and the tolerance is on the ‖A‖² scale.
- `tst/scripts/kinetics/` (new): `smr_interface.py` (interface order, mass leak, coupled
  and LoMaC on SMR), `tt_smr.py` (TT vs dense on SMR: 1D wave/Sod/Sod+LoMaC, 2D blast
  constant and linear), `tt_amr.py` (1D AMR wave: constant/linear/coupled/LoMaC, dense
  and TT; Sod refined by the kinetic criterion alone), `amr_mpi_restart.py` (SMR
  restart, 1 vs 2 ranks), `sod_amr.py`, `blast2d_amr.py`.

All new source and test files start with a generative-AI disclaimer and the Triad
copyright header.

### Docs

- `doc/sphinx/src/packages/kinetics.rst`: new "Mesh refinement" section (injection and
  linear prolongation, TT restriction, shared substep, adaptive criterion and remesh
  handling, history columns, regression checks); parameter rows for `amr_prolong`,
  `amr_noneq_refine`, `amr_noneq_derefine`; the closure-coupling note now says static
  and adaptive refinement are supported.

Study artifacts outside the PR tree: `claude_sessions/kinetic_bgk/S4_DESIGN.md`,
`s4/RESULTS.md`, scratch scripts in `s4/step*/`.

## Testing

All runs on CPU (macOS, gcc-16, Open MPI, Serial Kokkos). **No GPU runs were made.**

### Interface order (`smr_interface.py`, 1D entropy wave, ν = 0, middle quarter refined)

| case | L1 at nx 64 / 128 / 256 | orders |
|---|---|---|
| sl_order 1, injection | 2.07e-3 / 1.09e-3 / 5.64e-4 | 0.92, 0.96 |
| sl_order 1, root level only | 2.17e-3 / 1.14e-3 / 5.89e-4 | 0.92, 0.96 |
| sl_order 2, injection | 2.74e-4 / 1.26e-4 / 6.03e-5 | 1.12, 1.06 |
| sl_order 2, linear | 1.43e-4 / 3.55e-5 / 8.84e-6 | 2.01, 2.01 |
| 2D, sl_order 1, patch with corners (scratch) | 2.26e-3 / 1.21e-3 / 6.18e-4 | 0.90, 0.97 |

Kinetic mass leak at fine-coarse faces: 4.3e-4, 2.2e-4, 1.1e-4 (first order). Coupled on
SMR, hydro mass and energy drift 7.8e-16 and 8.7e-16. LoMaC brings the kinetic sums to
the hydro sums within 4e-15.

### TT vs dense (`tt_smr.py`, `tt_amr.py`; tt_eps 1e-14, max scaled moment difference)

| case | difference |
|---|---|
| SMR 1D wave / Sod coupled / Sod + LoMaC | 1.1e-11 / 1.3e-12 / 2.1e-12 |
| SMR 2D blast, constant / linear | 1.2e-12 / 1.5e-12 |
| AMR 1D wave, constant / linear / coupled / LoMaC | 4.5e-12 / 2.3e-11 / 3.6e-12 / 4.6e-12 |
| AMR 2D blast, 2 ranks (scratch) | 2.4e-13 |

Controls with each piece disabled (the test detects a missing piece): RestrictTT off
(SMR) 0.77; RestrictGhostsTT off (2D linear) 3.8e-4; RestrictForRemesh off (AMR) 0.50.
For `tt_eps` (1e-8 against 1e-14), the difference rises from 4.5e-12 to 6.8e-6 (AMR) and
from 1.3e-12 to 3.5e-7 (SMR). There are no rank-cap hits.

Bugs these probes found, now fixed: stale parent ghosts at the remesh with linear TT
(2.5e-5 → 5e-13); LoMaC skipping every cell after a remesh for lack of TT rank room, so f
is now rounded to cap − 2 first (→ 2.9e-12); missing corner ghosts in 2D linear TT
(8e-4 → 2.1e-13).

### Restart and MPI (`amr_mpi_restart.py`)

SMR restart, dense and TT, coupled: f is bitwise identical; hydro agrees to roundoff
(1e-12, as on uniform meshes). 1 vs 2 ranks: SMR Sod TT and the AMR coupled wave TT are
bitwise identical in all dumps.

### Sod AMR (`sod_amr.py`, 1D, ν = 1e3, coupled, Nv 16³, t = 0.15)

| run | L1(ρ) vs uniform 256 | mean cells |
|---|---|---|
| uniform 64 | 1.20e-2 | 64 |
| AMR 64 + 2 levels (hydro gradient OR noneq) | 7.6e-4 | 133.5 (0.52) |
| TT AMR (tt_eps 1e-10) vs dense AMR | 2e-9 | same layout |

### 2D blast showcase (S4-Q18; TT, tt_eps 1e-8, Nv 16³, coupled, 4 ranks, box [−1, 1]²)

| run | rel L1(ρ) vs uniform 64² | mean cell fraction | wall time |
|---|---|---|---|
| uniform 64² | — | 1 | 52.1 s |
| AMR 16² + 2 levels | 1.2e-4 | 0.46 | 46.3 s |
| uniform 16² | 1.8e-2 | 0.0625 | 2.2 s |

- Target (a), rel L1 ≤ 1e-2: met. Target (b), ≥ 2× fewer cells: met (0.46). Lower wall
  time: met only marginally (**1.13×**).
- Each cell costs more under AMR: 8² blocks instead of 16² (more ghost exchange and
  teams), mean TT rank 4.0 against 2.5 (refined cells are where f is out of
  equilibrium), and the restriction/prolongation work.
- On the default [−0.5, 0.5]² box, the blast had filled the whole box with refined
  blocks by t = 0.07. That run saved nothing: 332 s against 234 s uniform.
- Regression `blast2d_amr.py` (Nv 12³, 2 ranks): rel L1 5.0e-5 against 3.0e-2 on the
  coarse mesh, cell fraction 0.56. The wall-time ratio was **1.42, so AMR was slower**;
  fixed costs dominate in a case this small. The wall-time ratio is reported only and
  does not gate the test.

### Verification status

- Steps 1-4: `ctest` 48/48 and the kinetics regressions 23/23 passed. After the step 4
  ghost-depth fix, only `tt_smr`, `smr_interface`, `tt_closure2d` and `tt_stream1d` were
  rerun; all pass.
- Step 5: `ctest` 49/49 and the kinetics regressions 24/24 passed.
- Steps 1-8 together: `ctest` 49/49 and the kinetics regressions 27/27 passed
  (2026-10-06, run under `caffeinate -s -i` on AC power).
- Kokkos portability, by reading the code only: steps 1-4 clean, apart from one fixed
  bug (ghost-restriction depth 1 layer instead of nghost/2; wrong only for nghost ≥ 4,
  not visible in outputs since SL reads 1 ghost). Step 5 review: no GPU-portability defects; its
  findings were fixed before the commit (history columns on static meshes, a failed
  post-step guard, threshold validation, an O(n r^3) `TT::Dot`, no LoMaC re-run on a
  load-balance-only remesh). Nothing was run on a GPU.
- Nothing from S4 is pushed (the branch is 7 commits ahead of the fork at `f75c1de`).

## Known limitations and open items

- **Modest wall-time gain.** The 2D showcase gains 1.13× at 64² (Nv 16³). The small
  regression case is 1.42× slower than uniform. A blast that fills the domain gets no
  benefit.
- **One global SL substep (S4-Q9).** The finest level sets it; there is no per-level
  subcycling. Coarse blocks do extra work. Revisit this if that work dominates.
- **f is not conserved at fine-coarse faces (S4-Q10).** The leak is O(dx) because SL is
  nodal. Hydro is flux-corrected; LoMaC restores kinetic = hydro.
- **AMR restart is untested (S4-Q17).** Only SMR restart is tested; it gives bitwise f.
- **Coupled kinetic walls with refinement are refused (S4-Q5).** This is a follow-up.
- **Performance.** `RestrictTT`/`RestrictBox` and `ProlongBox` launch one kernel per
  block. A possible duplicate corner prolongation affects the tally only.
- **Deferred, to be remembered (user request):** limited-slope prolongation (S4-Q7,
  needs cross approximation in TT) and a cross-approximation (DEIM) restriction of the
  child average that avoids the pairwise-add rank blow-up (S4-Q19).
- **Linear prolongation can make f negative.** This applies to dense f and to TT, since
  the slopes are unlimited. Injection stays positive.
- **Hydro/f mismatch on new blocks (S4-Q12).** Hydro uses minmod and f uses injection or
  linear prolongation; this is accepted. LoMaC removes the mismatch when it is on.
- **History rounding tallies** (`kinetics_tt_round_*`, `kinetics_tt_amr_round_*`) are
  not remeshed. On a remesh cycle they miss the new and moved blocks.
- **Verified in 1D and 2D only.** 3D is allowed but not verified. Cartesian coordinates
  only. GPU not tested.

## AI disclosure

Claude (Anthropic) helped write this change: the design and implementation logs, the
implementation, the tests, the docs and this file, under the direction and review of the
author. As the ML-disclaimer rule in the PR template requires, the new test files
(`smr_interface.py`, `tt_smr.py`, `tt_amr.py`, `amr_mpi_restart.py`, `sod_amr.py`,
`blast2d_amr.py`) start with a generative-AI disclaimer, and so does the new source
`kinetics_amr.cpp`.
