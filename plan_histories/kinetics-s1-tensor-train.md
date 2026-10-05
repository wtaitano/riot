# Kinetics S1: tensor-train representation of f per cell

> **Filename note:** no PR number existed when this file was written. Rename it to
> `plan_histories/<PR-number>.md` when the PR is opened.

Adds `kinetics/representation = tt`. In each spatial cell, f is stored as a 3-core tensor
train (TT) over (vx, vy, vz) instead of a dense Nv³ array. Streaming, relaxation, boundary
conditions, moments, restart and MPI all work on the TT cores directly. The dense S0
solver (`representation = dense`, still the default) is unchanged and serves as the
oracle. The scope is CPU and MPI only, with hydro on and uncoupled, as in S0.

Scope: the commits on `taitano/vlasov` after S0 (`75a834e`): `4a67732`, `1e2263c`,
`072cad9`, `54033d8`, `a1ccbb4`, `957627b` (black only), `252d984`, `4b8b504` and
`b392d32` (steps 1–8), plus step 9 (Nv scaling study and docs). The design and decision
log is `claude_sessions/kinetic_bgk/S1_DESIGN.md`. The decision numbers S1-Qn below refer
to it.

## Motivation

Dense f costs Nv³ reals per cell: 262,144 at Nv = 64. S1 tests whether a low-rank TT
representation gives the same accuracy as dense at tight tolerance while using measurably
less memory, and how its cost scales with Nv (S1-Q1). A wall-clock win at Nv = 32 is not a
success criterion. 2D-3V runs too large for dense memory belong to S2.

Out of scope: hydro coupling, moment-preserving rounding or other conservation repair
(S3), variable-size TT storage (OQ4/S2), GPU-efficient kernels, mesh refinement in TT
mode, and the minmax limiter in TT mode.

## Design decisions

- **TT-native (S1-Q2, Q15).** `kinetics.f_tt` holds the cores plus the actual ranks
  (r1, r2) for each cell. It is allocated only for `tt`; dense `kinetics.f` is allocated
  only for `dense`. There is no dense f in TT mode. The representation is chosen in one
  branch when the task list is built. The derived moment outputs are shared between the
  two representations.
- **Fixed-size storage (S1-Q11).** Each cell gets 2 + Nx·r + Ny·r² + Nz·r reals, sized for
  `tt_rank_max`; for example 9,218 vs 32,768 at Nv = 32, r = 16. Reports give both the
  allocated size (r_max) and the size actually used (ranks from history). Ranks travel
  inside the variable, so Parthenon's ghost exchange is unchanged (S1-Q32).
- **Truncation (S1-Q3, Q12, Q13, Q25).** `tt_eps` is a relative ε per rounding,
  ‖f − f̃‖_F ≤ ε‖f‖_F, with the cap `tt_rank_max`. `tt_eps = 0` gives a fixed rank
  min(r, numerical rank). Defaults: `tt_eps = 1e-8`, `tt_rank_max = 16`. Errors from
  repeated roundings add up, and this is documented and measured. Rate tests run at
  ε ≤ 1e-12.
- **Rounding (S1-Q4, Q8, Q18).** f is rounded after every elementary operation:
  right-to-left QR, then left-to-right SVD with ε/√2 per cut. The small Householder QR
  and one-sided Jacobi SVD are written in-house as serial `KOKKOS_INLINE_FUNCTION` code
  (KokkosBatched is not used). The kernels run one cell per team with level-1 team
  scratch.
- **Operators (S1-Q5, Q18).** The Mieussens equilibrium M is rank 1, built from 1D
  factors. Relaxation is `(1−c) f + c M` followed by one rounding. Streaming is a
  semi-Lagrangian (SL) update whose weights depend only on v_d, so each shift scales one
  core. Multi-D streaming in TT is done as x, y, z sweeps with a rounding after each
  sweep; dense keeps its single tensor-product pass. Moments are exact separable
  contractions of the cores.
- **BCs (S1-Q18).** Outflow copies the TT. A specular wall reverses the wall-normal core,
  which is exact. A diffuse wall forms a masked block sum of the reflected f and a rank-1
  wall Maxwellian, then rounds.
- **Conservation (S1-Q6).** Rounding is allowed to break discrete conservation in S1. The
  drift is monitored against dense. A moment-preserving rounding is deferred to S3 (OQ5).
- **Limiter (S1-Q7).** `sl_limiter = minmax` is rejected with `representation = tt`.
- **No f = M + g split (S1-Q27).** g has no positivity principle, and rank study 2 found
  no clear rank gain.
- **Fixed core order (vx, vy, vz) (S1-Q26).** To be revisited in S2.
- **Nonlinear diagnostics by cross approximation (S1-Q14, Q19–Q24, Q33).** These are
  computed by an in-house DEIM + greedy cross, `TTCross(φ, f_tt) → g_tt`.
  `tt_diag = cross` (the default) or `exact`, which decompresses every cell at history
  cadence. Entropy uses the regularized f ln(f + δ) with
  δ = `tt_cross_delta` × sampled max|f| (Q33 amends Q21). Min f is a sampled upper bound:
  the trial set plus coordinate descent along fibers. Negative mass is always exact
  (decompression), because its integrand is rounding noise.
- **Initial data and restart (S1-Q17, Q30, Q32).** TT runs start from rank-1 equilibrium
  or bi-Maxwellian data, or from the new rank-2 `init = two_maxwellian`, which is exact in
  both representations. TT runs restart from the cores. A restart requires the same
  representation, Nv and `tt_rank_max`.
- **Verification approach (S1-Q9, Q16).** Separate dense and TT runs are compared by
  analysis scripts; there is no shadow mode. The Nv scaling study uses Nv = 16/32/64;
  Nv = 16 uses r_max = 8.

## Proposed changes

### New TT headers (`src/kinetics/`)

- `tt_linalg.hpp`: strided `Mat`, Householder QR, `ApplyQ`, and a one-sided Jacobi SVD
  with an overflow-safe rotation.
- `tt_tensor.hpp`: fixed-capacity cell layout `TTLayout` [r1, r2, G1, G2, G3], `TTRef`
  over any `data(n)` functor (caches the ranks), `FillMaxwellian(s)`, `CopyTT`,
  `AddInto` / `AddScaledInto`, `ReverseAxis`, `Contract`, `Round` (returns
  `RoundInfo{r1, r2, cap_hit, discarded, norm, svd_ok}`) and the host check
  `RequireTeamScratch`.
- `tt_moments.hpp`: raw moments, central moments and heat flux by exact core
  contractions, O(n r1 r2).
- `tt_relax.hpp`: `RelaxCellTT`.
- `tt_stream.hpp`: `StreamCellTT`, which rounds after the neighbour on each side is added.
- `tt_cross.hpp`: `TTCross` (two DEIM projections, pivoted MGS `Orth`, `QDEIM`, greedy
  enrichment on a fixed hash trial set that does not depend on the MPI decomposition),
  `EntropyPhi` and `SampledMinF`.
- `kinetics_cell.hpp`: `DenseCell` / `TTCell` and `WithCell` dispatch (Raw, Central,
  ForEach, Fill), so init, resolution checks, derived moments and history do not depend
  on the representation.

### Kernels, tasks, BCs, init, output (existing S0 files)

- `bgk.cpp`: kernel `RelaxTT` (`par_for_outer`, one cell per team, `RelaxScratch`).
- `semi_lagrangian.cpp`: `StreamTT` as x1/x2/x3 sweeps (`SweepTT`). Each sweep covers
  the interior along the dimensions already swept, plus one ghost layer along the ones
  not yet swept, so one ghost exchange per step is enough. A third register
  `kinetics_k2` is used only for multi-D TT.
- `kinetics_tasks.cpp`: TT branch for both integrators (`sl_dirk2`, `strang`). New
  `kinetics/streaming` switch (default true); false runs collisions only, for 0D tests.
- `kinetics_bcs.cpp`: `KineticBCImplTT` (outflow, specular and diffuse walls as above).
  `CopyF` works for both f and f_tt.
- `kinetics_init.cpp`: representation-agnostic init and the new `two_maxwellian` init
  (half the density at u ∓ δ e_axis, δ = `init_drift`·√θ, θ' = θ − δ²/3).
- `kinetics_output.cpp`: derived fields `kinetics.tt_rank`, `kinetics.tt_round`
  (summed discarded norm / ‖f‖, cap hits, SVD failures per hydro step) and
  `kinetics.tt_cross`. Kernel `CrossEntropyTT`. History columns
  `kinetics_tt_ranks_0..3`, `kinetics_tt_max_rank`, `kinetics_tt_round_0..2` and
  `kinetics_tt_cross_0..3`.
- `kinetics.{hpp,cpp}`: inputs `representation`, `tt_eps`, `tt_rank_max` (2–64),
  `tt_diag`, `tt_cross_eps` (1e-10), `tt_cross_rank_max` (32), `tt_cross_trial_factor`
  (16, cap 64) and `tt_cross_delta` (1e-12). Guards in TT mode: no mesh refinement and no
  minmax limiter. Restart layout guard: params `f_layout` / `f_layout_file` and
  `CheckRestartLayout` (in `UserWorkBeforeLoopMesh`) abort with both layouts printed on a
  mismatch, and refuse restart files written before the guard existed.
- `src/CMakeLists.txt`: new headers added.

### Tests

- Unit (Catch2): `tst/unit/test_kinetics_tt.cpp` with T1 (QR/SVD, rank-1 M, rounding vs
  the analytic optimum, block sums), T2 (TT moments = dense moments), T3 (TT relaxation =
  dense), T4 (TT streaming = dense SL node by node, ReverseAxis, Contract) and T5 (cross).
  Registered in `tst/unit/CMakeLists.txt`.
- Regression (`tst/scripts/kinetics/`): `tt_init.py`, `tt_relax0d.py`, `tt_stream1d.py`,
  `tt_accumulation.py`, `tt_multid.py`, `tt_restart_mpi.py` and `tt_cross_diag.py`.
  `stiff_time_order.py` is extended to `representation = tt`.
- Helpers in `tst/scripts/utils/kinetics_utils.py`: `tt_decompress` (dense f from restart
  cores) and `read_history`.

### Decks and docs

- `inputs/kinetics/blast2d.py`: new 2D-x deck (S1-Q28). Pressure pulse in r < 0.2 with a
  drift, periodic, ν = 1e2, Nv 20×20×14, and a restart output block.
- `doc/sphinx/src/packages/kinetics.rst`: new "Tensor-train representation" section
  (parameters, `two_maxwellian`, rounding accumulation note, diagnostics), plus step 9
  scaling results.

## Testing

The results below come from the step-status table in `S1_DESIGN.md` and, for step 9, from
`claude_sessions/kinetic_bgk/s1/step9/RESULTS.md` (table `results_table.md`, runner
`run_study.py`, analysis `analyze.py`). All runs were CPU and serial or MPI; **no GPU runs were made.** At
step 8 the counts were ctest 47/47 and kinetics regressions 18/18. Each new comparator was
checked with a deliberate mutation that it caught (listed per step in `S1_DESIGN.md`).

| Step | Test | Recorded result |
|---|---|---|
| 1 | T1 | QR/SVD to 1e-14; Mieussens M rank 1 to 1e-14; rounding of a tensor with a known spectrum gives the analytic ranks and the analytic optimal error; reported discarded norm = actual error; A + A and 2A − 3B exact in fixed-rank mode. |
| 2 | T2, `tt_init.py` | TT moments = dense moments to 1e-13. Dense vs TT initial f 3e-16, moments ≤ 5e-14, history ≤ 5e-14. Ranks exactly 1 (equilibrium) and 2 (`two_maxwellian`). |
| 3 | T3, `tt_relax0d.py` | TT relaxation = dense to 1e-13 (exact and rational c, including c > 1). relax0d TT vs dense history ≤ 3.9e-13, anisotropy ≤ 1e-12 at every dump, max rank 2, min f / max f ≥ −1.8e-16. Negative control (rank cap 1): 1.3e-2. |
| 4 | T4, `tt_stream1d.py` | At `tt_eps` 1e-14 vs dense: Sod ν = 1e4 1.0e-12 (max rank 7); ν = 0 5.4e-13; smooth wave with `sl_order` 2, unlimited, 7.1e-12; specular box 2.0e-12; diffuse box 4.6e-12; Couette (Nv 24×24×12) 1.2e-12 (rank 11). ε convergence on Sod: 1e-10 → 2.7e-9, 1e-12 → ~1e-11, 1e-14 → 1e-12. |
| 5 | `stiff_time_order.py` (TT), `tt_accumulation.py` | TT orders: `sl_dirk2` 1.98/1.99, `strang` 1.06/0.98, identical to dense to 9 digits; TT vs dense ≤ 2.9e-11. Accumulation: error = 0.17–0.69 · N · ε over 192–1536 roundings; the summed discarded norm is 0.14–0.38 of the measured error. |
| 6 | `tt_multid.py` | blast2d 24×24: 1.5e-12 at ε 1e-14 (max rank 17, cap 24); 9.0e-9 at 1e-10; 1.0e-6 at 1e-8 (negative control). 3D 8³, Nv 14³, ε 1e-12: 4.1e-11 (rank 11). |
| 7 | `tt_restart_mpi.py` | Restart round trip bitwise (f_tt and derived moments). 1 vs 4 ranks bitwise on 1D Sod and blast2d (4 blocks); history 1.4e-14. Three layout-mismatch cases refused. |
| 8 | T5, `tt_cross_diag.py` | Cross vs exact entropy: Sod ν = 1e4 1.0e-12 (1,972 evals/cell of 32,768, g rank 3.1); ν = 1e1 4.1e-12; Couette 3.6e-12; blast2d 5.4e-12. Min f gap ≤ 1.8e-11 max f, and the upper bound held. Mass, energy and negative mass bitwise. Negative control (rank cap 1): 3.5e-3. |
| 9 | Nv scaling (S1-Q16) | See below. |

**Step 9: Nv = 16/32/64 scaling study.** relax0d, and 1D Sod (128 cells, t = 0.1) at
ν = 1e4 and ν = 1e1, each run dense and TT at ε = 1e-14 and 1e-8, serial on one core.
r_max = 8 at Nv = 16 and 16 otherwise. The numbers below are from `RESULTS.md`; a summary is in the
"Scaling with the velocity grid" paragraph of `doc/sphinx/src/packages/kinetics.rst`.

- **Accuracy vs dense.** At ε = 1e-14 the worst moment error is ≤ 2.8e-11 in every case;
  at ε = 1e-8 it is 4e-7 to 1.2e-6. These values do not grow with Nv.
- **Rank.** It does not grow with Nv: max rank 2 (relax0d), 9 (Sod ν = 1e4) and 8
  (Sod ν = 1e1) at ε = 1e-14, and 5 at ε = 1e-8. There were no rank-cap hits and no cross
  cap hits.
- **Storage per cell at Nv = 64** (dense 262,144):
  - allocated: 18,434;
  - used, mean at the end: 512 (relax0d), 1,305 (Sod ν = 1e4, ε = 1e-14) and 3,454
    (Sod ν = 1e1, ε = 1e-14).
- **Wall time at Nv = 64:**
  - Sod ν = 1e4: dense 224 s, TT 14.1 s (ε = 1e-14), 4.4 s (ε = 1e-8);
  - Sod ν = 1e1: dense 331 s, TT 21.5 s (ε = 1e-14);
  - peak RSS: dense 2.7 GB, TT 0.28 GB.
- **Small Nv.** At Nv = 16, TT at ε = 1e-14 is slower than dense on Sod (2.3–4.0 s vs
  1.6 s).
- **Cross cost.** Evaluations per cell as a fraction of Nv³ fall from 0.3–0.6 at Nv = 16
  to 0.02–0.07 at Nv = 64.

## Known limitations and open items

- **CPU only (S1-Q10).** The kernels are device-safe, and the portability reviewer ran
  at each step, but each AUTO-sized team runs only one active thread (about 1% lane use
  on a GPU). Level-1 scratch grows as rcap²·Nv: about 1.2 MB/team for `StreamTT` at
  Nv = 32, rcap = 16. Forcing team size 1 would blow up the level-1 pool. The GPU fix
  (TeamThreadRange over the QR/ApplyQ/AddInto loops) is deferred.
- **Cross at the rank cap (`tt_cross.hpp`, around L430).** A cross that stops at
  `tt_cross_rank_max` without meeting `tt_cross_eps` still returns its estimate. It gives
  no runtime warning; only the history columns `kinetics_tt_cross_2/3` count such cases.
  A trial set that is too sparse can also report "converged" while wrong: at
  `tt_cross_trial_factor` 4, relax0d was 1.6e-8 off, which is why the default is 16.
- **`CrossEntropyTT` runs once per MeshData partition** inside `HistorySums`, at history
  cadence. This is fine with one partition and redundant with more than one.
- **No conservation under rounding (S1-Q6, OQ5).** Mass and energy drift at the level of
  the rounding error (for example 2–4e-9 mass drift on Sod at ε = 1e-8). A
  moment-preserving rounding is needed in S3.
- **TT mode does not support mesh refinement or the `minmax` limiter.** Both are rejected
  at startup.
- **Fixed-size storage.** Below the rank cap, the allocation is larger than the data in
  use. At small Nv the TT allocation can exceed dense: Couette Nv 24×24×12 6,722 vs 6,912;
  blast2d 5,666 vs 5,600; 3D Nv 14³ 4,034 vs 2,744. Variable-size storage is OQ4/S2.
- **Default rank cap in 2D.** `tt_rank_max = 16` is marginal for tight ε in 2D: blast2d
  needs rank about 17 at ε = 1e-14.
- **3D ghost layers.** The first sweep leaves half-swept values in the base ghost layers
  until the next exchange (documented in the code).
- The "cap hit" history column counts cells at the cap, not individual rounding cuts.

### Bugs found and fixed during S1

- `tt_round` missing from the k0/k1 register list (segfault, step 3).
- `tt_round` not reset in collisionless TT runs (step 4).
- The 2D sweep register table read an unwritten `k2` (step 6; caught by the TT-vs-dense
  comparison).
- A restart with a different `tt_rank_max` or representation read f silently wrong
  (step 7; fixed with the layout guard).

## AI disclosure

Claude (Anthropic) helped write this change: the design log, the implementation, the
tests, the docs and this file, under the direction and review of the author. Per the
ML-disclaimer rule in the PR template, the new source, test and deck files carry a
generative-AI disclaimer at the top.
