# Kinetics S0: dense discrete-velocity BGK reference solver

> **Filename note:** no PR number existed when this file was written. Rename it to
> `plan_histories/<PR-number>.md` when the PR is opened.

Adds a new `kinetics` physics package: a dense discrete-velocity BGK solver for a neutral
monatomic gas, with 1D/2D/3D physical space (Parthenon) and always 3V velocity space. It
runs alongside RIOT hydro, **uncoupled**, and is validated on CPU with MPI only.

## Motivation

S0 is the uncompressed reference solver for the kinetic project. The later stages are
verified against it: tensor-train (TT) compression of f in velocity space (S1/S2), the
kinetic closures fed back into hydro, and the LoMaC-style additive moment correction (S3).
S0 is therefore permanent code, not scaffolding. A runtime switch
`kinetics/representation = dense | tt` is reserved; S0 accepts only `dense`.

Out of scope for S0: TT, hydro coupling, the moment correction, flux-form SL,
characteristic IMEX-RK, AMR, GPU validation, multiple species, and charged species.

## Proposed changes

### Shared code (wiring only)

- `src/riot.cpp`: new `physics/kinetics` boolean (default `false`). It requires
  `physics/hydro` (`PARTHENON_REQUIRE`), adds the `do_kinetics` param, calls
  `Kinetics::Initialize(pin)`, and pushes `&Kinetics::KineticsTasks` onto
  `OperatorSplitTasks`.
- `src/CMakeLists.txt`: the `src/kinetics/*` files added to `SRC_LIST`.
- `tst/unit/CMakeLists.txt`: the three new Catch2 files added to `riot_unit_tests`.
- `doc/sphinx/index.rst`: new `src/packages/kinetics` page in the toctree.
  `doc/sphinx/src/introduction.rst`: `kinetics` row in the `<physics>` options table.
- **Hydro, driver and problem-generator code are unchanged.** Test R8 (below) checks
  that hydro output is bitwise identical with kinetics on and off.

### New package `src/kinetics/`

- `velocity_grid.hpp`: uniform cell-centered velocity box per axis
  (`v{d}min/v{d}max/nv{d}`), midpoint weights, and the flat index
  n = (iz·Nvy + iy)·Nvx + ix.
- `equilibrium.hpp`: Mieussens discrete equilibrium M_d = exp(α·φ), giving exact discrete
  conservation. It uses damped Newton (Cholesky plus Armijo backtracking) in scaled
  ξ = (v − u)/c, starting from the continuous-Maxwellian α. There are a 5-constraint
  version and a 7-constraint anisotropic version (the second builds bi-Maxwellian
  initial data). Both are separable, i.e. rank 1 in TT. If Newton fails, it falls back
  to the sampled Maxwellian and counts the fallback.
- `moments.hpp`: per-cell moment sums (n, ρu, E, T, stress P_ij, heat flux q_i).
- `bgk.{hpp,cpp}`: relaxation f ← f + c(M_d − f), with c = 1 − e^{−νh} (exact) or
  c = aνh/(1 + bνh) (closed-form implicit DIRK stage; M_d[f] = M_d[f*] because
  relaxation conserves), with
  `nu_model = constant | power_law` (ν = p/μ(T), μ = μ_ref(T/T_ref)^ω). Counts equilibrium
  fallbacks and aborts above `eq_fallback_abort`.
- `semi_lagrangian.{hpp,cpp}`: nodal backward semi-Lagrangian streaming (SL), with
  linear (`sl_order = 1`) or quadratic (`sl_order = 2`) interpolation. Multi-D uses a
  single-pass tensor-product stencil. The optional bracketing min/max limiter is
  `sl_limiter = minmax | none`. `MaxStreamingStep` caps the substep so that every node
  moves at most one cell (`cfl ≤ 1`).
- `kinetics_tasks.cpp`: `KineticsTasks` subcycles inside each hydro step, with
  n = ⌈Δt/h_max⌉ MPI-reduced. `integrator = sl_dirk2 | strang`. Default `sl_dirk2` is a
  characteristic IMEX-RK step (explicit SL, two-stage stiffly accurate L-stable DIRK for
  BGK, γ = 1 − 1/√2): SL(γh), relax c₁ = (1−γ)νh/(1+γνh), SL((1−γ)h), relax
  c₂ = γνh/(1+γνh). The explicit stage combination is folded into c₁, so no stage
  storage; the cap allows h = 1.41× the single-step limit. `strang` is
  SL(h/2)·BGK(h)·SL(h/2), with adjacent half steps merged by default
  (`merge_half_steps`). It alternates registers k0/k1
  (k0 is a shallow copy of base) with one ghost exchange per SL step. Kinetics does not
  vote on the global dt.
- `kinetics_bcs.{hpp,cpp}`: outflow, specular and diffuse-Maxwell walls, implemented as
  package `UserBoundaryFunctions` that run after the mesh BC. A kinetics face must be
  `periodic` exactly when the mesh face is. Specular walls require a v-box symmetric about
  0 on the wall-normal axis. The diffuse-wall equilibrium is solved once at startup, and
  the wall density is set so that the net mass flux is zero.
- `kinetics.{hpp,cpp}`: `Initialize` reads params and defines fields. f is stored as
  `kinetics.f_<species>` with flags Independent, FillGhost, Restart and OperatorSplit.
  Derived output fields are ρ, velocity, temperature, pressure, stress, heat flux and
  `eq_fallback`. A gas-consistency check requires exactly one ideal-gas material with
  Γ = 5/3, and the particle mass is m = k_B/((Γ−1)C_v). A velocity-resolution check (edge
  mass fraction, v_th/Δv) runs at startup and every `check_every` cycles; it warns and
  can abort.
- `kinetics_init.cpp`: `PostInitializationBlock` fills f from the hydro state
  (`init = equilibrium | bimaxwellian`), so both solvers start from the same state.
- `kinetics_output.cpp`: fills the derived moments in `UserWorkBeforeOutputMesh`. History
  outputs: kinetic mass/momentum/energy, entropy H, min f, negative mass, the fallback
  count, the worst moment error and the substeps per hydro step. The same invariants are
  recorded for hydro.
- Every new source file carries the Triad copyright header and the generative-AI
  disclaimer.

### Tests, decks, docs

- Unit tests (Catch2): `tst/unit/test_kinetics_equilibrium.cpp` (U1, U2),
  `test_kinetics_bgk.cpp` (U4) and `test_kinetics_sl.cpp` (U3).
- Regression scripts `tst/scripts/kinetics/`: `relax0d`, `freestream`,
  `free_molecular_sod`, `sod_continuum`, `walls`, `couette`, `non_interference`,
  `restart`, `mpi`, `resolution_checks`, `stiff_time_order`. Shared helpers are in
  `tst/scripts/utils/kinetics_utils.py`: the exact discrete free-streaming solution, an
  exact Euler Riemann solver (`sod_exact`), and the comparators. No gold files; the
  scripts compare against analytic or self-consistent references.
- Decks `inputs/kinetics/{relax0d,sod,freestream,couette}.py`.
- Docs page `doc/sphinx/src/packages/kinetics.rst`: method, parameters, and known
  limitations, including the limiter finding below.

## Testing

The results below were recorded in the design log (S0_DESIGN.md §8) as each step was
finished. The writer of this file did **not** re-run them. Validation is CPU and MPI
only; **no GPU runs were made.**

| ID | Test | Recorded result |
|---|---|---|
| U1 | Velocity quadrature | Spectral convergence, at roundoff by nv = 24 on [−8, 8]. |
| U2 | Mieussens solve | ≤ 3 Newton iterations at nv ≥ 12 (5- and 7-constraint, CGS non-cubic box); recomputed moments ≤ 2e-13 (solver tol 1e-13 plus summation roundoff). Fallback and invalid-input paths tested. |
| U3 | SL weights | Linear = upwind and quadratic = Lax–Wendroff to 1e-15; periodic conservation; exact one-cell shifts; limiter bounds; 2D tensor product. |
| U4 | Relaxation | Invariants ≤ 1e-13 for νh up to 50; node-wise decay ≤ 1e-15; semigroup property; anisotropy e^{−νt} to 1e-12. Rational (DIRK) step satisfies the implicit stage equation to 1e-12 and conserves, also for c > 1. |
| R1 | 0D relaxation | Both integrators: drift ≤ 6e-14, H non-increasing. Strang decays as e^{−νt} to 3e-14; sl_dirk2 matches its discrete factor ∏(1−c₁)(1−c₂) to 2.5e-14 and differs from e^{−νt} by 1.4e-5 (the comparator separates them). Control drift 4e-2. The history file uses `data_format=%.17e`. |
| R2 | Sod, small Kn (linear SL) | L1 vs exact Euler falls with ν (3.2e-2, 1.6e-2, 1.3e-2 at ν = 1e2, 1e3, 1e4) and with mesh at ν = 1e5 (2.0e-2, 1.3e-2, 8.4e-3; rate ≈ 0.6). Shock within 2 cells; mass/energy drift 3e-11 / 1.1e-10 with sl_dirk2 (bound 5e-10; set by eq_tol = 1e-13, 3e-12 at eq_tol = 1e-15). Strang: 1.4e-11 / 5e-11. |
| R3 | Sod, free-molecular (ν = 0) | Errors fall with refinement, quadratic below linear. A per-node upwind emulation in Python reproduces the code's linear errors to 3 digits. |
| R3b | Smooth free streaming | Order 2.00 (quadratic, unlimited) and 0.87–0.92 (linear; bound 0.85). Drift ≤ 2e-15; merged vs unmerged half steps within 2%. |
| R4 | Convergence / OQ1 | See finding (b). The velocity-space part is covered by U1/U2. |
| R5 | MPI, 1 vs 4 ranks | Bitwise on all dumps and the restart f; history to 4e-15 (reduction order). |
| R6 | Restart round-trip at Nv = 32³ | f and derived moments bitwise after restart; OQ2 settled. |
| R7 | Couette / heat conduction | μ/μ_BGK = 0.9989, κ/κ_BGK = 0.997, P_xy uniform to 3e-4, slip length 1.08 l, temperature-jump length 1.10 l. Unchanged with 2× mesh and 1.5× velocity grid. The ν = 0 control gives a flat interior, as expected. |
| R8 | Non-interference | Hydro bitwise identical with kinetics on vs off, all 4 dumps. |
| R10 | Stiff time order | Freestream wave, ν = 1e5, νh = 8–800, quadratic unlimited SL, dt = 0.64Δx forced, 32→256 cells: sl_dirk2 order 1.98, 1.99; strang 1.06, 0.98 (control); drift ≤ 7e-12. |
| R9 | Resolution diagnostics | Narrow box: edge-mass abort. 4³ grid: fallback abort. [−6, 6]³: warning only. Resolved: silent. |
| BCs | `walls.py` (step 5) | Specular box: mass 1e-13, energy 1e-12. Diffuse box: mass 1e-13, exchanges energy. Isothermal walls leave a uniform gas unchanged to 4e-16; hot walls heat it (control). Outflow mass loss equals hydro's to 13 digits. |

Each pass/fail comparator was first shown to report a failure on a deliberately perturbed
input. Initial kinetic moments match the hydro state to 5e-14 (OQ3: `PostInitialization`
runs after the problem generator and the derived fills).

## Findings for reviewers

a. **Quadratic SL with the minmax limiter breaks conservation.** The limiter
   is not in flux form. Unlimited SL with the ≤ 1-cell cap is algebraically upwind or
   Lax–Wendroff, so it conserves exactly on periodic domains. Controlled pairs, varying
   only the limiter:
   - Smooth wave at ν = 0: mass drift 1.2e-4 with minmax vs 5e-15 unlimited (but unlimited
     gives min f < 0).
   - Sod at ν = 1e5: 0.34% mass and 0.9% energy lost, and the shock sits about 2.3% of the
     domain (about 12 cells at 512) behind the exact position. Mesh or velocity
     refinement does not fix it. Linear SL is conservative and places the shock
     correctly.
   - Unlimited quadratic at ν = 1e4 makes f negative at the Sod contact, which triggers
     the equilibrium-fallback abort.

   **Decision (user, 2026-10-02): the default is `sl_order = 1`.** A conservative
   limiter is not pursued: nonlinear local operations such as limiters do not globalize
   well in the planned tensor-train representation. Quadratic + minmax stays available as
   an option.

b. **OQ1, time order.** Strang plus subcycling (merged half steps) is globally second
   order when νh ≲ 1, including with n_sub = 4. In the stiff regime νh ≫ 1 it drops to
   first order (0.98–0.99 at n_sub = 4), whether or not subcycling is on. **Decision
   (user, 2026-10-02): add the characteristic IMEX-RK integrator `sl_dirk2` and make it
   the default.** It restores second order for any νh (R10). Trade-offs, accepted by the
   user: 1.41× the SL work, relaxations and ghost exchanges per unit time; for νh > 2.41
   the first stage over-relaxes (c₁ > 1), so f can go slightly negative (Sod at ν = 1e5,
   512 cells: negative mass 1.9e-12 of the total, `kinetics_sums_6`; strang stays ≥ 0;
   conservation unaffected). With the
   default `sl_order = 1` the spatial error dominates and the time gain is not visible.

c. **Restart cost of f.** At Nv = 32³ the HDF5 restart works (32,778 component labels),
   but with default settings each dump takes about 6 s, dominated by the XDMF sidecar
   (66 MB) and gzip. The decks set `write_xdmf = false` and `hdf5_compression_level = 0`
   on the restart output block. With that, the 40-cycle Sod run with 3 restart dumps
   takes 0.9 s instead of 20.8 s. A storage form for f in 2D/3D is tracked as OQ4.

d. **Hydro restarts only to roundoff (not caused by this change).** A control run with
   kinetics off shows the same 5e-16 hydro differences after restart. Kinetic substeps
   follow the hydro dt, so R6 forces dt (`parthenon/time/dt_force`) and requires bitwise
   agreement only for the kinetics variables.

e. **R5 MPI launcher.** On machines where an MPICH `mpiexec` comes first on PATH while
   RIOT links Open MPI, R5 runs 4 serial copies and reports a history-shape mismatch.
   Set `RIOT_MPI_COMMAND` to Open MPI's `mpiexec` (e.g.
   `/opt/homebrew/opt/open-mpi/bin/mpiexec`).

f. **`script/format.sh` only formats files tracked at HEAD.** None of the new files had
   been `git add`-ed, so the script skipped them. The copyright headers and AI disclaimers
   were added by hand. Rerun `CFM=clang-format-20 ./script/format.sh` after `git add` and
   before merging.

### Kokkos portability reviews

The `kokkos-portability-reviewer` ran after each step, and its fixes are in this change
set:

- a function-local struct in a test replaced (nvcc cannot use it in device lambdas);
- `if constexpr` in the equilibrium expansion, and reuse of the line-search buffer;
- `ForceCapture` for variables used only inside `if constexpr` branches of BC lambdas
  (nvcc capture);
- the MPI collective in `PostInitialization` skipped on ranks with no blocks (it hung),
  and the order of collectives made consistent across ranks;
- Parthenon's ndim rule followed in the BCs;
- the zero-velocity node of odd-sized grids treated as outgoing at walls.

These reviews are static. Nothing was compiled or run on a GPU.

## AI disclosure

Claude (Anthropic) helped write this change. It wrote the design document, the
implementation, the tests, the docs page and this file, under the direction and review of
the author. Every new source, test, deck and docs file has a generative-AI disclaimer at
the top, per the ML-disclaimer rule in the PR template. The one exception is
`tst/scripts/kinetics/__init__.py`, which has only the license header and no code.
