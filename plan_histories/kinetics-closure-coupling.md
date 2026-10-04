# Kinetics: closure coupling of the kinetic stress and heat flux into hydro

> **Filename note:** no PR number existed when this file was written. Rename it to
> `plan_histories/<PR-number>.md` when the PR is opened.

Adds `kinetics/closure_coupling` (default true). The hydro momentum and energy fluxes take
the non-equilibrium stress Π and the heat flux q of the kinetic distribution f, so hydro
solves the BGK moment equations instead of the Euler equations. The coupling is one-way:
f does not see hydro. It works with both the dense and the tensor-train (TT)
representation.

Scope: the uncommitted changes on `taitano/vlasov` on top of S1 (`52fdd00`). The design
and decision log is `claude_sessions/kinetic_bgk/CLOSURE_DESIGN.md`; the decision numbers
C-Qn below refer to it, and the sanity results are in its §7 implementation log.

## Motivation

Up to S1, hydro is evolved alongside the kinetic gas but uncoupled, so in the transitional
and free-molecular regimes the hydro solution has no relation to the kinetic one. This
stage closes the hydro moment equations with Π and q taken from f. On a face with normal
d (Cartesian):

    F_mom_i += Π_id,    F_E += Π_id u_i + q_d,    Π_ij = P_ij − p δ_ij

Here P_ij = m Σ c_i c_j f w, p = trace(P)/3, and u is the hydro face velocity. The mass
flux is unchanged. The sign convention matches
`Ionization::ComputePlasmaViscousFluxes` (there σ = −Π). Π and q come from f, not from the
hydro state, so for hydro they are a prescribed forcing: this stage adds no new hydro
stability limit.

Out of scope: hydro → f feedback (S3, LoMaC), curvilinear coordinates, mesh refinement,
and fixing the equilibrium anisotropy on anisotropic velocity grids.

## Design decisions

- **One-way coupling (C-Q1).** Kinetic Π, q go into the hydro fluxes. hydro → f is S3.
- **Interpolated time level (C-Q2).** When coupled, the kinetics step runs *before* the
  hydro step. The closure is stored at t^n (`closure_old`, from f^n) and at t^{n+1}
  (`closure_new`, from f^{n+1}). RK stage s uses (1 − c_s) Π^n + c_s Π^{n+1}, so the
  coupling is second order in time. `closure_new` of step n becomes `closure_old` of step
  n+1, which means one closure evaluation per step plus one at init or restart.
- **Face values (C-Q3).** Each face uses the average of its two adjacent cells, as in
  plasma viscosity.
- **Energy-flux velocity (C-Q4).** Π · u uses the hydro face velocity
  (`c.c.bulk.face_velocity`). Hydro's `store_vf` is turned on when the coupling is on.
- **Ghost cells (C-Q5).** After the kinetics step there is one extra f ghost exchange plus
  kinetic BCs, and then Π, q are computed on the whole block, ghosts included. Walls
  therefore follow the kinetic BCs.
- **Default on (C-Q6).** `kinetics/closure_coupling` defaults to true. Tests that need
  uncoupled hydro set it to false explicitly.
- **Tests (C-Q7).** The tests cover the continuum limit, the free-molecular limit,
  conservation with TT, and the hydro-tracks-kinetic gap. The gap test is this stage's
  consistency check: the gap should go to 0 with resolution. In S3, LoMaC makes the
  kinetic moments follow hydro, so S3 will change this test to compare coupled against
  standalone kinetic runs. The four tests planned in the design are implemented as one
  script.
- **Field metadata (changed during implementation).** `closure_old` / `closure_new` are
  Cell, Derived, OneCopy and have *no* `OperatorSplit` flag. With the flag they were
  missing from hydro's `u0` register, and `AddClosureFluxes` segfaulted. Being OneCopy,
  they share base's memory. Restart data is unchanged: `closure_old` is recomputed from f
  before the loop starts.

## Proposed changes

### Kinetics package (`src/kinetics/`)

- `kinetics_closure.cpp` (new):
  - `ComputeClosure(md)`: Π (traceless, from `IsotropicTarget` / central moments) and q
    of f on the entire block, written to `closure_new`. It goes through `WithCell`, so the
    dense and TT paths are the same code.
  - `ShiftClosure(md)`: copies `closure_new` to `closure_old`.
  - `AddClosureFluxes(md, w)`: a face loop per direction (`AddClosureFluxesDir<DIR>`)
    that adds the time-interpolated, face-averaged closure to the hydro momentum and
    `total_material_energy` fluxes. Works in any ndim.
  - `BeforeLoop`: `CheckRestartLayout`, then the closure of the initial or restarted f.
- `kinetics.hpp`: new 9-component fields `closure_old` / `closure_new` (Π xx, yy, zz,
  xy, xz, yz; q x, y, z). New inline `ClosureCoupling(pin)`, which hydro, the driver and
  the package all read. New declarations for the functions above.
- `kinetics.cpp`: new param `closure_coupling`. When the coupling is on, the closure
  fields are allocated and the run is required to be Cartesian with
  `parthenon/mesh/refinement = none`. `UserWorkBeforeLoopMesh` changes from
  `CheckRestartLayout` to `BeforeLoop`.
- `kinetics_tasks.cpp`: when the coupling is on, the kinetics task collection starts with
  `ShiftClosure`. It ends with an f ghost exchange (+ BCs) followed by `ComputeClosure`.

### Driver and hydro

- `riot_driver.{hpp,cpp}`: new static `PreHydroTasks` list, which `Step()` runs before
  `RiotStepTasks` (dt is already fixed by then). New flag `kinetic_closure`. In each RK
  stage, `Kinetics::AddClosureFluxes(mu0, c[stage-1])` runs after the hydro, mix and
  plasma-viscosity fluxes and before `LoadAndSendFluxCorrections`.
- `riot.cpp`: `KineticsTasks` goes into `PreHydroTasks` when the coupling is on, and into
  `OperatorSplitTasks` (after hydro, as before) otherwise.
- `hydro/hydro.cpp`: `store_vf` also turns on when `Kinetics::ClosureCoupling(pin)` is
  true.
- `src/CMakeLists.txt`: adds `kinetics/kinetics_closure.cpp`.

### Docs

- `doc/sphinx/src/packages/kinetics.rst`: the package introduction now describes the
  default coupling. New "Closure coupling" section covering the flux formulas, step order
  and time interpolation, the Cartesian / no-refinement restrictions, a summary of the
  checks, and a note on the anisotropic velocity grid.

## Testing

### New regression: `tst/scripts/kinetics/closure_coupling.py`

Velocity grid 24³ on [−8, 8]³ for the Sod cases.

- **Gap.** Sod at ν = 0 and 1e2, nx = 64/128/256. The L1 gap between hydro and kinetic
  ρ, u, T must shrink at a rate ≥ 0.4. Control: uncoupled at ν = 0, the gap must not
  shrink (rate < 0.15). Instrument check: the gap at t = 0 is below 1e-12.
- **Continuum.** Sod at ν = 1e4 and 1e5, nx = 64, forced dt. Max |Π_xx|, max |q_x| and
  the L1 change in hydro ρ caused by the coupling must each scale like 1/ν (ratio 7–13).
- **Free-molecular.** At ν = 0, the L1 error of the coupled hydro density against the
  exact discrete-velocity free-streaming solution must decrease with the mesh.
- **Conservation + TT.** Periodic freestream deck at ν = 30, amp 0.3, 16 cells, Nv
  16×12×12, forced dt. The drift of hydro mass, momentum and energy must be < 1e-13 for
  both dense and TT. TT (`tt_eps` 1e-14) must match dense in hydro ρ to 1e-10, and the
  coupling must change hydro ρ by more than 1e-2 (instrument control).

Sanity values recorded in `CLOSURE_DESIGN.md` §7:

- Gap rates: coupled 0.51–0.68 at ν = 0 and 0.58–0.75 at ν = 1e2; uncoupled at ν = 0
  between −0.08 and 0.
- Continuum ratios: 9.4–10.0.
- Conservation: hydro drift ≤ 6e-16.
- Hydro ρ: TT vs dense 2.1e-12; coupling on vs off 8e-2.
- One-way check, forced dt, ν = 100: the kinetic fields are bitwise identical with and
  without the coupling, while hydro ρ differs by 8e-2.

### Existing regressions changed

- `non_interference.py`, `sod_continuum.py` and `walls.py` now set
  `kinetics/closure_coupling=false`, since each checks uncoupled behaviour: hydro bitwise
  unchanged with kinetics on, hydro against exact Euler, and kinetic walls with outflow
  hydro boundaries. `non_interference.py` also gains a control: with the coupling on,
  the final hydro dump must differ from the kinetics-off run.
- `tt_restart_mpi.py`: the floor in the relative history comparison goes from 1e-300 to
  1e-16. With the coupling on, hydro transverse momentum is roundoff (~1e-33, from Π_xy,
  Π_xz), and summing it in a different order on 4 ranks made the relative difference
  meaningless.

### Verification status

All runs were on CPU (serial and MPI). **No GPU runs were made.**

- Unit tests: ctest 47/47 pass.
- Kinetics regressions: 19/19 pass. 17 passed in the full run. `walls.py` and
  `tt_restart_mpi.py` failed there and passed when re-run after the fixes above.
- Kokkos portability review of the changed C++: no issues found.
- `script/format.sh`: no changes.

## Known limitations and open items

- **Cartesian coordinates only, no mesh refinement.** Both are rejected at startup
  (`PARTHENON_REQUIRE` in `kinetics.cpp`). `closure_old` carries over between steps and
  is not remeshed.
- **One-way coupling.** f does not see hydro, so hydro and kinetic moments still drift
  apart at finite resolution. This is the gap the test measures. Two-way coupling is S3.
- **Anisotropic velocity grids.** When the velocity grid has different spacings per axis,
  the discrete equilibrium is slightly anisotropic: P_xx ≠ p at 3e-3 relative for
  24×12×12 nodes (24³: 4e-15). The coupling passes this to hydro as a spurious,
  ν-independent stress. This is documented in `kinetics.rst`, and the Sod closure tests
  use equal spacings.
- **Cost.** Each step adds one f ghost exchange, and `store_vf` adds 9 face-velocity
  components.
- **Step order.** When coupled, kinetics runs before hydro. Other operator-split packages
  still run after hydro.
- **Restart.** f restarts bitwise. Hydro restarts to roundoff, as before, but the coupled
  hydro amplifies that roundoff faster: the momentum difference after a restart at
  cycle 20 is 1e-16 at cycle 21 and 1.4e-12 at cycle 40 coupled (1.4e-15 uncoupled). A
  coupled run repeated twice is bitwise identical. `restart.py` does not require hydro
  to be bitwise.
- **Behaviour change for existing users.** Runs with `physics/kinetics = true` are
  coupled by default. Set `kinetics/closure_coupling = false` to get the previous
  uncoupled behaviour.

### Bugs found and fixed during this stage

- With the `OperatorSplit` flag, the closure fields were missing from `u0`, and
  `AddClosureFluxes` segfaulted. Fixed by removing the flag.
- An early TT-vs-dense conservation run showed a 1.7e-2 difference. The cause was a setup
  error: `sl_order=1` had been passed to the TT run only. It was not a TT effect.

## AI disclosure

Claude (Anthropic) helped write this change: the design log, the implementation, the
tests, the docs and this file, under the direction and review of the author. As the
ML-disclaimer rule in the PR template requires, the new source and test files
(`kinetics_closure.cpp`, `closure_coupling.py`) start with a generative-AI disclaimer.
