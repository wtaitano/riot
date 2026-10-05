# Kinetics S2: 2D-3V tensor-train run at the dense memory limit

> **Filename note:** no PR number existed when this file was written. Rename it to
> `plan_histories/<PR-number>.md` when the PR is opened.

S2 runs a verified 2D-3V tensor-train (TT) kinetic BGK case at a size where a dense f is at
the memory limit of a 32 GB machine. S2 adds **no solver code**. It adds one small
regression test, a docs paragraph and the study scripts and results.

Scope: the uncommitted changes on `taitano/vlasov` on top of the closure coupling
(`1c0aece`). The design and decision log is `claude_sessions/kinetic_bgk/S2_DESIGN.md`;
the decision numbers S2-Qn below refer to it. All measured numbers below come from
`claude_sessions/kinetic_bgk/s2/RESULTS.md`.

## Motivation

S1 and the closure stage established that TT matches dense at small sizes. The case for
TT is memory. S2 shows a consistent TT solution at a size where dense f is at the memory
limit of the development machine. It makes no physics or speed claim (S2-Q1).

Out of scope: running dense at production size, a mesh-convergence ladder,
variable-size TT storage (OQ4), the core-order study, GPU and OpenMP backends.

## Design decisions

- **Goal is memory (S2-Q1, S2-Q16).** 64² cells × Nv 64³. Counted, not run: dense f is
  64² × 64³ × 8 B = 8 GiB interior, 12.5 GiB per copy with 2 ghost layers, and ≥ 2 copies
  + hydro ≈ 25–27 GiB. That is at the limit of the 32 GB machine, not clearly beyond it.
  The 25 GiB quoted during design was 2 copies, not one. The user kept this size after
  the recount.
- **Problem (S2-Q2, S2-Q11).** `inputs/kinetics/blast2d.py` (drifting pressure pulse,
  periodic), changed only through command-line overrides: cubic velocity box ±9.5 on
  every axis with equal spacings, ν = 1e2, t = 0.1, 16² meshblocks (16 blocks, 2 per
  rank), restart output off. The deck file is unchanged.
- **Runtime (S2-Q3/Q5/Q7).** 8 MPI ranks, Serial Kokkos backend. No serial runs (too
  slow). **Dense is not run at production size** (user decision).
- **Closure coupling on (S2-Q6).** The default from the closure stage.
- **Reference (S2-Q4).** TT at `tt_eps = 1e-8` compared with TT at `tt_eps = 1e-12`, at
  64² only. No mesh ladder and no self-convergence study.
- **Rank cap (S2-Q9, S2-Q17).** `tt_rank_max = 24`. Partway through, the `tt_eps = 1e-12`
  reference was seen to hit this cap. Both cap-24 runs
  were allowed to finish, then `tt_eps = 1e-12` was rerun at `tt_rank_max = 32`.
  `tt_eps = 1e-8` is compared against both references, and cap 24 is compared with cap 32.
- **Pass criterion (S2-Q10).** The max relative difference between `tt_eps` 1e-8 and
  1e-12 must be ≤ 1e-4 in kinetic ρ, u, T, Π, q and coupled hydro ρ, u, T. Also reported:
  kinetic and hydro mass/energy drift, max/mean/p95 rank, and cap hits.
- **Dense tie-in (S2-Q8/Q14).** One coarse run at 16² cells and the production Nv 64³:
  TT at `tt_eps = 1e-14` compared with dense.
- **Deliverables (S2-Q12/Q15).** The `s2/` scripts and `RESULTS.md`, a docs paragraph,
  and a new small regression (below).
- **Deferred (S2-Q13).** OQ4 (variable-size TT storage) and the core-order study.

## Proposed changes

### Tests

- `tst/scripts/kinetics/tt_closure2d.py` (new): `blast2d` deck at 16² cells (8²
  meshblocks), cubic box ±9.5 with 16³ velocity nodes, closure coupling on, 2 MPI ranks,
  forced dt 4e-3, `tt_rank_max = 24`, `tt_diag = exact`. It runs dense, TT
  `tt_eps = 1e-14` and TT `tt_eps = 1e-8`, then compares final dumps with a max relative
  difference. Each field is scaled by its own max, except that stress and heat flux are
  scaled by the max stress and velocities by the max kinetic velocity. Fields: kinetic ρ,
  u, T, stress, heat flux and hydro ρ, u, T. The file starts with a generative-AI
  disclaimer and the Triad copyright header.

### Docs

- `doc/sphinx/src/packages/kinetics.rst`: new paragraph "2D-3V at the dense memory limit"
  after the S1 scaling results. It covers the production setup, the dense memory count,
  TT peak memory per rank, the `tt_eps` 1e-8 vs 1e-12 agreement and wall time, the cap-24
  vs cap-32 difference, and the 16² dense tie-in.

### Study artifacts (not part of the PR tree)

- `claude_sessions/kinetic_bgk/S2_DESIGN.md`: decision log.
- `claude_sessions/kinetic_bgk/s2/`: `compare.py` (comparator), `analyze.py`, `tiein/`
  and `production/` run scripts and outputs, and `RESULTS.md`.

### Solver code

- None.

## Testing

### New regression: `tst/scripts/kinetics/tt_closure2d.py`

- **TT vs dense.** `tt_eps = 1e-14` must match dense to ≤ 1e-10. Measured 1.2e-12.
- **Control.** `tt_eps = 1e-8` must differ from dense by > 1e-9. Measured 6.4e-7.
- **Instrument.** The first and final dense dumps must differ by > 1e-2, so the
  comparator is shown to see the evolution.
- Run time about 30 s.

### Studies (from `s2/RESULTS.md`)

**Dense tie-in** (16² cells, 4 blocks of 8², 4 ranks, Nv 64³, forced dt 4e-3, 25 cycles,
`tt_rank_max = 24`):

| | wall | peak RSS / rank |
|---|---|---|
| dense | 210 s | 2.59 GB |
| TT `tt_eps` 1e-14 | 280 s | 0.56 GB |

TT vs dense, max 2.4e-12 (kinetic ρ 2.3e-12, u 1.6e-12, T 6.3e-13, Π 2.4e-12, q 1.2e-12;
hydro ρ 1.9e-13, u 2.3e-13, T 3.4e-13). Final TT ranks: mean r1 23.0, r2 9.4; 94 of 256
cells at the cap; 7 cap hits over the run.

**Production** (64² cells, 16 blocks of 16², 8 ranks, Nv 64³, forced dt 2e-3, 50 cycles,
TT only):

| | eps 1e-8, cap 24 | eps 1e-12, cap 24 | eps 1e-12, cap 32 |
|---|---|---|---|
| wall (8 ranks) | 725 s | 3,270 s | unreliable (see limitations) |
| peak RSS / rank | 1.55 GB | 1.40 GB | 2.09 GB |
| mean rank (r1, r2), final | 11.0, 4.5 | 22.6, 7.4 | 23.4, 7.4 |
| p95 (r1, r2) / max rank | 15, 5 / 17 | 24, 8 / 24 | 27, 8 / 28 |
| cells at cap, final | 0 | 2,329 of 4,096 | 0 |
| cap hits, summed | 0 | 38,200 | 0 |
| kinetic mass / energy drift | 2.2e-8 / 2.9e-7 | 1.8e-12 / 4.3e-11 | 7.9e-13 / 2.2e-11 |
| hydro mass / energy drift | 5.9e-16 / 1.1e-14 | 5.9e-16 / 1.1e-14 | 5.9e-16 / 1.1e-14 |

- `tt_eps` 1e-8 vs 1e-12 cap 32: max **8.1e-7** (kinetic ρ, u, T, Π, q: 2.5e-7, 3.3e-7,
  8.1e-7, 7.0e-7, 4.1e-7; hydro ρ, u, T: 2.5e-7, 3.0e-7, 4.2e-7).
- `tt_eps` 1e-8 vs 1e-12 cap 24: max 8.1e-7 (same to 3 digits).
- `tt_eps` 1e-12, cap 24 vs cap 32: max 2.1e-10.
- Instrument: the first and final dumps of the same run differ by 2.1 (relative).
- **S2-Q10 pass (≤ 1e-4): yes, 8.1e-7.** Cap 24 pins the `tt_eps = 1e-12` run in 57 % of
  cells but changes the solution by only 2.1e-10 against cap 32, about 4,000× below the
  `tt_eps = 1e-8` difference. `tt_eps = 1e-8` never reaches cap 24.
- Memory: TT at cap 32 peaks at about 17 GB over 8 ranks (2.09 GB × 8, RSS including
  hydro, the three f registers k0/k1/k2 and MPI buffers). For dense, f alone is about
  25–27 GiB.

### Verification status

All runs were on CPU (Serial Kokkos + Open MPI). **No GPU runs were made.**

- Full kinetics suite: 20/20 pass (2026-10-04, incl. the new `tt_closure2d`). Unit tests: ctest 47/47 at `1c0aece`; S2 adds no C++.

## Known limitations and open items

- **Dense not run at production size** (user decision, S2-Q3/Q16). The dense memory
  figure is a count, not a measurement. TT-vs-dense agreement at production Nv rests on
  the 16² tie-in and the earlier S1 and closure-stage tests.
- **Dense at the memory limit, not clearly beyond it.** About 25–27 GiB for a 32 GB
  machine (S2-Q16).
- **Cap-32 wall time is unreliable.** By mistake, two identical copies of the cap-32 run
  ran at the same time: a chained "wait, then launch" background job outlived its tool
  timeout. Both had the same problem id and appended to the same history file. Every row
  pair was bitwise identical (the runs are deterministic), and the duplicates were
  removed. The wall times (5,840 s and 6,110 s) were measured with two 8-rank jobs on 10
  cores, so they are not quoted as the cap-32 cost. The user chose to record them as
  unreliable without a rerun. Peak RSS is per process and is not affected.
- **No mesh convergence.** The production reference is `tt_eps` 1e-8 vs 1e-12 at one
  resolution (S2-Q4).
- **Deferred (S2-Q13).** OQ4 (variable-size TT storage) and the core-order study.
- **Backends.** Serial Kokkos + MPI only. No OpenMP or GPU runs.

## AI disclosure

Claude (Anthropic) helped write this change: the design log, the study scripts, the
regression test, the docs paragraph and this file, under the direction and review of the
author. As the ML-disclaimer rule in the PR template requires, the new test file
`tt_closure2d.py` starts with a generative-AI disclaimer.
