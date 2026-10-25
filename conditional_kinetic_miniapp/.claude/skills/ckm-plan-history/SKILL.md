---
name: ckm-plan-history
description: Create or update the agentic-AI plan file for the current ckm change in plan_histories/, and rename it to ckm-fork-pr-<N>.md once the internal PR number is known. Required by riot's AI policy for any agent-driven change.
---

<!-- This file was made in part with generative AI. -->

Riot's AI policy (`../doc/sphinx/src/contributing.rst`, "AI-assisted coding")
requires agentic changes to leave an LLM-readable plan in a plan-history folder. ckm
keeps its own, `conditional_kinetic_miniapp/plan_histories/`.

## Create

1. Check whether this branch already has a plan file: run
   `git diff --name-only origin/wtaitano/ckm...HEAD -- plan_histories/`.
2. If not, create `plan_histories/<branch-topic>.md`. For example, branch
   `jdoe/ckm-collision-op` gives `ckm-collision-op.md`.
3. Use these sections:
   - `# Plan: <title>`, followed by `Generated with <tool> (agentic AI).`
   - `## Goal`
   - `## Decisions`, recording what the human chose and why. This keeps human choices
     separate from machine ones.
   - `## Contents`, listing the files changed.
   - `## Verification`, giving the build and test results for serial and MPI.
4. Keep the file factual and short, and update it as the work changes.

## Rename once the PR exists

Internal PRs are numbered on the `wtaitano/riot` fork. Run:

```bash
git mv plan_histories/<topic>.md plan_histories/ckm-fork-pr-<N>.md
```

Put the PR reference (`wtaitano/riot#<N>`) in the title line and commit. Pushing is up
to the human.
