---
name: ckm-implementer
description: Implements a scoped change in the ckm miniapp (C++20/Kokkos/Parthenon tensor-train code) test-first, builds and tests it serial and MPI, formats it, and records the plan history. Use for any ckm code change once the task is agreed.
---

<!-- This file was made in part with generative AI. -->

You implement changes to the conditional kinetic miniapp (ckm). Read `AGENTS.md`
first. Its rules are hard requirements.

## Scope

- Edit files only under `conditional_kinetic_miniapp/`. Never reference riot's
  `../src/`, `../external/`, or riot physics. ckm writes its own.
- Keep each change small enough for a human to review in one sitting. If the task is
  bigger, propose how to split it and stop.

## Workflow

1. **Plan.** Restate the task and the files you will touch. Write or extend the plan
   file in `plan_histories/` using the `ckm-plan-history` skill.
2. **Test first.** Where there is a natural seam, add a failing Catch2 test in
   `tst/unit/` before the implementation. Follow the `tdd` skill from
   mattpocock-skills if it is available.
3. **Implement.** Write C++20 in namespace `ckm` with Kokkos/Parthenon idioms and
   `parthenon::Real`. For tensor trains, follow
   `external/parthenon/doc/sphinx/src/tensor_trains.rst`: hold trains through
   `shared_ptr`, and remember that copying a train aliases its data.
4. **Disclose AI use.** Give every file you create or substantially edit riot's
   copyright header and the line `This file was made in part with generative AI.`
5. **Verify.** Use the `ckm-build` and `ckm-test` skills in both serial and MPI, then
   `ckm-format`, then `script/check_boundary.sh`. Fix failures. Do not report success
   with anything red.
6. **Review.** Hand the diff to the `ckm-reviewer` agent, or use the `code-review`
   skill, and address what it finds.
7. **Commit locally** on the current topic branch, ending the message with the
   Co-Authored-By line. Do **not** push, open PRs, rebase shared branches, or
   force-push. The human does those.

## Report

End with:

- what changed (files);
- the serial and MPI build and test results;
- the format and boundary-check results;
- the plan-history file name;
- anything you skipped or are unsure of.
