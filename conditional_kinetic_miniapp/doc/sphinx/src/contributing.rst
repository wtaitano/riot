..
   This file was made in part with generative AI.

Contributing
============

ckm follows riot's `contributing guide
<https://github.com/lanl/riot/blob/main/doc/sphinx/src/contributing.rst>`_. That
covers the pull request protocol, code review expectations, formatting, and the
AI-assisted coding policy. This page lists only how ckm **differs** from riot. If
something is not listed here, riot's rule applies.

Differences from riot
---------------------

* **Isolation is one-way.**

  * Riot never depends on ckm. Files outside ``conditional_kinetic_miniapp/`` never
    reference ckm, except the files that wire it in: ``.gitmodules``,
    ``CMakeLists.txt`` (the off-by-default ``RIOT_BUILD_CKM`` option),
    ``.github/CODEOWNERS``, and ``.github/workflows/ckm.yml``.
  * ckm may use riot's ``src/`` utilities and build targets. It does not reuse
    riot's physics packages; the ckm team writes its own.
  * ``script/check_boundary.sh`` enforces this in CI.

* **Dependencies.** ckm uses riot's ``external/`` and has no submodules of its own.
  On the feature branch, riot's ``external/parthenon`` points at the tensor-train
  branch of Parthenon.
* **Build.** ckm is built as part of riot with ``-DRIOT_BUILD_CKM=ON``. HDF5 is
  required. See :doc:`building`.
* **Namespace.** All ckm code is in namespace ``ckm``.
* **Plan histories.**

  * Agentic-AI plan files go in ``conditional_kinetic_miniapp/plan_histories/``.
  * They are named ``ckm-fork-pr-<N>.md`` after the internal fork PR number, not
    after a ``lanl/riot`` PR number.

* **Where PRs go.** ckm PRs target the ``wtaitano/ckm`` integration branch on the
  ``wtaitano/riot`` fork, not ``lanl/riot:main``.

Workflow
--------

ckm is developed on one long-lived integration branch and merged into riot as a
single PR at the end::

   lanl/riot:main                    <- one final PR (opened early as a draft)
     ^
   wtaitano/riot:wtaitano/ckm        <- integration branch (protected)
     ^  internal PRs, reviewed by the subteam
   <you>/riot:<you>/ckm-<topic>      <- your topic branches, on your own fork

One-time setup
~~~~~~~~~~~~~~

1. Fork `lanl/riot <https://github.com/lanl/riot>`_ to your GitHub account.
2. Clone your fork and configure the remotes and pull settings:

   .. code-block:: bash

      git clone git@github.com:<you>/riot.git && cd riot
      git fetch https://github.com/wtaitano/riot.git wtaitano/ckm && git switch -c wtaitano/ckm FETCH_HEAD
      conditional_kinetic_miniapp/script/setup_remotes.sh <you>
      git fetch ckm
      git submodule update --init --recursive

   This gives you the following remotes:

   * ``origin``: your fork.
   * ``ckm``: ``wtaitano/riot``, fetch only.
   * ``upstream``: ``lanl/riot``, fetch only.

   It also sets ``pull.rebase=false`` and ``pull.ff=only``.

Making a change
~~~~~~~~~~~~~~~

.. code-block:: bash

   git fetch ckm
   git switch -c <you>/ckm-<topic> ckm/wtaitano/ckm
   # ... edit, build, test, commit ...
   git push                     # goes to your fork

Open a PR from ``<you>/riot:<you>/ckm-<topic>`` into
``wtaitano/riot:wtaitano/ckm``. A PR needs these before it can merge:

* one approving review from someone other than the last pusher;
* passing ``ckm`` CI (serial, MPI, docs, and the boundary check);
* all review threads resolved.

PRs are merged with a merge commit. Squash and rebase merges are disabled.

History rules
~~~~~~~~~~~~~

* Never rebase or force-push ``wtaitano/ckm``. GitHub rejects force-pushes to it.
* Riot changes come in by **merging** ``upstream/main`` into ``wtaitano/ckm``, done by
  the maintainers through a normal PR.
* You may rebase or amend your own topic branch before it is merged. Once it is
  merged, it is shared history.

Plan histories
~~~~~~~~~~~~~~

If you used agentic AI, add a plan file to ``conditional_kinetic_miniapp/plan_histories/``
in your PR. Rename it to ``ckm-fork-pr-<N>.md`` once the PR number is known. Riot's
file-level AI disclaimer comment rule applies unchanged.

AI coding agents
----------------

``conditional_kinetic_miniapp/AGENTS.md`` is the single source of instructions for
AI coding agents (Claude Code, Codex, Copilot, Cursor, ...). ``CLAUDE.md`` only
imports it. Read it whether or not you use an agent; it is also a concise summary of
the rules on this page.

Claude Code users get shared configuration from
``conditional_kinetic_miniapp/.claude/`` on ``git pull``. It only applies when
``claude`` is **launched from** ``conditional_kinetic_miniapp/``, because project
settings are not inherited from parent directories.

=================================  ==================================================
Item                               What it does
=================================  ==================================================
``settings.json`` plugins          Enables ``mattpocock-skills`` from the official
                                   marketplace (installed on first trusted launch)
``settings.json`` permissions      Allows build/test/format and local git; asks
                                   before ``git push``, ``merge``, ``rebase``,
                                   ``reset``; denies pushes to ``upstream``,
                                   force-pushes, and reading credential files
``hooks/guard_edit.sh``            Blocks edits outside the miniapp, except the
                                   four riot wiring files
``hooks/guard_git.sh``             Blocks pushes to ``lanl/riot``, force-pushes, and
                                   rebase/reset/delete of ``wtaitano/ckm`` or ``main``
``agents/ckm-implementer.md``      Implements a scoped change test-first, builds and
                                   tests serial and MPI, formats, writes the plan file
``agents/ckm-reviewer.md``         Read-only review: correctness, tensor-train
                                   semantics, isolation, conventions, AI policy
``skills/ckm-*``                   ``ckm-build``, ``ckm-test``, ``ckm-format``,
                                   ``ckm-plan-history``
=================================  ==================================================

When you first launch ``claude`` in the miniapp, accept the workspace-trust prompt so
the shared settings take effect. The hooks run small shell scripts from the repo.
Personal preferences (model, theme, extra plugins) go in your own
``.claude/settings.local.json``, which is git-ignored.

The hooks and permissions are guardrails for agents, not security. The GitHub
rulesets on the fork (see :doc:`maintainers`) are what actually protect shared
history. Agents commit locally; a human pushes and opens PRs.
