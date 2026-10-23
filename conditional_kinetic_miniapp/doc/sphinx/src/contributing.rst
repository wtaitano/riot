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

* **Isolation.**

  * ckm code and build files never reference riot's ``src/``, ``external/``, or
    physics packages.
  * Files outside ``conditional_kinetic_miniapp/`` never reference ckm. The
    exceptions are ``.gitmodules``, ``.github/CODEOWNERS``, and
    ``.github/workflows/ckm.yml``.
  * ``script/check_boundary.sh`` enforces both directions in CI.

* **Dependencies.** ckm has its own Parthenon (tensor-train branch) and Catch2
  submodules under ``conditional_kinetic_miniapp/external/``. Fetch them with
  ``script/setup.sh``.
* **Build.** ckm is a standalone CMake project with ``CKM_*`` options, and HDF5 is
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
      conditional_kinetic_miniapp/script/setup_remotes.sh <you>
      git fetch ckm
      conditional_kinetic_miniapp/script/setup.sh

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
