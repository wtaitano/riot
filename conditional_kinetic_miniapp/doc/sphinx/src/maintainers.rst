..
   This file was made in part with generative AI.

Maintainers
===========

This page is for the owner of the ``wtaitano/riot`` fork and the co-maintainer. It
covers setting up GitHub so contributors cannot push to the integration branch or
rewrite its history.

Access model
------------

* **Owner** (``wtaitano``): administers the fork and merges PRs.
* **Co-maintainer**: the only other collaborator, with write access. The co-maintainer
  reviews the owner's PRs, because GitHub does not count self-approval.
* **Everyone else** has no access to ``wtaitano/riot``. They fork ``lanl/riot`` and
  open PRs from their own fork into ``wtaitano/riot:wtaitano/ckm``. GitHub allows PRs
  between any forks of the same repository.

Personal repositories have no "triage" or "PR-only" role. Any collaborator can push
to any branch that a ruleset does not protect. Keep the collaborator list to the
co-maintainer only.

One-time GitHub setup
---------------------

Run the lockdown **before** inviting the co-maintainer or accepting the first PR.

1. Install the `GitHub CLI <https://cli.github.com>`_ and log in as the fork owner:

   .. code-block:: bash

      gh auth login

2. Create and push the integration branch, starting from ``lanl/riot:main``:

   .. code-block:: bash

      git fetch upstream
      git branch --no-track wtaitano/ckm upstream/main
      git push origin wtaitano/ckm

3. Apply the settings:

   .. code-block:: bash

      conditional_kinetic_miniapp/script/github/setup_fork_repo.sh

   The script is idempotent; re-run it to restore the settings if they drift. It does
   the following:

   * **Merge settings:** only merge commits are allowed. Squash and rebase merging are
     off, and head branches are deleted after merge.
   * **Actions:** GitHub Actions are enabled (they are off by default on forks).
   * **Ruleset** ``ckm-protect`` on ``wtaitano/ckm``:

     * no deletion and no force-push;
     * changes only through a PR, with 1 approval from someone other than the last
       pusher;
     * stale approvals are dismissed and review threads must be resolved;
     * merge method is ``merge`` only;
     * the ``ckm (serial)``, ``ckm (mpi)``, ``ckm boundary`` and ``ckm docs`` checks
       must pass.

   * **Ruleset** ``main-no-rewrite`` on ``main``: no deletion and no force-push.
     Fast-forward syncs from ``lanl/riot`` are still allowed.
   * **No bypass:** neither ruleset has bypass actors, so both apply to the owner too.

4. Add the co-maintainer: **Settings → Collaborators → Add people**.

Doing it in the web UI instead
------------------------------

If ``gh`` is not available, make the same changes by hand:

* **Settings → General → Pull Requests:**

  * Check *Allow merge commits*.
  * Uncheck *Allow squash merging* and *Allow rebase merging*.
  * Check *Automatically delete head branches*.

* **Settings → Actions → General:** select *Allow all actions and reusable
  workflows*.
* **Settings → Rules → Rulesets → New branch ruleset**, named ``ckm-protect``:

  * Enforcement *Active*, with no bypass list.
  * Target branch ``wtaitano/ckm``.
  * Enable *Restrict deletions* and *Block force pushes*.
  * *Require a pull request before merging*, with:

    * 1 approval;
    * dismiss stale approvals;
    * require approval of the most recent push;
    * require conversation resolution;
    * allowed merge methods: *Merge* only.

  * *Require status checks to pass*: ``ckm (serial)``, ``ckm (mpi)``, ``ckm docs``,
    ``ckm boundary``. These appear after ``ckm.yml`` has run once.

* A second ruleset named ``main-no-rewrite``, targeting ``main``, with *Restrict
  deletions* and *Block force pushes*.

Keeping up with riot
--------------------

Bring riot changes into ``wtaitano/ckm`` by **merging**, never rebasing. Do this
monthly and after each riot release tag:

From the owner's clone, where ``origin`` is ``wtaitano/riot``:

.. code-block:: bash

   git fetch upstream origin
   git switch -c wtaitano/ckm-sync-$(date +%Y%m%d) origin/wtaitano/ckm
   git merge upstream/main
   git push

Then open a PR into ``wtaitano/ckm`` like any other change. Because ckm is isolated,
conflicts should only appear in ``.gitmodules``, ``.github/CODEOWNERS``, or
``.github/workflows/ckm.yml``.

Bumping the Parthenon pin
-------------------------

.. code-block:: bash

   cd conditional_kinetic_miniapp/external/parthenon
   git fetch origin <tag-or-sha> && git checkout <tag-or-sha>
   git submodule update --init --recursive
   cd - && git add conditional_kinetic_miniapp/external/parthenon

Build and run the tests in both serial and MPI before opening the PR. Prefer a tag in
``parthenon-hpc-lab/parthenon``. Commits on a PR branch can disappear if the branch is
force-pushed.

The final PR into lanl/riot
---------------------------

* Open ``wtaitano/riot:wtaitano/ckm`` → ``lanl/riot:main`` early, as a **draft**, so
  riot's developers can follow along.
* Ask for it to be merged with a **merge commit**, so the reviewed internal PRs stay
  as separate steps in riot's history.
* In the description, list the ``plan_histories/ckm-fork-pr-*.md`` files and the
  internal PRs they belong to.
