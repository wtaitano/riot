# Plan: ckm shared developer environment (second internal PR, stacked on #1)

Generated with Claude Code (agentic AI). Rename to `ckm-fork-pr-<N>.md` once the
internal PR number is known.

## Goal

Give everyone on the ckm subteam the same agent behavior and the same guardrails when
they `git pull`, without adding anything AI-related at the riot root.

## Decisions (agreed with the ckm lead)

- **One source of instructions.** `AGENTS.md` holds the instructions for every agent
  (Claude Code, Codex, Copilot, Cursor). `CLAUDE.md` is only `@AGENTS.md`. Teammates
  on other tools rely on `AGENTS.md` and CI.
- **Launch directory.** Everyone launches `claude` from `conditional_kinetic_miniapp/`.
  Project settings are not inherited from parent directories, so the shared
  `.claude/` only applies there. There is no root `.claude/`, `CLAUDE.md`, or
  `AGENTS.md`.
- **Three layers of config.**
  1. Third-party plugins through `enabledPlugins`: `mattpocock-skills` from
     `claude-plugins-official`. It is unpinned, and the version drift that comes with
     that is accepted.
  2. In-house agents, skills, and hooks committed as files.
  3. Personal settings in git-ignored `settings.local.json`.
- **Agents.**
  - `ckm-implementer`: works test-first, builds and tests serial and MPI, formats,
    writes the plan file, and commits locally only. No model is pinned.
  - `ckm-reviewer`: read-only.
- **Skills.** `ckm-setup`, `ckm-build`, `ckm-test`, `ckm-format`, `ckm-plan-history`.
- **Hooks.**
  - `guard_edit.sh` allows only the miniapp plus `.gitmodules`, `.github/CODEOWNERS`,
    and `.github/workflows/ckm.yml`.
  - `guard_git.sh` blocks pushes to `lanl/riot`, force-pushes, and
    rebase/reset/delete of `wtaitano/ckm` or `main`. Rebasing your own topic branch
    is allowed.
- **Permissions.**
  - Allow: build, test, MPI launch, format, local git.
  - Ask: push, merge, rebase, reset.
  - Deny: push upstream, force-push, and reading credentials (`~/.ssh`,
    `~/.config/gh`, `~/.netrc`, `~/.git-credentials`).
- **No format-on-edit hook.** CI enforces formatting already.

## Contents

- `AGENTS.md`, `CLAUDE.md`, `.gitignore` (for `settings.local.json` and
  `CLAUDE.local.md`).
- `.claude/settings.json`, `.claude/hooks/{guard_edit,guard_git}.sh`.
- `.claude/agents/{ckm-implementer,ckm-reviewer}.md`.
- `.claude/skills/ckm-{setup,build,test,format,plan-history}/SKILL.md`.
- Docs: the "AI coding agents" section in `doc/sphinx/src/contributing.rst`, and a
  README section.

## Verification

- Hook scripts were unit-tested with synthetic PreToolUse JSON, invoked through
  `$CLAUDE_PROJECT_DIR`. 26 cases:
  - **Edits:** inside the miniapp, the allowed root files, and outside scratch paths
    are allowed. Riot `src/`, root CMake, other workflows, and a `../` escape are
    blocked.
  - **Git:** push to upstream or a lanl URL, `-f`, `--force`, `--force-with-lease`,
    `+refspec`, deleting the shared branch, rebase or reset onto `wtaitano/ckm`, and
    `git -C`/`-c` global-option forms are blocked. Plain push, topic-branch rebase,
    deleting a topic branch, and `git merge upstream/main` are allowed.
- `settings.json` parses as JSON. The boundary check passes. The docs build with
  `-W -n`.
- **Not yet tested:** a live Claude Code session launched from the miniapp. Plugin
  install and workspace trust need a human, interactively.
