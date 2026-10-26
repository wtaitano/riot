# Plan: shared `project-map` agent (fourth sub-PR, stacked on #3)

Generated with Claude Code (agentic AI). Rename to `ckm-fork-pr-<N>.md` once the
internal PR number is known.

## Goal

Give every teammate who launches `claude` from the miniapp a subagent that draws one
self-contained HTML map of where ckm stands: the main parts (done / in progress / not
started / stuck, with what a stuck part waits on), the milestones, the items left
before the next one, a suggested next step, what changed since the last map, and the
decisions waiting on a human, each with a default.

## Decisions (agreed with the ckm lead)

- **Agent:** `.claude/agents/project-map.md`, `model: opus`, `effort: medium`,
  `memory: user`. It preloads the design skills `dataviz` and
  `frontend-design:frontend-design`; Claude Code skips a preloaded skill that is not
  installed, so a missing plugin does not break the agent.
- **Plugin:** `settings.json` enables `frontend-design@claude-plugins-official` for
  the team, next to `mattpocock-skills`.
- **Style lives in user memory** (`~/.claude/agent-memory/project-map/`), so each
  person is asked once ("dark or light, and one accent color?"), and a style saved by
  a `dashboard-builder` agent is reused. Subagents cannot ask questions (Claude Code
  removes `AskUserQuestion` from them), so the agent replies `STYLE_NEEDED` and the
  main session asks and reruns it.
- **Writes are limited** to `.project-map/`, the miniapp `.gitignore` and the agent's
  memory by a frontmatter `PreToolUse` hook, `.claude/hooks/guard_project_map.sh`. It
  runs only while the agent runs, fails closed, and resolves paths with `realpath`.
  Like the other frontmatter hooks, it runs only after the workspace-trust prompt is
  accepted. The agent is told never to write with Bash.
- **The map is personal:** `/.project-map/` is git-ignored. Milestones live in
  `.project-map/milestones.md`, which the user owns; the agent only adds under
  `## Proposed`, and drafts a first version from the README, plan histories, PR stack
  and history when there is none.
- **Read-only inputs:** code, docs, plan histories, local git, and the public GitHub
  REST API for the sub-PR stack and `ckm` issues (no `gh` login needed; api.github.com
  is already in the sandbox network allowlist). It never builds or runs tests on the
  shared front end.
- **Panels are chosen per run** from what ckm has; the agent records its part names
  and panels in its memory so maps keep the same shape.
- **Standing rules** for the main session (draw a map in the background before long
  autonomous stretches, update after each milestone, answer "where are we?" from the
  map, follow its next step, put decisions in the map and go ahead with the default)
  go in a committed `.claude/rules/project-map.md`, so teammates get them too. Push,
  PRs, deleting and messaging others still go to a human.

## Contents

- `.claude/agents/project-map.md` (new)
- `.claude/hooks/guard_project_map.sh` (new, executable)
- `.claude/rules/project-map.md` (new; written by the human, see below)
- `.claude/settings.json`: enable `frontend-design`; allow
  `Bash(curl -s -G https://api.github.com/repos/*)` (written by the human)
- `.gitignore`: `/.project-map/`
- `AGENTS.md` pointers and the `.claude/` table in `doc/sphinx/src/contributing.rst`

The rules file and the `curl` permission were added by the human: Claude Code's
auto-mode permission check refuses an agent widening its own permissions or writing
standing instructions for future agents.

## Verification

- The hook was fed paths: `.project-map/**`, the miniapp `.gitignore` and the memory
  folder pass; other miniapp files, `.project-map/../AGENTS.md`, the riot root
  `.gitignore`, a relative path, empty input and invalid JSON are blocked (exit 2).
- `script/check_boundary.sh` passes. No C++ changed, so there is no build or test
  change.
- Live check, after a restart in a trusted session: run the agent once and confirm it
  writes only `.project-map/` and its memory.
