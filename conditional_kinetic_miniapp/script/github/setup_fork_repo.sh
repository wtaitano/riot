#!/bin/bash
# ========================================================================================
#  (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#
#  This program was produced under U.S. Government contract 89233218CNA000001 for Los
#  Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
#  for the U.S. Department of Energy/National Nuclear Security Administration. All rights
#  in the program are reserved by Triad National Security, LLC, and the U.S. Department
#  of Energy/National Nuclear Security Administration. The Government is granted for
#  itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
#  license in this material to reproduce, prepare derivative works, distribute copies to
#  the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================
# This file was made in part with generative AI.
#
# Lock down the ckm integration fork on GitHub. Run once by the fork owner (and
# again any time to re-apply). Requires the gh CLI, logged in as the owner
# (`gh auth login`). Idempotent.
#
# What it does:
#  * Repo merge settings: merge commits only (squash and rebase merging off),
#    delete head branches after merge.
#  * Enables GitHub Actions on the fork.
#  * Creates/updates ruleset "ckm-protect" on the integration branch: no
#    deletion, no force-push (non_fast_forward), changes only via PR with 1
#    approval from someone other than the last pusher, merge commits only,
#    ckm CI must pass.
#  * Creates/updates ruleset "main-no-rewrite" on the fork's main: no deletion,
#    no force-push. Fast-forward syncs from lanl/riot stay allowed.
#  Both apply to everyone, including the owner (no bypass actors).
#
# Usage: setup_fork_repo.sh [owner/repo] [integration-branch]
#   defaults: wtaitano/riot wtaitano/ckm
set -euo pipefail

REPO=${1:-wtaitano/riot}
BRANCH=${2:-wtaitano/ckm}
# Must match the check names produced by .github/workflows/ckm.yml.
REQUIRED_CHECKS=("ckm (serial)" "ckm (mpi)" "ckm boundary" "ckm docs")

command -v gh > /dev/null || {
  echo "gh CLI not found. Install it and run 'gh auth login' first." >&2
  exit 1
}
gh auth status > /dev/null

# upsert_ruleset <name> <json>: create the named ruleset or replace it in place.
upsert_ruleset() {
  local name=$1 json=$2 id
  id=$(gh api "repos/${REPO}/rulesets" --jq ".[] | select(.name==\"${name}\") | .id")
  if [ -n "${id}" ]; then
    echo "${json}" | gh api -X PUT "repos/${REPO}/rulesets/${id}" --input - > /dev/null
    echo "   updated ruleset ${name} (id ${id})"
  else
    echo "${json}" | gh api -X POST "repos/${REPO}/rulesets" --input - > /dev/null
    echo "   created ruleset ${name}"
  fi
}

echo "== Merge settings on ${REPO}"
gh api -X PATCH "repos/${REPO}" \
  -F allow_merge_commit=true \
  -F allow_squash_merge=false \
  -F allow_rebase_merge=false \
  -F delete_branch_on_merge=true > /dev/null

echo "== Enabling GitHub Actions on ${REPO}"
gh api -X PUT "repos/${REPO}/actions/permissions" \
  -F enabled=true -f allowed_actions=all > /dev/null

echo "== Ruleset ckm-protect on refs/heads/${BRANCH}"
checks_json=$(printf '{"context":"%s"},' "${REQUIRED_CHECKS[@]}")
checks_json="[${checks_json%,}]"
upsert_ruleset ckm-protect "{
  \"name\": \"ckm-protect\",
  \"target\": \"branch\",
  \"enforcement\": \"active\",
  \"bypass_actors\": [],
  \"conditions\": {\"ref_name\": {\"include\": [\"refs/heads/${BRANCH}\"], \"exclude\": []}},
  \"rules\": [
    {\"type\": \"deletion\"},
    {\"type\": \"non_fast_forward\"},
    {\"type\": \"pull_request\",
     \"parameters\": {
       \"required_approving_review_count\": 1,
       \"dismiss_stale_reviews_on_push\": true,
       \"require_code_owner_review\": false,
       \"require_last_push_approval\": true,
       \"required_review_thread_resolution\": true,
       \"allowed_merge_methods\": [\"merge\"]
     }},
    {\"type\": \"required_status_checks\",
     \"parameters\": {
       \"strict_required_status_checks_policy\": false,
       \"required_status_checks\": ${checks_json}
     }}
  ]
}"

echo "== Ruleset main-no-rewrite on refs/heads/main"
upsert_ruleset main-no-rewrite "{
  \"name\": \"main-no-rewrite\",
  \"target\": \"branch\",
  \"enforcement\": \"active\",
  \"bypass_actors\": [],
  \"conditions\": {\"ref_name\": {\"include\": [\"refs/heads/main\"], \"exclude\": []}},
  \"rules\": [
    {\"type\": \"deletion\"},
    {\"type\": \"non_fast_forward\"}
  ]
}"

echo "== Current state"
gh api "repos/${REPO}" --jq \
  '{allow_merge_commit, allow_squash_merge, allow_rebase_merge, delete_branch_on_merge}'
gh api "repos/${REPO}/rulesets" --jq '.[] | {id, name, enforcement}'
echo "Done."
