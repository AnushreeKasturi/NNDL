#!/usr/bin/env bash
# Rebuild the dashboard from the current runs and publish it to GitHub Pages.
#
# The page is generated, so publishing is: regenerate, then replace the single
# file on the gh-pages branch. No working tree is touched — the branch is built
# with git plumbing, so this is safe to run from a dirty checkout.
set -euo pipefail

cd "$(dirname "$0")/.."
RUN_DIR="${1:-outputs/cnn}"
REMOTE="${REMOTE:-origin}"

echo "building from $RUN_DIR"
python -m legal_risk_classifier.export_dashboard --run_dir "$RUN_DIR"

blob=$(git hash-object -w dashboard/index.html)
empty=$(printf '' | git hash-object -w --stdin)
tree=$(printf "100644 blob %s\tindex.html\n100644 blob %s\t.nojekyll\n" "$blob" "$empty" | git mktree)
commit=$(printf 'Publish the results dashboard\n\nGenerated from %s at %s.\n' \
    "$RUN_DIR" "$(date -u '+%Y-%m-%d %H:%M UTC')" | git commit-tree "$tree")

git push -q "$REMOTE" "${commit}:refs/heads/gh-pages" --force
echo "published -> https://anushreekasturi.github.io/NNDL/"
