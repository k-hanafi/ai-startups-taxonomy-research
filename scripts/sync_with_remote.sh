#!/bin/bash
# Helper script to sync local repository with cloud agent work

set -e

echo "=== AI-Native Taxonomy: Sync with Remote ==="
echo

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Check if origin remote exists
if ! git config --get remote.origin.url > /dev/null 2>&1; then
    echo "Error: No 'origin' remote configured"
    echo
    echo "Please add your remote first:"
    echo "  git remote add origin <YOUR_REPO_URL>"
    echo
    exit 1
fi

echo "Current branch:"
git branch --show-current
echo

echo "Fetching latest changes from remote..."
git fetch origin

echo

echo "Recent commits on current branch:"
git log --oneline -5
echo

echo "Cloud agent branches (cursor/*):"
git branch -r | grep 'cursor/' || echo "  (none found)"
echo

echo "Status:"
git status
echo

echo "=== Sync Options ==="
echo
echo "To pull latest changes:"
echo "  git pull origin \$(git branch --show-current)"
echo
echo "To see what changed:"
echo "  git log origin/\$(git branch --show-current)..HEAD"
echo
echo "To push your local commits:"
echo "  git push origin \$(git branch --show-current)"
echo

read -p "Pull latest changes now? (y/N): " -n 1 -r || true
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    CURRENT_BRANCH=$(git branch --show-current)
    echo "Pulling from origin/$CURRENT_BRANCH..."
    git pull origin "$CURRENT_BRANCH"
    echo "✓ Sync complete"
else
    echo "Skipped pull. Run manually when ready."
fi
