#!/usr/bin/env bash
# ralph-loop.sh — Run Claude Code in a Ralph loop
# Usage: ./ralph-loop.sh [max_iterations]
# Default: 20 iterations

set -euo pipefail

MAX_ITERATIONS=${1:-20}
ANCHOR=".ralph/ralph_task.md"
PROMPT="Read .ralph/ralph_task.md, .ralph/progress.md, and .ralph/guardrails.md. Pick up the current focus. Work on the next unchecked item. Run tests after changes. Before exiting, update .ralph/progress.md, .ralph/ralph_task.md checkboxes, and append to .ralph/activity.log. If you hit a repeated error, add a sign to .ralph/guardrails.md."

# Verify we're in a project with ralph state
if [ ! -f "$ANCHOR" ]; then
    echo "ERROR: $ANCHOR not found. Are you in the project root?"
    echo "Run this script from the directory containing .ralph/"
    exit 1
fi

echo "Starting Ralph loop (max $MAX_ITERATIONS iterations)"
echo "Anchor: $ANCHOR"
echo "Watch progress: tail -f .ralph/activity.log"
echo "---"

for i in $(seq 1 "$MAX_ITERATIONS"); do
    echo ""
    echo "=== Ralph iteration $i / $MAX_ITERATIONS — $(date +%H:%M:%S) ==="

    # Check if all boxes are already checked
    if ! grep -q '\[ \]' "$ANCHOR"; then
        echo "All checkboxes complete. Stopping at iteration $i."
        exit 0
    fi

    # Count remaining tasks
    REMAINING=$(grep -c '\[ \]' "$ANCHOR" || true)
    echo "Unchecked tasks remaining: $REMAINING"

    # Run Claude Code
    claude --print \
        -p "$PROMPT" \
        --dangerously-skip-permissions \
        2>&1 | tee -a .ralph/ralph_output_iter${i}.log

    echo "--- Iteration $i complete ---"
    sleep 2
done

# Final status
echo ""
echo "=== Ralph loop finished after $MAX_ITERATIONS iterations ==="
if grep -q '\[ \]' "$ANCHOR"; then
    REMAINING=$(grep -c '\[ \]' "$ANCHOR" || true)
    echo "WARNING: $REMAINING unchecked tasks remain."
    echo "Review .ralph/ralph_task.md and .ralph/progress.md"
else
    echo "All checkboxes complete."
fi
