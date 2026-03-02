#!/usr/bin/env bash
# Usage: check_lines.sh <glob_pattern>
# Prints filename: line_count for each matching file.
# Example: check_lines.sh results/exp1a/dana_1a_026_seed*.jsonl

for f in $@; do
    echo "$(basename "$f"): $(wc -l < "$f")"
done
