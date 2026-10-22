#!/usr/bin/env bash
# Run a pipeline module and surface its ::error::/::warning:: log lines as GitHub
# annotations, WITHOUT losing the module's exit status (pipefail). Piping into a
# `while read` loop without pipefail used to swallow every SystemExit(1).
#
# usage: bash .github/scripts/run_pipeline.sh src.scheduler.daily_job
set -o pipefail

module="${1:?usage: run_pipeline.sh <python.module>}"

python -m "$module" 2>&1 | while IFS= read -r line; do
  echo "$line"
  case "$line" in
    *"::error::"*) echo "::error::${line#*::error::}" ;;
    *"::warning::"*) echo "::warning::${line#*::warning::}" ;;
  esac
done
