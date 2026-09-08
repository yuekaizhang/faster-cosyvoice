#!/usr/bin/env bash
# Run fixed-concurrency points in order. Stop only when returned audio is invalid
# (incomplete PCM or missing audible onset), not merely when playback underruns.
set -euo pipefail

FCV_ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONCURRENCIES=${CONCURRENCIES:-"1 2 3 4 6 8 10 12 14 16"}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-$FCV_ROOT/benchmarks/results/fixed-concurrency}
STOP_ON_INVALID_AUDIO=${STOP_ON_INVALID_AUDIO:-1}

for concurrency in $CONCURRENCIES; do
  output="${OUTPUT_PREFIX}-${concurrency}-seed-${SEED:-0}"
  echo "Running fixed concurrency ${concurrency}: ${output}"
  CONCURRENCY=$concurrency OUTPUT=$output \
    bash "$FCV_ROOT/scripts/run_fixed_concurrency_benchmark.sh" "$@"

  if [ "$STOP_ON_INVALID_AUDIO" = 1 ]; then
    benchmark_python="${BENCH_VENV:-$FCV_ROOT/.benchmark-venv}/bin/python"
    if ! "$benchmark_python" - "$output/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    summary = json.load(stream)
started = summary["started_requests"]
complete = summary["pcm_complete_requests"]
audible = summary["audible_requests"]
if started == 0 or complete != started or audible != complete:
    print(
        f"invalid audio gate: started={started}, complete_pcm={complete}, audible={audible}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
    then
      echo "Stopping sweep after concurrency ${concurrency}: invalid audio was observed." >&2
      break
    fi
  fi
done
