#!/bin/sh
# Run one (or more) seeded episodes for each installed MiniGrid-* environment.
set -u

PROJECT_DIR="$(CDPATH= cd -P "$(dirname "$0")" && pwd)"
PYTHON_BIN="${MINIGRID_PYTHON:-python}"
EPISODES="${EPISODES:-1}"
SEED="${SEED:-0}"
MAX_STEPS="${MAX_STEPS:-1000}"
RENDER="${RENDER:-0}"
DELAY="${DELAY:-0.4}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/work/minigrid_all_runs}"
mkdir -p "$OUTPUT_DIR"
SUMMARY="$OUTPUT_DIR/summary.csv"
ENV_LIST="$OUTPUT_DIR/.minigrid_envs.$$"
printf 'environment,status,episodes,log\n' > "$SUMMARY"

if ! "$PYTHON_BIN" -c 'import gymnasium, minigrid; from gymnasium.envs.registration import registry; print("\n".join(sorted(e for e in registry if e.startswith("MiniGrid-"))))' > "$ENV_LIST"; then
  printf 'Failed to list MiniGrid environments using %s\n' "$PYTHON_BIN" >&2
  rm -f "$ENV_LIST"
  exit 1
fi
trap 'rm -f "$ENV_LIST"' 0

runner_errors=0
solved=0
partial=0
unsolved=0
total=0
while IFS= read -r env_id || [ -n "$env_id" ]; do
  [ -n "$env_id" ] || continue
  total=$((total + 1))
  safe_name="$(printf '%s' "$env_id" | tr '/' '_')"
  log_file="$OUTPUT_DIR/$safe_name.log"
  printf '[%s] running %s\n' "$total" "$env_id"

  set -- --env "$env_id" --episodes "$EPISODES" --seed "$SEED" --max-steps "$MAX_STEPS"
  if [ "$RENDER" = "1" ]; then
    set -- "$@" --render --delay "$DELAY"
  fi

  if (cd "$PROJECT_DIR" && "$PYTHON_BIN" minigrid_agent.py "$@" > "$log_file" 2>&1); then
    result_line="$(grep '^summary env=' "$log_file" | tail -n 1 || true)"
    successes="$(printf '%s\n' "$result_line" | sed -n 's/.*successes=\([0-9][0-9]*\)\/.*/\1/p')"
    successes="${successes:-0}"
    if [ "$successes" -ge "$EPISODES" ]; then
      status="solved"
      solved=$((solved + 1))
    elif [ "$successes" -gt 0 ]; then
      status="partial"
      partial=$((partial + 1))
    else
      status="unsolved"
      unsolved=$((unsolved + 1))
    fi
  else
    status="error"
    runner_errors=$((runner_errors + 1))
  fi
  printf '%s,%s,%s,%s\n' "$env_id" "$status" "$EPISODES" "$log_file" >> "$SUMMARY"
  tail -n 1 "$log_file" || true
done < "$ENV_LIST"

printf 'Completed %s environments: %s solved, %s partial, %s unsolved, %s runner errors. Summary: %s\n' "$total" "$solved" "$partial" "$unsolved" "$runner_errors" "$SUMMARY"
if [ "$runner_errors" -ne 0 ]; then
  exit 1
fi
