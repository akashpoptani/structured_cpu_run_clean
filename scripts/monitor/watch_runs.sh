#!/usr/bin/env bash
# Event stream for every job this user has in flight. One stdout line = one event.
#
# Exists because three jobs today died ~13 min into weight loading (NameError,
# GLIBCXX, OOM) and nothing surfaced it until the job left the queue. Covers the
# FAILURE paths as well as progress: silence must not look like success.
#
# Emits: state transitions, prefill chunk times (with their stage breakdown),
# PREFILL done, [kernel] per-step splits, decode TPOT every 16 steps, anomalies
# (a chunk >1.5x the previous one), and any error signature in .out or .err.
set -uo pipefail
LOGS=/home/akashpt/DeepSeekRun/structured_cpu_run_clean/results_clean/logs
declare -A STATE OFF ERROFF LASTCHUNK

# Seed offsets to the CURRENT end of every log, so a restart reports only what
# happens from now on. Without this each re-arm replays the whole history and
# old failures look like fresh ones.
for f in "$LOGS"/*.out; do [[ -f "$f" ]] && OFF[$f]=$(wc -l < "$f" 2>/dev/null || echo 0); done
for f in "$LOGS"/*.err; do [[ -f "$f" ]] && ERROFF[$f]=$(wc -l < "$f" 2>/dev/null || echo 0); done
while read -r id nm st; do [[ -n "${id:-}" ]] && STATE[$id]="$st"; done \
  < <(squeue -u "$USER" -h -o "%i %j %T" 2>/dev/null)
echo "monitor armed: watching $(ls "$LOGS"/*.out 2>/dev/null | wc -l) logs from their current end"

while true; do
  # --- job state transitions -------------------------------------------------
  while read -r id nm st; do
    [[ -z "${id:-}" ]] && continue
    if [[ "${STATE[$id]:-}" != "$st" ]]; then
      [[ -n "${STATE[$id]:-}" ]] && echo "[$nm $id] $st"
      STATE[$id]="$st"
    fi
  done < <(squeue -u "$USER" -h -o "%i %j %T" 2>/dev/null)

  # --- finished jobs: report exit state once ---------------------------------
  for id in "${!STATE[@]}"; do
    squeue -j "$id" -h >/dev/null 2>&1 && continue
    fin=$(sacct -j "$id" -o State -X -n 2>/dev/null | head -1 | tr -d ' ')
    [[ -n "$fin" ]] && echo "[$id] EXIT $fin"
    unset "STATE[$id]"
  done

  # --- new interesting lines in each log -------------------------------------
  for f in "$LOGS"/*.out; do
    [[ -f "$f" ]] || continue
    [[ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt 1800 ]] && continue   # stale
    b=$(basename "$f" .out)
    n=$(wc -l < "$f" 2>/dev/null || echo 0); o=${OFF[$f]:-0}
    if (( n > o )); then
      sed -n "$((o+1)),${n}p" "$f" 2>/dev/null | grep -E \
        "prefill chunk ->|PREFILL done|\[kernel\]|\[bs-sweep\]|first token|GLOO_SOCKET_IFNAME|bf16-allreduce|moe-batched|flash-attn-cpu|chunked-prefill|indexer-bcast|all_reduce|iface=|Traceback|Error|error:|OOM|Killed|out of memory|CANCELLED|DUE TO" \
        | while IFS= read -r line; do
            # decode steps are filtered below; keep everything else
            echo "[$b] ${line:0:300}"
          done
      # TPOT every 16th decode step only
      sed -n "$((o+1)),${n}p" "$f" 2>/dev/null \
        | grep -oE "DECODE token [0-9]+/[0-9]+ in [0-9]+ ms" \
        | awk -v b="$b" 'NR%16==1 {print "["b"] "$0}'
      # anomaly: a chunk markedly slower than the one before it
      last=$(grep -oE "prefill chunk -> pos [0-9]+/[0-9]+ in [0-9]+ ms" "$f" 2>/dev/null \
             | tail -1 | grep -oE "in [0-9]+" | grep -oE "[0-9]+")
      if [[ -n "${last:-}" && -n "${LASTCHUNK[$f]:-}" ]]; then
        if (( last > LASTCHUNK[$f] * 3 / 2 )); then
          echo "[$b] SLOWDOWN chunk ${LASTCHUNK[$f]} -> ${last} ms (>1.5x)"
        fi
      fi
      [[ -n "${last:-}" ]] && LASTCHUNK[$f]=$last
      OFF[$f]=$n
    fi
  done

  # --- stderr: anything that looks like a failure ----------------------------
  for f in "$LOGS"/*.err; do
    [[ -f "$f" ]] || continue
    [[ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt 1800 ]] && continue
    b=$(basename "$f" .err)
    n=$(wc -l < "$f" 2>/dev/null || echo 0); o=${ERROFF[$f]:-0}
    if (( n > o )); then
      sed -n "$((o+1)),${n}p" "$f" 2>/dev/null \
        | grep -vE "^\[20|srun:|torch/distributed|^ *\^|^ *File \"|warnings.warn|FutureWarning" \
        | grep -iE "error|exception|killed|oom|assert|fatal|traceback" \
        | head -4 | while IFS= read -r line; do echo "[$b ERR] ${line:0:300}"; done
      ERROFF[$f]=$n
    fi
  done
  sleep 20
done
