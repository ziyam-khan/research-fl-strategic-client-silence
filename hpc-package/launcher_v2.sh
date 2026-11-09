#!/bin/bash
# Pod-side v2 launcher, after setup_v2.sh printed SETUP PASS. Run detached:
#   cd /workspace/paper-1/hpc-package && nohup bash launcher_v2.sh > logs-v2/launcher.out 2>&1 < /dev/null &
# 1. Mode: gate G1 picks deterministic or nondeterministic (PREREG-v2), once; a relaunch reuses
#    logs-v2/mode, so the grid never mixes modes.
# 2. --preflight: every dataset x arm on real data in that mode (first launch only). Deterministic
#    failure falls back to nondeterministic; if both fail, nothing is launched.
# 3. Workers: keeps up to MAXW claim-workers alive while claimable cells exist. When nothing is
#    running and nothing is claimable but cells are unfinished (crash, .err), clears their locks
#    and retries them once. Respawn budget: if workers keep dying, it STOPS instead of looping.
# Ends with ALL DONE or STOPPED in logs-v2/launcher.log. Watch: tail logs-v2/launcher.log
set -u
cd "$(dirname "$0")" && . .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=results/v2; LOCKS=${AAWA_LOCK_DIR:-/tmp/aawa-v2-locks}; L=logs-v2
mkdir -p $R $L $LOCKS
log(){ echo "$(date -u +%H:%M:%S) $*" | tee -a $L/launcher.log; }

# CPUs actually granted to this container (cgroup quota), not the host's thread count
read -r q per < /sys/fs/cgroup/cpu.max 2>/dev/null || q=max
CPUS=$([ "$q" != "max" ] && [ -n "${per:-}" ] && echo $(( q / per )) || nproc)
DEFAULT_W=$(( CPUS - 2 )); [ "$DEFAULT_W" -gt 30 ] && DEFAULT_W=30; [ "$DEFAULT_W" -lt 1 ] && DEFAULT_W=1
MAXW=${MAXW:-$DEFAULT_W}

# --- mode (G1) and preflight: first launch only ----------------------------------
if [ -f $L/mode ]; then
  DET=$(cat $L/mode); log "relaunch: reusing AAWA_DETERMINISTIC=$DET"
else
  AAWA_DETERMINISTIC=1 python run_v2.py --gate > $L/gate_det.txt 2>&1
  AAWA_DETERMINISTIC=0 python run_v2.py --gate > $L/gate_nondet.txt 2>&1
  DET=$(python - $L/gate_det.txt $L/gate_nondet.txt <<'EOF'
import re, sys
def parse(p):
    m = re.search(r"identical=(\w+) s_per_round=([\d.]+)", open(p).read())
    return (m.group(1) == "True", float(m.group(2))) if m else (False, None)
(same, t_det), (_, t_non) = parse(sys.argv[1]), parse(sys.argv[2])
print(1 if same and t_det and t_non and t_det <= 1.5 * t_non else 0)
EOF
)
  log "G1: $(grep -h '^G1' $L/gate_det.txt $L/gate_nondet.txt | tr '\n' ' ') -> AAWA_DETERMINISTIC=$DET"
  if ! AAWA_DETERMINISTIC=$DET python run_v2.py --preflight > $L/preflight_$DET.txt 2>&1; then
    log "preflight FAILED with AAWA_DETERMINISTIC=$DET (see $L/preflight_$DET.txt)"
    [ "$DET" = 1 ] || { log "STOPPED: preflight failed; nothing launched"; exit 1; }
    DET=0
    AAWA_DETERMINISTIC=0 python run_v2.py --preflight > $L/preflight_0.txt 2>&1 \
      || { log "STOPPED: preflight failed in both modes; nothing launched"; exit 1; }
  fi
  echo $DET > $L/mode
  log "PREFLIGHT PASS (AAWA_DETERMINISTIC=$DET)"
fi
export AAWA_DETERMINISTIC=$DET

# --- workers ------------------------------------------------------------------------
python run_v2.py --list | grep '__seed' > $L/cells.txt
TOTAL=$(wc -l < $L/cells.txt)
[ "$TOTAL" -eq 650 ] || { log "STOPPED: expected 650 cells, got $TOTAL"; exit 1; }
BUDGET=$(( 3 * MAXW + 20 ))   # spawns allowed: first wave + one retry wave + crash replacements
log "start: cpus=$CPUS MAXW=$MAXW budget=$BUDGET total=$TOTAL"
free_cells(){ local n=0 k; while read -r k; do [ -e "$R/$k.done" ] || [ -e "$LOCKS/$k.lock" ] || n=$((n + 1)); done < $L/cells.txt; echo $n; }
w=0; spawned=0; retried=0; last=-1; last_progress=$(date +%s); warned=0
while :; do
  ndone=$(find $R -name '*.done' | wc -l)
  free=$(free_cells)
  live=$(pgrep -fc "[r]un_v2.py --claim")
  if [ "$ndone" != "$last" ]; then
    log "done=$ndone/$TOTAL free=$free live=$live err_files=$(find $R -name '*.err' | wc -l) spawned=$spawned gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -d ' ') ram_free=$(free -g | awk '/Mem/{print $7}')G"
    last=$ndone; last_progress=$(date +%s); warned=0
  fi
  if [ "$ndone" -ge "$TOTAL" ]; then log "ALL DONE: $TOTAL cells"; break; fi
  if [ "$live" -eq 0 ] && [ "$free" -eq 0 ]; then
    [ "$retried" -ge 1 ] && { log "STOPPED: $((TOTAL - ndone)) cells unfinished after one retry; see $R/*.err"; exit 1; }
    n=0
    while read -r k; do
      [ -e "$R/$k.done" ] || { rm -f "$LOCKS/$k.lock"; n=$((n + 1)); }
    done < $L/cells.txt
    retried=1; log "retrying $n unfinished cells (their locks cleared)"
    continue
  fi
  while [ "$live" -lt "$MAXW" ] && [ "$live" -lt "$free" ]; do
    [ "$spawned" -ge "$BUDGET" ] && { log "STOPPED: spawn budget $BUDGET used up; workers keep dying (see $L/w_*.log)"; exit 1; }
    nohup python run_v2.py --claim > $L/w_$w.log 2>&1 < /dev/null &
    w=$((w + 1)); spawned=$((spawned + 1)); live=$((live + 1))
  done
  if [ $(( $(date +%s) - last_progress )) -gt 14400 ] && [ "$warned" -eq 0 ]; then
    log "WARNING: no cell finished in 4 h (live=$live free=$free); check $L/w_*.log"; warned=1
  fi
  sleep "${POLL:-60}"
done
