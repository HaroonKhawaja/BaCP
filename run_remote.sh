#!/usr/bin/env bash
# Run the ladder on a remote box, detached from your terminal.
#
#   ./run_remote.sh setup            # one-time: venv, deps, dataset, tests
#   ./run_remote.sh start 1          # launch tier 1 in tmux, then detach
#   ./run_remote.sh status           # cells complete
#   ./run_remote.sh snapshot         # tar the records for pulling down
#   ./run_remote.sh attach / stop / report
#
# Closing your laptop, dropping SSH or losing wifi does not affect a run started
# with `start`: it lives in tmux on the remote machine.
#
# WHY THIS IS SAFE TO INTERRUPT
# A cell counts as complete iff a run record carrying its key exists, so
# resuming is just re-running the same command -- you lose at most the cell in
# flight. There is no progress file to fall out of sync.
#
# WITHOUT A NETWORK VOLUME (Community Cloud often has none)
# Results live on the pod's own disk, so pod TERMINATION loses them. The two
# artefacts need different treatment, because they differ by four orders of
# magnitude in size and in what losing them costs:
#
#   records (JSON)  ~1.5 MB at tier 1  -- lose these and you lose EVERYTHING:
#                                         every result and all resume state.
#   checkpoints     ~10 GB at tier 1   -- lose these and you re-run the 5 dense
#                                         cells, about 1.5 GPU-hours.
#
# So: keep everything under /workspace (survives pod stop/start), and pull the
# records down regularly with `snapshot`. A dead pod then costs ~$1 of dense
# re-runs rather than the sweep.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${BACP_SESSION:-bacp}"

# Which interpreter every subcommand uses. A venv only exists when `setup` had
# to build one; on a box whose image already ships a CUDA-capable torch we use
# that interpreter directly and never create a venv. Resolving this in one place
# keeps `start` and `status` from reaching for a venv that setup deliberately
# did not create.
if [ -x "${REPO}/.venv/bin/python" ]; then
    PY="${REPO}/.venv/bin/python"
else
    PY="$(command -v python3 || command -v python)"
fi

# /workspace is RunPod's persistent container volume: it survives a pod stop and
# restart, though not termination. Anything outside it dies on a restart.
if [ -d /workspace ] && [ -z "${BACP_RESULTS_DIR:-}" ]; then
    export BACP_RESULTS_DIR=/workspace/bacp_results
fi
export BACP_RESULTS_DIR="${BACP_RESULTS_DIR:-${REPO}/results}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

cmd="${1:-help}"
tier="${2:-1}"

case "$cmd" in

setup)
    # Use a CUDA-capable torch if the image already has one, and DO NOT touch it.
    #
    # requirements.txt pins torch==2.5.1, which predates Blackwell: an RTX 50-
    # series card is sm_120 and needs torch >= 2.7 with cu128. On a box whose
    # image ships a working Blackwell torch, installing our pin would replace a
    # functioning setup with one that raises "no kernel image is available for
    # execution on the device". So the image's torch wins, and we install only
    # the packages around it.
    if python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "== using the image's torch (CUDA available); NOT creating a venv"
        "$PY" -c "import torch;print(f'   torch {torch.__version__}  cuda {torch.version.cuda}  sm_{\"\".join(map(str,torch.cuda.get_device_capability(0)))}')"
        # Everything except torch/torchvision, which the image owns.
        "$PY" -m pip install -q transformers==4.46.3 datasets==3.1.0             fvcore pandas scipy pytest matplotlib tqdm || true
    else
        echo "== no CUDA torch found; building a CPU venv from requirements.txt"
        python3 -m venv "${REPO}/.venv"
        "$PY" -m pip install --upgrade pip -q
        "$PY" -m pip install -r "${REPO}/requirements.txt"
    fi

    echo "== GPU"
    "$PY" - <<'EOF'
import torch
ok = torch.cuda.is_available()
print(f"  cuda available : {ok}")
print(f"  torch built for: CUDA {torch.version.cuda}")
if ok:
    n = torch.cuda.device_count()
    print(f"  devices        : {n} x {torch.cuda.get_device_name(0)}")
    print(f"  memory each    : {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} GB")
else:
    print("  !! No GPU visible. Do not start a sweep.")
import os
print(f"  host vCPU      : {os.cpu_count()}")
EOF

    echo "== results dir: ${BACP_RESULTS_DIR}"
    mkdir -p "${BACP_RESULTS_DIR}"
    case "$BACP_RESULTS_DIR" in
        /workspace/*) echo "   on /workspace -- survives pod stop/start, NOT termination" ;;
        *) echo "   !! NOT on /workspace. A pod restart will lose it." ;;
    esac

    echo "== pre-fetching CIFAR-10"
    cd "${REPO}/project/scripts"
    "$PY" -c "
from torchvision.datasets import CIFAR10
for t in (True, False): CIFAR10('./cache', train=t, download=True)
print('  ready')"

    echo "== verifying the code before spending GPU on it"
    cd "$REPO" && "$PY" -m pytest -q -W ignore::DeprecationWarning \
        -W ignore::FutureWarning
    echo "== setup complete"
    ;;

start)
    command -v tmux >/dev/null || { echo "need tmux: apt-get install -y tmux"; exit 1; }
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "'$SESSION' already running -- ./run_remote.sh attach"; exit 1
    fi
    mkdir -p "${BACP_RESULTS_DIR}/logs"
    LOG="${BACP_RESULTS_DIR}/logs/sweep_t${tier}_$(date +%Y%m%d_%H%M%S).log"

    GPUS="${BACP_GPUS:-$("$PY" -c 'import torch;print(max(torch.cuda.device_count(),1))')}"
    PERGPU="${BACP_PER_GPU:-1}"

    # `resume` is the default and is what you want after any interruption. Use
    # --no_resume ONLY when the code changed, so every cell shares one
    # code_fingerprint.
    tmux new-session -d -s "$SESSION" \
        "cd '$REPO' && BACP_TIER=$tier '$PY' project/experiments/pool.py \
         --tier $tier --gpus $GPUS --per-gpu $PERGPU 2>&1 | tee '$LOG'"

    # Snapshot the records every 5 minutes. They are ~1.5 MB, so this is free,
    # and it is the difference between losing a pod and losing the sweep.
    tmux new-session -d -s "${SESSION}-snap" \
        "while true; do '$REPO/run_remote.sh' snapshot >/dev/null 2>&1; sleep 300; done"

    echo "started tier $tier on $GPUS GPU(s) x $PERGPU"
    echo "  log      : $LOG"
    echo "  snapshots: ${BACP_RESULTS_DIR}/records_latest.tar.gz (every 5 min)"
    echo
    echo "Safe to close your laptop. PULL THE SNAPSHOT PERIODICALLY:"
    echo "  runpodctl send ${BACP_RESULTS_DIR}/records_latest.tar.gz"
    ;;

snapshot)
    # Records only. Checkpoints are ~10 GB and regenerable; records are 1.5 MB
    # and are the entire result set plus all resume state.
    out="${BACP_RESULTS_DIR}/records_latest.tar.gz"
    tar -czf "${out}.tmp" -C "${BACP_RESULTS_DIR}" \
        runs 2>/dev/null && mv "${out}.tmp" "$out"
    n=$(ls "${BACP_RESULTS_DIR}"/runs/*.json 2>/dev/null | wc -l)
    echo "snapshot: $n record(s) -> $out ($(du -h "$out" | cut -f1))"
    echo "pull with:  runpodctl send $out"
    ;;

status)
    cd "$REPO"
    "$PY" - <<'EOF'
import glob, json, os, sys
sys.path[:0] = ['project', 'project/experiments']
import manifest as M
root = os.environ.get('BACP_RESULTS_DIR', 'results')
ok, failed = set(), set()
for f in glob.glob(os.path.join(root, 'runs', '*.json')):
    d = json.load(open(f, encoding='utf-8'))
    g = d.get('experiment_group')
    if g:
        (ok if d.get('status') == 'ok' else failed).add(g)
cells = {c['key'] for c in M.cells(M.TIER)}
print(f"tier {M.TIER}: {len(ok & cells)}/{len(cells)} cells complete")
if failed - ok:
    print(f"  FAILED: {sorted(failed - ok)[:5]}")
EOF
    tmux has-session -t "$SESSION" 2>/dev/null \
        && echo "  RUNNING" || echo "  not running"
    ;;

attach)  tmux attach -t "$SESSION" ;;
stop)    tmux kill-session -t "$SESSION" 2>/dev/null || true
         tmux kill-session -t "${SESSION}-snap" 2>/dev/null || true
         echo "stopped; 'start' resumes where it left off" ;;
report)  cd "$REPO" && "$PY" -c "
import sys; sys.path[:0]=['project','project/experiments']
import ladder; ladder.report()" ;;
*)       sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
esac
