#!/bin/bash
# Pod-side setup for the v2 grid. Run once, from the unpacked package:
#   cd /workspace/paper-1/hpc-package && bash setup_v2.sh
# Stops at the first failure and prints SETUP FAIL; prints SETUP PASS when the pod is ready for
# launcher_v2.sh. Datasets are fetched on the pod (never uploaded from home) and md5-checked;
# EMNIST arrives in the package and is md5-checked too.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs-v2 data
log(){ echo "$(date -u +%H:%M:%S) $*" | tee -a logs-v2/setup.log; }
die(){ log "SETUP FAIL: $*"; exit 1; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 1. python env: reuse the image's torch (no large pip download), as in v1
python3 -m venv --system-site-packages .venv && . .venv/bin/activate || die "venv"
python - <<'EOF' 2>&1 | tee -a logs-v2/setup.log || die "torch/CUDA"
import sys, torch, torchvision, numpy
assert torch.cuda.is_available(), "CUDA not available"
print(sys.version.split()[0], torch.__version__, torchvision.__version__, numpy.__version__, torch.cuda.get_device_name(0))
EOF
log "cpus=$(nproc) mem=$(free -g | awk '/Mem/{print $2}')G disk=$(df -h /workspace | awk 'NR==2{print $4}') free"

# 2. datasets, in parallel; each verified by md5 (3 attempts, single connection)
fetch(){ local url=$1 out=$2 md5=$3
  for i in 1 2 3; do
    if [ -f "$out" ] && [ "$(md5sum "$out" | cut -d' ' -f1)" = "$md5" ]; then log "ok $out"; return 0; fi
    rm -f "$out"; curl -fsSL --retry 3 --connect-timeout 20 -o "$out" "$url" || log "attempt $i failed: $url"
  done
  [ -f "$out" ] && [ "$(md5sum "$out" | cut -d' ' -f1)" = "$md5" ] && { log "ok $out"; return 0; }
  log "md5/download FAILED $out"; return 1; }
fetch https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/resolve/main/cifar-10-python.tar.gz \
      data/cifar-10-python.tar.gz c58f30108f718f92721af3b95e74349a & p1=$!
fetch https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz \
      data/cifar-100-python.tar.gz eb9058c3a382ffc7106e4002c42a8d85 & p2=$!
fetch "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip" \
      data/har.zip 96924e20fe5b8c9ffc2196eec7d5048f & p3=$!
wait $p1 || die "CIFAR-10"; wait $p2 || die "CIFAR-100"; wait $p3 || die "UCI HAR"

( cd data/EMNIST/raw && md5sum -c --quiet - <<'EOF' ) || die "EMNIST md5"
5e3e63ca1e5dbe4797ac12b7da3ae315  emnist-balanced-test-images-idx3-ubyte
b80127b7765990ce5f32bcbe260fdb18  emnist-balanced-test-labels-idx1-ubyte
7c2bc81c97e776b4834c380c6b6c191b  emnist-balanced-train-images-idx3-ubyte
21a5bbfab1704c4b3c3817fbd0a43ba2  emnist-balanced-train-labels-idx1-ubyte
EOF
log "ok EMNIST md5"

# 3. extract (torchvision reuses the verified archives; HAR: only the six feature/label files)
python -I - <<'EOF' 2>&1 | tee -a logs-v2/setup.log || die "extract"
import io, os, sys, zipfile
sys.path.insert(0, os.path.abspath("src"))
from data import get_cifar10, get_cifar100, get_emnist, get_har
inner = zipfile.ZipFile(io.BytesIO(zipfile.ZipFile("data/har.zip").read("UCI HAR Dataset.zip")))
keep = ("X_train.txt", "y_train.txt", "subject_train.txt", "X_test.txt", "y_test.txt", "subject_test.txt")
for n in inner.namelist():
    if n.split("/")[-1] in keep:
        inner.extract(n, "data")
for name, fn in (("cifar10", get_cifar10), ("cifar100", get_cifar100), ("emnist", get_emnist), ("har", get_har)):
    tr, te = fn("data")
    print("data ok", name, len(tr), len(te))
EOF

# 4. code checks: v1 smoke test (v1 path unchanged), v2 smoke test (G0), frozen-file hashes, cell count
python smoke_test.py 2>&1 | tail -2 | tee -a logs-v2/setup.log | grep -q "all smoke checks passed" || die "v1 smoke test"
python smoke_v2.py 2>&1 | tail -2 | tee -a logs-v2/setup.log | grep -q "G0 v2 smoke checks passed" || die "v2 smoke test"
python -c "import run_v2 as V; print('frozen files verified:', V.prereg_ok())" 2>&1 | tee -a logs-v2/setup.log | grep -q "frozen files verified" || die "frozen-file hashes"
[ "$(python run_v2.py --list | tail -1)" = "650 cells" ] || die "cell count"
log "SETUP PASS"
