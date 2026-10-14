#!/usr/bin/env bash
# One-shot setup of a de.NBI (Ubuntu) VM for Flow_BED.
#
#   scp scripts/setup_vm.sh denbi:~/ && ssh -t denbi 'bash ~/setup_vm.sh'
#
# Stages (each is idempotent, re-run after a reboot to continue):
#   1. system packages, Python, uv, tmux, git
#   2. mount the data volume at /vol/main (auto-detects the attached but unmounted disk)
#   3. NVIDIA driver (GPU flavours only) -> reboots once if a driver was installed
#   4. clone/sync the repo into /vol/main/Flow_BED, create venv, install torch (CUDA if GPU) + package
#   5. run the test-suite
#
# Options (env vars):
#   REPO=<git url>        clone from git (otherwise expects the code to be rsync'ed to $PROJECT_DIR)
#   DEVICE=/dev/vdb       disk for /vol/main (skips auto-detection)
#   FORMAT=1              allow mkfs.ext4 on DEVICE if it has no filesystem (destructive!)
#   NO_GPU=1              skip the driver stage
#   CUDA=cu124            torch CUDA wheel index to use (default cu124)
set -euo pipefail

MOUNT=/vol/main
PROJECT_DIR=${PROJECT_DIR:-$MOUNT/Flow_BED}
CUDA=${CUDA:-cu124}
log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# ------------------------------------------------------------------ 1. system
log "System packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    build-essential git curl tmux htop rsync python3 python3-venv python3-pip \
    ubuntu-drivers-common pciutils

if ! command -v uv >/dev/null; then
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
grep -q '.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# ------------------------------------------------------------------ 2. volume
log "Data volume at $MOUNT"
if mountpoint -q "$MOUNT"; then
    echo "$MOUNT already mounted: $(findmnt -no SOURCE,FSTYPE,SIZE "$MOUNT")"
else
    if [[ -z "${DEVICE:-}" ]]; then
        # largest whole disk that is not mounted anywhere and holds no partitions (typical de.NBI volume: /dev/vdb)
        DEVICE=$(lsblk -dpno NAME,TYPE,SIZE,MOUNTPOINT -b | awk '$2=="disk" && $4==""{print $3, $1}' \
                 | while read -r size dev; do
                       [[ -z "$(lsblk -no MOUNTPOINT "$dev" | tr -d '[:space:]')" ]] && echo "$size $dev"
                   done | sort -rn | head -1 | awk '{print $2}')
    fi
    if [[ -z "${DEVICE:-}" ]]; then
        echo "!! No unmounted disk found. Attach a volume in the de.NBI portal (or set DEVICE=) and re-run." >&2
        lsblk; exit 1
    fi
    FSTYPE=$(lsblk -no FSTYPE "$DEVICE" | head -1 | tr -d '[:space:]')
    echo "device: $DEVICE  fstype: ${FSTYPE:-<none>}"
    if [[ -z "$FSTYPE" ]]; then
        if [[ "${FORMAT:-0}" == "1" ]]; then
            log "Formatting $DEVICE as ext4 (new volume)"
            sudo mkfs.ext4 -L main "$DEVICE"
        else
            echo "!! $DEVICE has no filesystem. If it is a NEW empty volume re-run with FORMAT=1 DEVICE=$DEVICE" >&2
            exit 1
        fi
    fi
    sudo mkdir -p "$MOUNT"
    UUID=$(sudo blkid -s UUID -o value "$DEVICE")
    FST=$(sudo blkid -s TYPE -o value "$DEVICE")
    grep -q "UUID=$UUID" /etc/fstab || echo "UUID=$UUID  $MOUNT  $FST  defaults,nofail  0 2" | sudo tee -a /etc/fstab
    sudo mount "$MOUNT"
    sudo chown "$USER:$USER" "$MOUNT"
    echo "mounted: $(findmnt -no SOURCE,FSTYPE,SIZE "$MOUNT")"
fi

# ------------------------------------------------------------------ 3. GPU driver
if [[ "${NO_GPU:-0}" != "1" ]] && lspci | grep -qi nvidia; then
    if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
        log "NVIDIA driver OK"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    else
        log "Installing NVIDIA driver (GPU detected)"
        sudo ubuntu-drivers install --gpgpu || sudo ubuntu-drivers autoinstall
        echo
        echo ">>> Driver installed. Rebooting in 10s; afterwards run this script again to finish. <<<"
        sleep 10; sudo reboot; exit 0
    fi
    HAS_GPU=1
else
    log "No NVIDIA GPU (or NO_GPU=1) -> CPU torch"
    HAS_GPU=0
fi

# ------------------------------------------------------------------ 4. project
log "Project in $PROJECT_DIR"
if [[ ! -d "$PROJECT_DIR" ]]; then
    if [[ -n "${REPO:-}" ]]; then
        git clone "$REPO" "$PROJECT_DIR"
    else
        echo "!! $PROJECT_DIR missing. Either set REPO=<git url> or from your laptop run:" >&2
        echo "   rsync -av --exclude .venv --exclude checkpoints ~/Documents/Projects/Flow_BED denbi:$MOUNT/" >&2
        exit 1
    fi
fi
cd "$PROJECT_DIR"
[[ -d .venv ]] || uv venv --python 3.11 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
if [[ $HAS_GPU == 1 ]]; then
    uv pip install torch --index-url "https://download.pytorch.org/whl/$CUDA"
else
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
uv pip install -e ".[dev]"
mkdir -p "$MOUNT/checkpoints" && ln -sfn "$MOUNT/checkpoints" checkpoints

# ------------------------------------------------------------------ 5. check
log "Sanity check"
python - <<'EOF'
import torch, torchdyn, flow_bed
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("torchdyn", torchdyn.__version__, "| flow_bed", flow_bed.__version__)
EOF
pytest -q

cat <<EOF

Done. Next:
  ssh -t denbi 'cd $PROJECT_DIR && source .venv/bin/activate && tmux new -s train'
  python scripts/train_toy.py && python scripts/sample_toy.py
EOF
