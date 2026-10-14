VM ?= denbi
REMOTE ?= /vol/main/Flow_BED

.PHONY: install test sync vm-setup vm-shell

install:
	uv pip install -e ".[dev]"

test:
	pytest -q

# push local code to the VM (excluding env/checkpoints)
sync:
	rsync -av --delete --exclude .venv --exclude checkpoints --exclude '__pycache__' --exclude .git ./ $(VM):$(REMOTE)/

# first-time VM provisioning (see scripts/setup_vm.sh for options)
vm-setup:
	scp scripts/setup_vm.sh $(VM):~/setup_vm.sh
	ssh -t $(VM) 'bash ~/setup_vm.sh'

vm-shell:
	ssh -t $(VM) 'cd $(REMOTE) && source .venv/bin/activate && exec bash'
