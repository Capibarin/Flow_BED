VM ?= denbi
REMOTE ?= /vol/main/Flow_BED

.PHONY: install test sync vm-shell

install:
	uv pip install -e ".[dev]"

test:
	pytest -q

# push local code to the VM (excluding env/checkpoints)
sync:
	rsync -av --delete --exclude .venv --exclude checkpoints --exclude '__pycache__' --exclude .git ./ $(VM):$(REMOTE)/

vm-shell:
	ssh -t $(VM) 'cd $(REMOTE) && source .venv/bin/activate && exec bash'
