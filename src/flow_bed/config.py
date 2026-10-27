"""YAML config files for the training scripts: ``--config file.yaml`` sets the defaults of the script's arguments.

The keys are the argument names (``q_min`` or ``q-min``); arguments given on the command line override the file.
Unknown keys and values outside an argument's choices are errors. String values go through the argument's type, as
argparse does for string defaults (so paths become ``Path``).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import yaml


def parse_args(p: argparse.ArgumentParser, argv: Sequence[str] | None = None) -> argparse.Namespace:
    """``p.parse_args(argv)`` with an added ``--config`` option whose YAML file sets the defaults."""
    p.add_argument("--config", default=None, help="YAML file with defaults for these arguments (keys: argument names)")
    pre, _ = p.parse_known_args(argv)
    if pre.config is not None:
        with open(pre.config) as f:
            values = yaml.safe_load(f) or {}
        actions = {a.dest: a for a in p._actions}
        defaults = {}
        for key, value in values.items():
            dest = key.replace("-", "_")
            if dest not in actions or dest in ("help", "config"):
                p.error(f"{pre.config}: unknown argument '{key}'")
            choices = actions[dest].choices
            if choices is not None and value is not None and value not in choices:
                p.error(f"{pre.config}: {key} = {value!r} is not one of {list(choices)}")
            defaults[dest] = value
        p.set_defaults(**defaults)
    return p.parse_args(argv)
