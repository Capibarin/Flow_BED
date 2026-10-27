import argparse
from pathlib import Path

import pytest

from flow_bed.config import parse_args


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--q-min", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dims", type=int, nargs="+", default=[1])
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--act", choices=["silu", "selu"], default="silu")
    p.add_argument("--flag", action="store_true")
    return p


def test_config_sets_defaults_and_cli_overrides(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("q-min: 1\nlr: 1e-4\ndims: [3, 4]\nckpt: a/b.pt\nact: selu\nflag: true\n")
    args = parse_args(parser(), ["--config", str(cfg)])
    assert (args.q_min, args.lr, args.dims, args.ckpt, args.act, args.flag) == (1, 1e-4, [3, 4], Path("a/b.pt"), "selu", True)
    assert parse_args(parser(), ["--config", str(cfg), "--q-min", "5", "--lr", "0.5"]).q_min == 5
    assert parse_args(parser(), []).q_min == 2


@pytest.mark.parametrize("text", ["q_mn: 1\n", "act: relu\n"])
def test_config_rejects_unknown_keys_and_choices(tmp_path, text):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(text)
    with pytest.raises(SystemExit):
        parse_args(parser(), ["--config", str(cfg)])
