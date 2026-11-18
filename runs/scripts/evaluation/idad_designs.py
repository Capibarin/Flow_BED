# ruff: noqa
"""Run a trained iDAD policy (from its mlflow run) on our held-out SIR runs, save its designs, and score them with
iDAD's own trained critic.

Runs in iDAD's environment (../.venv, torch 1.8 + pyro 1.6), from the repo folder:
    cd repo && ../.venv/bin/python ../idad_designs.py RUN_DIR OUT.pt [--runs ../heldout_runs.pt]
RUN_DIR: mlruns/<experiment>/<run id>. The trained Epidemic model (policy inside) is loaded from
RUN_DIR/artifacts/model/data/model.pth, its data replaced by our held-out paths (heldout_runs.pt: theta, I on the
10000-point grid, one row per run), and its own forward pass (iDAD's Epidemic.forward: the policy, the increasing design
transform, the nearest-interior-grid-point observation) gives the designs of every run. Nothing of iDAD's code changes.
iDAD's critic (RUN_DIR/artifacts/critic, trained jointly with this policy) is a dot product
U(h, theta) = enc_h(h) . enc_theta(theta); it is scored with exactly the InfoNCE of eval_sir.py: per run,
U(h_i, theta_i) - logsumexp(U(h_i, theta_i), U(h_i, c_1..c_L)) + log(L + 1) with the shared contrastive draws
c = heldout_runs.pt["theta_contrastive"][:L], for L = 1000 and 10000.
OUT.pt: {"tau": (runs, T) designs in time units, "xi_u": untransformed designs, "y": (runs, T) observations,
"theta": (runs, 2), "infonce_idad_critic": {L: (runs,) per-run InfoNCE terms}}.
"""

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.getcwd())  # iDAD's repo (run from there)
# the model was pickled from iDAD's __main__ (epidemic.py): its classes must be visible in this __main__
from epidemic import *  # noqa: E402,F401,F403

def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("out")
    p.add_argument("--runs", default="../heldout_runs.pt")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    dev = torch.device(args.device)

    model = torch.load(os.path.join(args.run_dir, "artifacts/model/data/model.pth"), map_location=dev)
    # mlflow pickles the class by value (cloudpickle), so it is a copy of iDAD's Epidemic, not epidemic.Epidemic itself
    assert type(model).__name__ == "Epidemic" and hasattr(model, "design_net"), type(model)
    held = torch.load(args.runs)
    theta, ys, ts = held["theta"].to(dev), held["ys"].to(dev), held["ts"].to(dev)
    n = theta.shape[0]
    model.SIMDATA = {"ys": ys.T.contiguous(), "prior_samples": theta, "ts": ts, "dt": ts[1] - ts[0],
                     "num_samples": n}
    model.to(dev)
    th, designs, obs = model(torch.arange(n, device=dev))  # Epidemic.forward: no-grad policy rollout
    assert torch.allclose(th, theta)
    taus, prev = [], model.lower_bound.to(dev)
    for xi_u in designs:  # untransformed designs -> times (as Epidemic.eval)
        prev = model.transform_designs(xi_untransformed=xi_u.reshape(n, 1), xi_prev=prev)
        taus.append(prev.reshape(n))
    tau = torch.stack(taus, 1)
    y = torch.stack([o.reshape(n) for o in obs], 1)
    xi_u = torch.stack([x.reshape(n) for x in designs], 1)

    # iDAD's critic on these histories (the same (design, observation) pairs its training saw: untransformed designs)
    critic = torch.load(os.path.join(args.run_dir, "artifacts/critic/data/model.pth"), map_location=dev).to(dev).eval()
    theta_c = held["theta_contrastive"].to(dev)
    nce = {}
    with torch.no_grad():
        h_enc = critic.history_encoder_network(*zip(designs, obs))  # (n, enc)
        u_true = (h_enc * critic.latent_encoder_network(theta)).sum(-1)
        u_c = h_enc @ critic.latent_encoder_network(theta_c).T  # (n, 10000)
        for L in (1000, 10000):
            u_all = torch.cat([u_true[:, None], u_c[:, :L]], 1)
            nce[L] = (u_true - u_all.logsumexp(1) + math.log(L + 1)).cpu()
    torch.save({"tau": tau.cpu(), "xi_u": xi_u.cpu(), "y": y.cpu(), "theta": theta.cpu(), "infonce_idad_critic": nce},
               args.out)
    se = lambda v: v.std().item() / math.sqrt(v.numel())  # noqa: E731
    print(f"saved {args.out}: {n} runs, mean designs {[round(v, 1) for v in tau.mean(0).tolist()]}; "
          + "; ".join(f"iDAD critic InfoNCE L={L}: {v.mean().item():.3f} (s.e. {se(v):.3f})" for L, v in nce.items()))


if __name__ == "__main__":
    main()
