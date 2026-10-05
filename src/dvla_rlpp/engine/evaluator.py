"""Episodic evaluation with the deterministic mean policy (Section 5.1, evaluation protocol)."""
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from .. import utils
from ..models.dvla_rlpp import LAYER_KEYS
from .episode import unpack_episode


@torch.no_grad()
def evaluate(model, loader, dataset, idx_to_name, store, way, shot, query, device, desc="eval",
             mode=None, progress=True):
    """Return mean accuracy, its 95% CI and averaged purification / gate diagnostics."""
    model.eval()
    accs, diag, actions = [], defaultdict(list), defaultdict(list)
    iterator = tqdm(loader, desc=desc, leave=False) if progress else loader
    for batch in iterator:
        ep = unpack_episode(batch, dataset, idx_to_name, store, way, shot, query, device)
        res = model.run_episode(ep.support, ep.query, ep.banks, way, shot, mode=mode, sample=False,
                                record_cost=True)
        accs.append(utils.compute_acc(res.logits, ep.labels))
        if res.purification is not None:
            for k, v in res.purification.diagnostics().items():
                diag[k].append(v)
        diag["nuisance_cost"].append(res.cost.item())
        for k in LAYER_KEYS:
            actions[k].append(res.actions[k].item())
    mean, ci = utils.count_95acc(np.array(accs))
    summary = {"acc": mean, "ci95": ci, "episodes": len(accs)}
    summary.update({k: float(np.mean(v)) for k, v in diag.items()})
    summary.update({f"gate_{k}": float(np.mean(v)) for k, v in actions.items()})
    return summary


def format_summary(s: dict) -> str:
    base = "acc = {:.2f} +- {:.2f} (%) over {} episodes".format(s["acc"] * 100, s["ci95"] * 100, s["episodes"])
    extras = []
    for k in ("retention", "neutral_mass", "candidate_deficit", "nuisance_cost", "fallback_rate"):
        if k in s:
            extras.append("{}={:.3f}".format(k, s[k]))
    for k in LAYER_KEYS:
        if f"gate_{k}" in s:
            extras.append("a_{}={:.3f}".format(k, s[f"gate_{k}"]))
    return base + (" | " + ", ".join(extras) if extras else "")
