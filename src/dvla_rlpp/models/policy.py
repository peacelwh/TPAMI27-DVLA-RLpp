"""Counterfactual reinforcement-learning gating (Section 3.4, Fig. 5; Appendix S4).

* ``BetaGatePolicy``  pi_theta(a | H_l) = Beta(kappa p, kappa (1 - p)), Eq. (11), with the
  bounded mean of Eq. (S5). One scalar action per selected layer and episode.
* ``ValueBaseline``   b_omega(H_l), fitted by Eq. (16), separate parameters.
* ``ReferenceGate``   the frozen DVLA-RL gate (categorical over visual-dominant /
  text-dominant / fusion with a similarity bias) evaluated on the support-pooled
  base state. Its mean action a_l^0 = E_{pi_0}[a | H_l^0] propagates its own
  trajectory, Eq. (12).
* ``StaticGate``      support-independent scalar gate (ablation "static gate").
* ``cf_ppo_loss``     clipped surrogate of Eq. (18) with the active-clipping indicator of Eq. (S20).
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from ..config import PolicyConfig

STATS_DIM = 6   # mean evidence, retained mass, neutral mass, candidate deficit, nuisance presence, overlap

# scalar value of each DVLA-RL categorical action: visual-dominant, text-dominant, fusion
REFERENCE_ACTION_VALUES = (1.0, 0.0, 0.5)


@dataclass
class GateState:
    """Permutation-invariant summary H_l of the incoming support state (Appendix S1)."""
    key: str
    visual: torch.Tensor      # (d_l,) support-pooled incoming visual tokens
    pos: torch.Tensor         # (d_t,) intrinsic bank summary
    neg: torch.Tensor         # (d_t,) nuisance bank summary (zero if empty)
    stats: torch.Tensor       # (STATS_DIM,) rejection statistics
    layer_index: int

    def detach(self) -> "GateState":
        return GateState(self.key, self.visual.detach(), self.pos.detach(), self.neg.detach(),
                         self.stats.detach(), self.layer_index)


def stack_states(states: Sequence[GateState]):
    """Group states by layer key and stack their fields."""
    groups: Dict[str, List[GateState]] = {}
    for s in states:
        groups.setdefault(s.key, []).append(s)
    out = {}
    for key, group in groups.items():
        out[key] = dict(
            visual=torch.stack([s.visual for s in group]),
            pos=torch.stack([s.pos for s in group]),
            neg=torch.stack([s.neg for s in group]),
            stats=torch.stack([s.stats for s in group]),
            layer=torch.tensor([s.layer_index for s in group], device=group[0].visual.device),
        )
    return out


class StateEncoder(nn.Module):
    def __init__(self, dims: Dict[str, int], text_dim: int, hidden: int, num_layers: int):
        super().__init__()
        self.keys = list(dims.keys())
        self.num_layers = num_layers
        self.visual_proj = nn.ModuleDict({k: nn.Linear(d, hidden) for k, d in dims.items()})
        self.pos_proj = nn.Linear(text_dim, hidden)
        self.neg_proj = nn.Linear(text_dim, hidden)
        self.stats_proj = nn.Linear(STATS_DIM + num_layers, hidden)
        self.mlp = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())

    def forward(self, key: str, visual, pos, neg, stats, layer) -> torch.Tensor:
        onehot = F.one_hot(layer, self.num_layers).to(stats.dtype)
        h = torch.cat([
            torch.tanh(self.visual_proj[key](visual)), torch.tanh(self.pos_proj(pos)),
            torch.tanh(self.neg_proj(neg)), torch.tanh(self.stats_proj(torch.cat([stats, onehot], dim=-1))),
        ], dim=-1)
        return self.mlp(h)


class BetaGatePolicy(nn.Module):
    def __init__(self, dims: Dict[str, int], text_dim: int, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = StateEncoder(dims, text_dim, cfg.hidden, len(dims))
        self.head = nn.Linear(cfg.hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)          # initial mean action 0.5

    def mean(self, key: str, visual, pos, neg, stats, layer) -> torch.Tensor:
        f = self.head(self.encoder(key, visual, pos, neg, stats, layer)).squeeze(-1)
        eps = self.cfg.eps_g
        return eps + (1.0 - 2.0 * eps) * torch.sigmoid(f)                      # Eq. (S5)

    def distribution(self, key, visual, pos, neg, stats, layer) -> Beta:
        p = self.mean(key, visual, pos, neg, stats, layer)
        k = self.cfg.kappa_g
        return Beta(k * p, k * (1.0 - p))                                       # Eq. (11)

    @staticmethod
    def _unbatch(state: GateState):
        return (state.key, state.visual[None], state.pos[None], state.neg[None], state.stats[None],
                torch.tensor([state.layer_index], device=state.visual.device))

    def act(self, state: GateState, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (action, log_prob) for one state. The action carries no gradient."""
        dist = self.distribution(*self._unbatch(state))
        if sample:
            a = dist.sample().clamp(1e-4, 1.0 - 1e-4)
        else:
            a = dist.mean.detach()
        return a.squeeze(0), dist.log_prob(a).squeeze(0)

    def log_prob(self, states: Sequence[GateState], actions: torch.Tensor) -> torch.Tensor:
        """Batched log pi_theta(a_l | H_l); ``actions`` aligned with ``states``."""
        out = torch.zeros(len(states), device=actions.device)
        groups: Dict[str, List[int]] = {}
        for i, s in enumerate(states):
            groups.setdefault(s.key, []).append(i)
        for key, idx in groups.items():
            group = [states[i] for i in idx]
            stacked = stack_states(group)[key]
            dist = self.distribution(key, **stacked)
            a = actions[idx].clamp(1e-4, 1.0 - 1e-4)
            out[idx] = dist.log_prob(a)
        return out


class ValueBaseline(nn.Module):
    def __init__(self, dims: Dict[str, int], text_dim: int, cfg: PolicyConfig):
        super().__init__()
        self.encoder = StateEncoder(dims, text_dim, cfg.hidden, len(dims))
        self.head = nn.Linear(cfg.hidden, 1)

    def forward(self, states: Sequence[GateState]) -> torch.Tensor:
        device = states[0].visual.device
        out = torch.zeros(len(states), device=device)
        groups: Dict[str, List[int]] = {}
        for i, s in enumerate(states):
            groups.setdefault(s.key, []).append(i)
        for key, idx in groups.items():
            stacked = stack_states([states[i] for i in idx])[key]
            out[idx] = self.head(self.encoder(key, **stacked)).squeeze(-1)
        return out


class ReferenceGate(nn.Module):
    """One frozen DVLA-RL gate. ``policy_net`` and ``t2i`` keep the original parameter
    names so that ``rl_gate{1,2}.policy_net.*`` and ``t2i{,2}.weight`` of a DVLA-RL
    checkpoint load directly (see ``DVLARLpp.load_reference_from_dvla_rl``)."""

    def __init__(self, dim: int, text_dim: int = 512, use_similarity_bias: bool = True):
        super().__init__()
        self.policy_net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, 3, 1, bias=True))
        self.t2i = nn.Linear(text_dim, dim, bias=False)
        self.use_similarity_bias = use_similarity_bias
        self.register_buffer("action_values", torch.tensor(REFERENCE_ACTION_VALUES))

    @staticmethod
    def _similarity_bias(similarity: torch.Tensor) -> torch.Tensor:
        """Heuristic logit bias of DVLA-RL as a function of visual-text cosine similarity."""
        B = similarity.shape[0]
        bias = torch.zeros(B, 3, device=similarity.device)
        high = similarity > 0.5
        mid = (similarity > 0.0) & ~high
        low = ~(high | mid)
        bias[high] = torch.tensor([0.0, 0.5, 1.0], device=similarity.device)
        bias[mid] = torch.tensor([0.0, 1.0, 0.0], device=similarity.device)
        bias[low] = torch.tensor([1.0, 0.0, 0.0], device=similarity.device)
        return bias

    @torch.no_grad()
    def mean_action(self, x: torch.Tensor, pos_summary: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) incoming support maps; pos_summary: (d_t,) intrinsic summary."""
        pooled = x.mean(dim=0, keepdim=True)                                    # support-pooled base state
        logits = self.policy_net(pooled).flatten(1)                             # (1, 3)
        if self.use_similarity_bias:
            v = F.adaptive_avg_pool2d(pooled, 1).flatten(1)
            t = F.normalize(self.t2i(pos_summary.view(1, -1)), dim=-1)
            sim = F.cosine_similarity(v, t, dim=1)
            logits = logits + self._similarity_bias(sim)
        probs = logits.softmax(dim=-1)
        return (probs * self.action_values).sum(dim=-1).squeeze(0)             # E_{pi_0}[a]


class StaticGate(nn.Module):
    """Learned support-independent scalar gate per layer (ablation)."""

    def __init__(self, keys: Sequence[str]):
        super().__init__()
        self.logits = nn.ParameterDict({k: nn.Parameter(torch.zeros(())) for k in keys})

    def forward(self, key: str) -> torch.Tensor:
        return torch.sigmoid(self.logits[key])


def cf_ppo_loss(new_log_prob: torch.Tensor, old_log_prob: torch.Tensor, advantage: torch.Tensor,
                eps_p: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clipped surrogate L_CF-PPO, Eq. (18), and the active-clipping frequency, Eq. (S20)."""
    ratio = torch.exp(new_log_prob - old_log_prob)
    clipped = ratio.clamp(1.0 - eps_p, 1.0 + eps_p)
    surrogate = torch.min(ratio * advantage, clipped * advantage)
    active = ((advantage > 0) & (ratio > 1.0 + eps_p)) | ((advantage < 0) & (ratio < 1.0 - eps_p))
    return -surrogate.mean(), active.float().mean()
