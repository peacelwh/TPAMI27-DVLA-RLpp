"""DVLA-RL++: dual-level vision-language alignment with complementary semantic
purification and counterfactual reinforcement-learning gating.

An episode is processed as follows (Fig. 2):

  (A) intrinsic / nuisance banks are supplied by ``SemanticBankStore`` (supports only);
  (B) supports traverse the backbone; at each selected layer an RL-gated attention
      block injects the shared intrinsic tokens with a scalar action a_l;
  (C) purification turns the final support tokens into class prototypes;
  (D) queries replay the support-derived action schedule and are classified
      against the prototypes.

``encode`` supports five gate modes:
    "sample"    actions drawn from the current policy (policy-batch collection)
    "mean"      deterministic mean actions (deployment, representation learning)
    "reference" mean actions of the frozen DVLA-RL gate on its own trajectory
    "static"    learned support-independent scalar gate (ablation)
    "fixed"     externally supplied schedule (queries replay the support schedule)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..data.semantic import BankTensors
from .gated_attention import (RLGatedAttentionBlock, append_semantic_rows, map_to_tokens, strip_semantic_rows,
                              tokens_to_map)
from .policy import STATS_DIM, BetaGatePolicy, GateState, ReferenceGate, StaticGate, ValueBaseline
from .purification import CSPModule, PurificationOutput
from .visformer import visformer_tiny

LAYER_KEYS = ("stage2", "stage3")


@dataclass
class Trajectory:
    actions: Dict[str, torch.Tensor] = field(default_factory=dict)
    log_probs: Dict[str, torch.Tensor] = field(default_factory=dict)
    states: Dict[str, GateState] = field(default_factory=dict)
    costs: Dict[str, torch.Tensor] = field(default_factory=dict)      # per-layer post-action deficit
    layer_tokens: Dict[str, torch.Tensor] = field(default_factory=dict)  # incoming tokens per layer (calibration)
    final_tokens: Optional[torch.Tensor] = None                      # (B, M, d_f)
    pooled: Optional[torch.Tensor] = None                            # (B, d_f) unit features

    @property
    def action_vector(self) -> torch.Tensor:
        return torch.stack([self.actions[k] for k in LAYER_KEYS])

    @property
    def cost(self) -> torch.Tensor:
        """C(a): mean over layers of the per-layer deficit, Eq. (13)."""
        return torch.stack([self.costs[k] for k in LAYER_KEYS]).mean()


@dataclass
class EpisodeResult:
    logits: torch.Tensor
    prototypes: torch.Tensor
    support: Trajectory
    query: Trajectory
    purification: Optional[PurificationOutput]

    @property
    def actions(self) -> Dict[str, torch.Tensor]:
        return self.support.actions

    @property
    def cost(self) -> torch.Tensor:
        return self.support.cost


class DVLARLpp(nn.Module):
    def __init__(self, cfg: ModelConfig, num_classes: int = 64):
        super().__init__()
        self.cfg = cfg
        if cfg.backbone != "visformer-t":
            raise ValueError(f"Unsupported backbone {cfg.backbone}")
        self.backbone = visformer_tiny(num_classes=num_classes, img_size=cfg.image_size)
        d1, d2, d3 = self.backbone.stage_dims
        self.dims = {"stage2": d2, "stage3": d3}
        self.feat_dim = d3
        self.inject = dict(cfg.inject)

        self.gated = nn.ModuleDict({k: RLGatedAttentionBlock(d, cfg.text_dim) for k, d in self.dims.items()})
        self.csp = CSPModule({**self.dims, "final": d3}, cfg.text_dim, d3, cfg.purification)
        self.policy = BetaGatePolicy(self.dims, cfg.text_dim, cfg.policy)
        self.value = ValueBaseline(self.dims, cfg.text_dim, cfg.policy)
        self.reference = nn.ModuleDict({k: ReferenceGate(d, cfg.text_dim) for k, d in self.dims.items()})
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.static_gate = StaticGate(list(self.dims.keys()))

    # ------------------------------------------------------------------ #
    # parameter groups (Appendix S5: representation / scorer / policy / value)
    # ------------------------------------------------------------------ #
    def representation_parameters(self):
        params = list(self.backbone.parameters()) + list(self.gated.parameters()) + self.csp.anchor_parameters()
        if self.cfg.policy.gate == "static":
            params += list(self.static_gate.parameters())
        return params

    def scorer_parameters(self):
        return self.csp.scorer_parameters()

    def policy_parameters(self):
        return list(self.policy.parameters())

    def value_parameters(self):
        return list(self.value.parameters())

    def set_representation_trainable(self, flag: bool):
        for p in self.representation_parameters():
            p.requires_grad_(flag)

    def load_backbone(self, state_dict: dict, strict: bool = False):
        """Load a pre-trained Visformer checkpoint (``visformer-<dataset>.pth``)."""
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=strict)
        return missing, unexpected

    def load_reference_from_dvla_rl(self, state_dict: dict):
        """Copy the frozen gate of a DVLA-RL checkpoint into the reference modules."""
        mapping = {"stage2": ("rl_gate1", "t2i"), "stage3": ("rl_gate2", "t2i2")}
        loaded = 0
        for key, (gate_name, t2i_name) in mapping.items():
            ref = self.reference[key]
            sub = {}
            for name, tensor in state_dict.items():
                if name.startswith(gate_name + ".policy_net."):
                    sub[name[len(gate_name) + 1:]] = tensor
                elif name == t2i_name + ".weight":
                    sub["t2i.weight"] = tensor
            if sub:
                ref.load_state_dict(sub, strict=False)
                loaded += len(sub)
        return loaded

    # ------------------------------------------------------------------ #
    # policy state
    # ------------------------------------------------------------------ #
    def _gate_state(self, key: str, z: torch.Tensor, banks: BankTensors, way: int, shot: int) -> GateState:
        """Summary of the incoming support tokens z (N*K, M, d_l), Appendix S1."""
        N, K = way, shot
        M, C = z.shape[1], z.shape[2]
        with torch.no_grad():
            d, xi = self.csp.evidence(key, z.view(N, K, M, C), banks)
            a, rho = self.csp.allocate(d, xi)
            deficit = self.csp.deficit(d, xi)
            stats = torch.stack([d.mean(), rho.mean(), (1.0 - rho).mean(), deficit.mean(),
                                 banks.has_nuisance.float().mean(), banks.overlap.mean()])
            visual = z.mean(dim=(0, 1))
        layer_index = LAYER_KEYS.index(key)
        return GateState(key, visual, banks.pos_summary.mean(0), banks.neg_summary.mean(0), stats, layer_index)

    # ------------------------------------------------------------------ #
    # encoding with gated semantic injection
    # ------------------------------------------------------------------ #
    def _gated_block(self, key, block, x, banks, mode, actions, traj, way, shot, record_cost, record_tokens):
        B, C, H, W = x.shape
        z = map_to_tokens(x)
        if record_tokens:
            traj.layer_tokens[key] = z
        if mode == "fixed":
            a = actions[key]
        else:
            state = self._gate_state(key, z, banks, way, shot)
            traj.states[key] = state
            if mode == "reference":
                a = self.reference[key].mean_action(x, banks.pos_summary.mean(0))
            elif mode == "static":
                a = self.static_gate(key)
            else:
                a, logp = self.policy.act(state, sample=(mode == "sample"))
                traj.log_probs[key] = logp
        traj.actions[key] = a if mode != "fixed" else a.detach()

        tokens = self.gated[key].semantic_tokens(banks.pos_local_flat, banks.pos_global_flat,
                                                 self.csp.level_weights(key))
        gate_value = a if (mode == "static") else a.detach()
        z_ctx, fused = self.gated[key](z, tokens, gate_value)
        x_cat = append_semantic_rows(tokens_to_map(z_ctx, H, W), fused)
        x_out = strip_semantic_rows(block(x_cat), H)

        if record_cost and mode != "fixed":
            z_post = map_to_tokens(x_out).view(way, shot, H * W, C)
            traj.costs[key] = self.csp.nuisance_cost(key, z_post, banks)
        return x_out

    def encode(self, images: torch.Tensor, banks: BankTensors, mode: str, actions: Optional[Dict] = None,
               way: Optional[int] = None, shot: Optional[int] = None, record_cost: bool = True,
               record_tokens: bool = False) -> Trajectory:
        assert mode in ("sample", "mean", "reference", "static", "fixed"), mode
        if mode == "fixed":
            assert actions is not None
        traj = Trajectory()
        x = self.backbone.forward_stage1(images)
        x = self.backbone.embed_stage2(x)
        for key in LAYER_KEYS:
            if key == "stage3":
                x = self.backbone.embed_stage3(x)
            for i, block in enumerate(self.backbone.blocks(key)):
                if i == self.inject[key]:
                    x = self._gated_block(key, block, x, banks, mode, actions, traj, way, shot, record_cost,
                                          record_tokens)
                else:
                    x = block(x)
        xf = self.backbone.final_tokens(x)
        traj.final_tokens = map_to_tokens(xf)
        traj.pooled = F.normalize(traj.final_tokens.mean(dim=1), dim=-1)
        return traj

    # ------------------------------------------------------------------ #
    # prototypes and classification
    # ------------------------------------------------------------------ #
    def prototypes(self, support: Trajectory, banks: BankTensors, way: int, shot: int):
        M, d = support.final_tokens.shape[1:]
        tokens = support.final_tokens.view(way, shot, M, d)
        if self.cfg.use_csp:
            out = self.csp.purify(tokens, banks)
            return out.prototypes, out
        return self.csp.uniform_prototypes(tokens), None

    def classify(self, query_features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        return (query_features @ prototypes.t()) / self.cfg.tau_c                   # Eq. (3)

    def gate_mode(self, sample: bool) -> str:
        if not self.cfg.use_cfg or self.cfg.policy.gate == "reference":
            return "reference"
        if self.cfg.policy.gate == "static":
            return "static"
        return "sample" if sample else "mean"

    def run_episode(self, support_images: torch.Tensor, query_images: torch.Tensor, banks: BankTensors,
                    way: int, shot: int, mode: Optional[str] = None, sample: bool = False,
                    record_cost: bool = True, record_tokens: bool = False) -> EpisodeResult:
        mode = mode or self.gate_mode(sample)
        sup = self.encode(support_images, banks, mode, way=way, shot=shot, record_cost=record_cost,
                          record_tokens=record_tokens)
        protos, pur = self.prototypes(sup, banks, way, shot)
        que = self.encode(query_images, banks, "fixed", actions=sup.actions, record_cost=False)
        logits = self.classify(que.pooled, protos)
        return EpisodeResult(logits, protos, sup, que, pur)

    @staticmethod
    def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)

    def alignment_loss(self, support: Trajectory, banks: BankTensors, way: int, shot: int) -> torch.Tensor:
        """Calibration objective of the scoring maps over all selected layers and the final layer, Eq. (S4)."""
        losses = []
        for key, z in support.layer_tokens.items():
            M, C = z.shape[1:]
            losses.append(self.csp.alignment_loss(key, z.view(way, shot, M, C), banks))
        M, C = support.final_tokens.shape[1:]
        losses.append(self.csp.alignment_loss("final", support.final_tokens.view(way, shot, M, C), banks))
        return torch.stack(losses).mean()
