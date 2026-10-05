"""Intrinsic / nuisance semantic banks.

A bank file (``data/semantic/<dataset>_banks.pth``) is a dictionary::

    {
      "text_dim": 512,
      "classes": {
         "<class name>": {
             "name":      Tensor[512],                 # encoded class-name template (intrinsic fallback)
             "intrinsic": {"local": Tensor[R, 512], "global": Tensor[1, 512]},
             "nuisance":  {"local": Tensor[R', 512], "global": Tensor[1, 512]},   # may be empty (0, 512)
         }, ...
      },
      "supports": {                                   # optional, keyed by support identifiers
         "<relative image path>": {"intrinsic": {...}, "nuisance": {...}}, ...
      },
      "meta": {...}                                   # prompt, generator and encoder versions, seed
    }

``scripts/generate_semantics.py`` produces this file. The legacy DVLA-RL caches
(``<dataset>_semantic_clip_cot.pth`` and ``<dataset>_attributes_clip.pth``) are
also accepted and are treated as intrinsic banks with empty nuisance banks.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


def _unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1) if x.numel() > 0 else x.float()


@dataclass
class SemanticBank:
    """Banks of one episode class. All embeddings are unit vectors."""
    name: str
    name_embedding: torch.Tensor          # (d_t,)
    intrinsic_local: torch.Tensor         # (R, d_t)
    intrinsic_global: torch.Tensor        # (1, d_t)
    nuisance_local: torch.Tensor          # (R', d_t), possibly (0, d_t)
    nuisance_global: torch.Tensor         # (1, d_t) or (0, d_t)

    @property
    def has_nuisance(self) -> bool:
        return self.nuisance_local.shape[0] + self.nuisance_global.shape[0] > 0

    def to(self, device):
        return SemanticBank(self.name, self.name_embedding.to(device), self.intrinsic_local.to(device),
                            self.intrinsic_global.to(device), self.nuisance_local.to(device),
                            self.nuisance_global.to(device))


def _pad_stack(tensors: Sequence[torch.Tensor], d: int, device):
    """Stack variable-length (R_i, d) tensors into (N, R_max, d) with a boolean mask."""
    n = len(tensors)
    r_max = max([t.shape[0] for t in tensors] + [1])
    out = torch.zeros(n, r_max, d, device=device)
    mask = torch.zeros(n, r_max, dtype=torch.bool, device=device)
    for i, t in enumerate(tensors):
        if t.shape[0] > 0:
            out[i, : t.shape[0]] = t.to(device)
            mask[i, : t.shape[0]] = True
    return out, mask


@dataclass
class BankTensors:
    """Padded episode-level tensors used by purification and the gated blocks."""
    pos_local: torch.Tensor      # (N, R1, d_t)
    pos_local_mask: torch.Tensor
    pos_global: torch.Tensor     # (N, R2, d_t)
    pos_global_mask: torch.Tensor
    neg_local: torch.Tensor      # (N, R3, d_t)
    neg_local_mask: torch.Tensor
    neg_global: torch.Tensor     # (N, R4, d_t)
    neg_global_mask: torch.Tensor
    overlap: torch.Tensor        # (N,)   o_c in Eq. (5)
    has_nuisance: torch.Tensor   # (N,) bool
    pos_summary: torch.Tensor    # (N, d_t) mean intrinsic embedding (policy state)
    neg_summary: torch.Tensor    # (N, d_t) mean nuisance embedding, zero when empty
    pos_local_flat: torch.Tensor   # (sum_c R1_c, d_t) intrinsic local phrases of all classes
    pos_global_flat: torch.Tensor  # (sum_c R2_c, d_t)

    @property
    def num_classes(self) -> int:
        return self.pos_local.shape[0]

    @staticmethod
    def from_banks(banks: Sequence[SemanticBank], device) -> "BankTensors":
        d = banks[0].name_embedding.shape[-1]
        pos_local, pos_local_mask = _pad_stack([b.intrinsic_local for b in banks], d, device)
        pos_global, pos_global_mask = _pad_stack([b.intrinsic_global for b in banks], d, device)
        neg_local, neg_local_mask = _pad_stack([b.nuisance_local for b in banks], d, device)
        neg_global, neg_global_mask = _pad_stack([b.nuisance_global for b in banks], d, device)

        overlap, has_nuis, pos_sum, neg_sum = [], [], [], []
        for b in banks:
            pos = torch.cat([b.intrinsic_local, b.intrinsic_global], 0).to(device)
            neg = torch.cat([b.nuisance_local, b.nuisance_global], 0).to(device)
            pos_sum.append(pos.mean(0))
            if neg.shape[0] > 0:
                overlap.append((1.0 + (pos @ neg.t()).max()) / 2.0)   # Eq. (5)
                has_nuis.append(True)
                neg_sum.append(neg.mean(0))
            else:
                overlap.append(torch.zeros((), device=device))
                has_nuis.append(False)
                neg_sum.append(torch.zeros(d, device=device))
        return BankTensors(
            pos_local, pos_local_mask, pos_global, pos_global_mask,
            neg_local, neg_local_mask, neg_global, neg_global_mask,
            torch.stack(overlap), torch.tensor(has_nuis, device=device),
            torch.stack(pos_sum), torch.stack(neg_sum),
            torch.cat([b.intrinsic_local.to(device) for b in banks], 0),
            torch.cat([b.intrinsic_global.to(device) for b in banks], 0),
        )


class SemanticBankStore:
    """Loads a bank file and assembles per-episode banks from class names and,
    optionally, support identifiers. Query images never enter this construction."""

    def __init__(self, path: str, max_local: int = 8, device="cpu"):
        self.max_local = max_local
        self.device = device
        payload = torch.load(path, map_location="cpu")
        self.classes: Dict[str, dict] = {}
        self.supports: Dict[str, dict] = {}
        self.meta = {}
        if "classes" in payload:
            self.text_dim = int(payload.get("text_dim", 512))
            self.classes = payload["classes"]
            self.supports = payload.get("supports", {})
            self.meta = payload.get("meta", {})
        elif "semantic_feature" in payload:      # legacy DVLA-RL global descriptions
            self._load_legacy(path, payload)
        else:
            raise ValueError(f"Unrecognised semantic file: {path}")

    def _load_legacy(self, path: str, payload: dict):
        import os
        semantic = {k: v.float() for k, v in payload["semantic_feature"].items()}
        attr_path = path.replace("_semantic_clip_cot.pth", "_attributes_clip.pth")
        attributes = {}
        if os.path.exists(attr_path):
            attributes = {k: v.float() for k, v in torch.load(attr_path, map_location="cpu")["attribute_features"].items()}
        self.text_dim = next(iter(semantic.values())).shape[-1]
        for name, g in semantic.items():
            loc = attributes.get(name, g.view(1, -1))
            self.classes[name] = {
                "name": g.view(-1),
                "intrinsic": {"local": loc.view(-1, self.text_dim), "global": g.view(1, -1)},
                "nuisance": {"local": torch.zeros(0, self.text_dim), "global": torch.zeros(0, self.text_dim)},
            }
        self.meta = {"source": "legacy-dvla-rl"}

    # ------------------------------------------------------------------ #
    def _entry(self, class_name: str) -> dict:
        if class_name in self.classes:
            return self.classes[class_name]
        key = class_name.replace("_", " ")
        for cand in (key, key.lower(), class_name.lower()):
            if cand in self.classes:
                return self.classes[cand]
        raise KeyError(f"No semantic bank for class '{class_name}'")

    def _merge_role(self, entries: List[dict], role: str, level: str) -> torch.Tensor:
        parts = [e[role][level] for e in entries if role in e and e[role][level].numel() > 0]
        if not parts:
            return torch.zeros(0, self.text_dim)
        t = _unit(torch.cat(parts, 0))
        t = torch.unique(t, dim=0) if t.shape[0] > 1 else t   # drop exact duplicates
        if level == "local" and t.shape[0] > self.max_local:
            t = t[: self.max_local]
        if level == "global" and role == "nuisance" and t.shape[0] > 1:
            t = _unit(t.mean(0, keepdim=True))
        if level == "global" and role == "intrinsic" and t.shape[0] > 1:
            t = _unit(t.mean(0, keepdim=True))
        return t

    def bank(self, class_name: str, support_paths: Optional[Sequence[str]] = None) -> SemanticBank:
        cls = self._entry(class_name)
        entries = [cls]
        if support_paths:
            entries += [self.supports[p] for p in support_paths if p in self.supports]
        name_emb = _unit(cls["name"].view(-1))
        pos_loc = self._merge_role(entries, "intrinsic", "local")
        pos_glob = self._merge_role(entries, "intrinsic", "global")
        if pos_loc.shape[0] == 0:                       # intrinsic fallback: class-name template
            pos_loc = name_emb.view(1, -1)
        if pos_glob.shape[0] == 0:
            pos_glob = name_emb.view(1, -1)
        neg_loc = self._merge_role(entries, "nuisance", "local")
        neg_glob = self._merge_role(entries, "nuisance", "global")
        return SemanticBank(class_name, name_emb, pos_loc, pos_glob, neg_loc, neg_glob).to(self.device)

    def episode_banks(self, class_names: Sequence[str],
                      support_paths: Optional[Sequence[Sequence[str]]] = None) -> List[SemanticBank]:
        banks = []
        for i, name in enumerate(class_names):
            paths = support_paths[i] if support_paths is not None else None
            banks.append(self.bank(name, paths))
        return banks

    def episode_tensors(self, class_names, support_paths=None) -> BankTensors:
        return BankTensors.from_banks(self.episode_banks(class_names, support_paths), self.device)
