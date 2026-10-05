"""Episode unpacking shared by training and evaluation."""
from dataclasses import dataclass
from typing import List

import torch

from ..data.datasets import FewShotImageFolder
from ..data.semantic import BankTensors, SemanticBankStore


@dataclass
class Episode:
    support: torch.Tensor            # (N*K, 3, H, W)
    query: torch.Tensor              # (N*Q, 3, H, W)
    labels: torch.Tensor             # (N*Q,) episode labels 0..N-1
    class_names: List[str]
    support_paths: List[List[str]]   # support identifiers per class
    banks: BankTensors


def unpack_episode(batch, dataset: FewShotImageFolder, idx_to_name: dict, store: SemanticBankStore,
                   way: int, shot: int, query: int, device) -> Episode:
    images, glabels, indices = batch
    per = shot + query
    images = images.view(way, per, *images.shape[1:])
    sup = images[:, :shot].reshape(-1, *images.shape[2:]).to(device, non_blocking=True)
    que = images[:, shot:].reshape(-1, *images.shape[2:]).to(device, non_blocking=True)
    labels = torch.arange(way).repeat_interleave(query).to(device)
    glabels = glabels.view(way, per)[:, 0]
    indices = indices.view(way, per)[:, :shot]
    class_names = [idx_to_name[int(g)] for g in glabels]
    support_paths = [[dataset.relative_path(int(i)) for i in row] for row in indices]
    banks = store.episode_tensors(class_names, support_paths)
    return Episode(sup, que, labels, class_names, support_paths, banks)
