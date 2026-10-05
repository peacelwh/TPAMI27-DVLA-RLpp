"""Shared helpers: seeding, image transforms, logging, timers and metrics."""
import os
import random
import time

import numpy as np
import torch
from torchvision import transforms

_log_path = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Transforms (224x224 episodic protocol, identical to DVLA-RL / VT-FSL)
# --------------------------------------------------------------------------- #
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CIFAR_MEAN, CIFAR_STD = (0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)


def build_transform(dataset: str, split: str, image_size: int = 224):
    """Return the train / eval transform used by episodic meta-tuning and testing."""
    if dataset in ("CIFAR-FS", "FC100"):
        mean, std = CIFAR_MEAN, CIFAR_STD
        resize = (image_size, image_size)
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        resize = (int(image_size * 256 / 224), int(image_size * 256 / 224))
    ops = [transforms.Resize(resize), transforms.CenterCrop(image_size)]
    if split == "train" and dataset in ("CIFAR-FS", "FC100"):
        ops += [transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


def convert_raw(dataset: str, x: torch.Tensor) -> torch.Tensor:
    """Undo normalisation for visualisation."""
    mean, std = (CIFAR_MEAN, CIFAR_STD) if dataset in ("CIFAR-FS", "FC100") else (IMAGENET_MEAN, IMAGENET_STD)
    mean = torch.tensor(mean).view(3, 1, 1).type_as(x)
    std = torch.tensor(std).view(3, 1, 1).type_as(x)
    return x * std + mean


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_n_params(model, return_str: bool = True):
    tot = sum(int(np.prod(p.shape)) for p in model.parameters())
    if not return_str:
        return tot
    return "{:.1f}M".format(tot / 1e6) if tot >= 1e6 else "{:.1f}K".format(tot / 1e3)


def compute_acc(logits: torch.Tensor, label: torch.Tensor, reduction: str = "mean"):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == "none":
        return ret.detach()
    return ret.mean().item()


def compute_acc_mix(logits, label, reduction="mean"):
    if len(label.shape) > 1:
        label = torch.argmax(label, dim=1)
    correct = (torch.argmax(logits, dim=1) == label).float()
    if reduction == "mean":
        return correct.mean().item()
    if reduction == "sum":
        return correct.sum().item()
    return correct


def count_95acc(accuracies):
    """Mean accuracy and its 95% confidence interval over episodes."""
    acc = np.asarray(accuracies, dtype=np.float64)
    return float(acc.mean()), float(1.96 * acc.std() / np.sqrt(len(acc)))


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def ensure_path(path: str, exist_ok: bool = False) -> None:
    if os.path.exists(path) and not exist_ok:
        raise FileExistsError(f"'{path}' already exists. Choose another --name or delete it first.")
    os.makedirs(path, exist_ok=True)


def set_log_path(path: str) -> None:
    global _log_path
    _log_path = path
    os.makedirs(path, exist_ok=True)


def log(obj, filename: str = "train") -> None:
    print(obj, flush=True)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename + ".txt"), "a") as f:
            print(obj, file=f)


class Averager:
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n: float = 1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer:
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_str(t: float) -> str:
    if t >= 3600:
        return "{:.1f}h".format(t / 3600)
    if t >= 60:
        return "{:.1f}m".format(t / 60)
    return "{:.1f}s".format(t)
