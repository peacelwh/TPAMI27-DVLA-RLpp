"""Shared fixtures: a tiny synthetic ImageFolder dataset and a synthetic semantic bank file."""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

TEXT_DIM = 512
IMAGE_SIZE = 64
N_CLASSES = 4
N_IMAGES = 6


def _unit(n):
    return F.normalize(torch.randn(n, TEXT_DIM), dim=-1)


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    """data_root/<dataset>/{base,val,novel}/<class>/*.png with tiny random images."""
    from PIL import Image
    root = tmp_path_factory.mktemp("datasets")
    rng = np.random.RandomState(0)
    for split in ("base", "val", "novel"):
        for c in range(N_CLASSES):
            d = root / "toy" / split / f"{split}_class{c}"
            d.mkdir(parents=True)
            for i in range(N_IMAGES):
                arr = rng.randint(0, 255, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"img{i}.png")
    return str(root)


@pytest.fixture(scope="session")
def bank_file(tmp_path_factory, synthetic_root):
    torch.manual_seed(0)
    classes = {}
    for split in ("base", "val", "novel"):
        for c in range(N_CLASSES):
            name = f"{split}_class{c}"
            nuis = c % 2 == 0          # every other class has an empty nuisance bank
            classes[name] = {
                "name": _unit(1)[0],
                "intrinsic": {"local": _unit(5), "global": _unit(1)},
                "nuisance": {"local": _unit(3) if nuis else torch.zeros(0, TEXT_DIM),
                             "global": _unit(1) if nuis else torch.zeros(0, TEXT_DIM)},
            }
    supports = {"base/base_class0/img0.png": {"intrinsic": {"local": _unit(2), "global": torch.zeros(0, TEXT_DIM)},
                                               "nuisance": {"local": _unit(1), "global": torch.zeros(0, TEXT_DIM)}}}
    path = tmp_path_factory.mktemp("semantic") / "toy_banks.pth"
    torch.save({"text_dim": TEXT_DIM, "classes": classes, "supports": supports, "meta": {"source": "test"}}, path)
    return str(path)


@pytest.fixture
def tiny_model():
    from dvla_rlpp.config import ModelConfig
    from dvla_rlpp.models import DVLARLpp
    torch.manual_seed(0)
    return DVLARLpp(ModelConfig(image_size=IMAGE_SIZE), num_classes=N_CLASSES)


@pytest.fixture
def episode_banks():
    from dvla_rlpp.data.semantic import BankTensors, SemanticBank
    torch.manual_seed(1)

    def bank(i, nuis):
        return SemanticBank(f"c{i}", _unit(1)[0], _unit(5), _unit(1),
                            _unit(3) if nuis else torch.zeros(0, TEXT_DIM),
                            _unit(1) if nuis else torch.zeros(0, TEXT_DIM))
    return BankTensors.from_banks([bank(0, True), bank(1, True), bank(2, False)], "cpu")
