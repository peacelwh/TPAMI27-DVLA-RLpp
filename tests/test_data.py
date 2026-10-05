"""Episode sampler, dataset builders and semantic bank store."""
import os

import torch

from dvla_rlpp.data import EpisodeSampler, SemanticBankStore, build_episodic_datasets, class_display_name
from tests.conftest import N_CLASSES, N_IMAGES, TEXT_DIM


def test_episode_sampler_shapes_and_determinism():
    labels = [i // 10 for i in range(50)]
    s1 = EpisodeSampler(labels, 7, 3, 4, fix_seed=True)
    s2 = EpisodeSampler(labels, 7, 3, 4, fix_seed=True)
    b1, b2 = list(s1), list(s2)
    assert len(b1) == 7 and all(b.shape == (12,) for b in b1)
    assert all(torch.equal(x, y) for x, y in zip(b1, b2))
    for b in b1:
        lab = torch.tensor(labels)[b].view(3, 4)
        assert all((row == row[0]).all() for row in lab)      # n_per images from the same class


def test_class_display_name():
    assert class_display_name("FG-CUB", "001.Black_footed_Albatross") == "Black_footed_Albatross"
    assert class_display_name("FG-Dogs", "n02085620-Chihuahua") == "Chihuahua"
    assert class_display_name("miniImageNet", "n01532829") == "n01532829"


def test_episodic_datasets(synthetic_root):
    train, novel = build_episodic_datasets("toy", synthetic_root, image_size=64)
    assert len(train) == N_CLASSES * N_IMAGES and len(novel.classes) == N_CLASSES
    img, label, idx = train[0]
    assert img.shape == (3, 64, 64) and idx == 0
    assert train.relative_path(0).startswith("base_class0")
    assert train.idx_to_name("toy")[0] == "base_class0"


def test_semantic_store(bank_file):
    store = SemanticBankStore(bank_file, max_local=3)
    b = store.bank("base_class0", ["base/base_class0/img0.png"])
    assert b.intrinsic_local.shape[0] <= 3 and b.intrinsic_local.shape[1] == TEXT_DIM
    assert torch.allclose(b.intrinsic_local.norm(dim=-1), torch.ones(b.intrinsic_local.shape[0]), atol=1e-5)
    assert b.has_nuisance
    assert not store.bank("base_class1").has_nuisance
    t = store.episode_tensors(["base_class0", "base_class1", "novel_class2"])
    assert t.num_classes == 3
    assert t.has_nuisance.tolist() == [True, False, True]
    assert t.overlap[1] == 0.0 and 0.0 <= t.overlap[0] <= 1.0
    assert t.pos_local_flat.shape[0] == sum(store.bank(n).intrinsic_local.shape[0]
                                            for n in ["base_class0", "base_class1", "novel_class2"])


def test_legacy_format(tmp_path):
    sem = {"semantic_feature": {"cat": torch.randn(TEXT_DIM), "dog": torch.randn(TEXT_DIM)}}
    attr = {"attribute_features": {"cat": torch.randn(4, TEXT_DIM), "dog": torch.randn(2, TEXT_DIM)}}
    torch.save(sem, tmp_path / "toy_semantic_clip_cot.pth")
    torch.save(attr, tmp_path / "toy_attributes_clip.pth")
    store = SemanticBankStore(str(tmp_path / "toy_semantic_clip_cot.pth"))
    b = store.bank("cat")
    assert b.intrinsic_local.shape == (4, TEXT_DIM) and b.intrinsic_global.shape == (1, TEXT_DIM)
    assert not b.has_nuisance
    assert store.meta["source"] == "legacy-dvla-rl"
    assert os.path.exists(tmp_path / "toy_attributes_clip.pth")
