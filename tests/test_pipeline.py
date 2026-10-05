"""End-to-end: the Trainer schedule and the command-line scripts on the synthetic dataset (CPU)."""
import os
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_trainer_schedule(synthetic_root, bank_file, tmp_path):
    from argparse import Namespace

    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    from dvla_rlpp import utils
    from dvla_rlpp.config import ModelConfig, PolicyConfig
    from dvla_rlpp.data import EpisodeSampler, SemanticBankStore, build_episodic_datasets
    from dvla_rlpp.engine import Trainer
    from dvla_rlpp.models import DVLARLpp

    utils.set_seed(0)
    args = Namespace(dataset="toy", way=3, shot=1, query=2, lr=1e-3, backbone_lr=1e-5, align_weight=1.0,
                     policy_batch=2, rep_batch=2, warmup_epochs=1, epoch=2, save_every=0, work_dir=str(tmp_path))
    train_set, val_set = build_episodic_datasets("toy", synthetic_root, 64, eval_split="val")
    train_loader = DataLoader(train_set, batch_sampler=EpisodeSampler(train_set.targets, 6, 3, 3, fix_seed=False))
    val_loader = DataLoader(val_set, batch_sampler=EpisodeSampler(val_set.targets, 3, 3, 3, fix_seed=True))
    store = SemanticBankStore(bank_file, max_local=4)
    cfg = ModelConfig(image_size=64, policy=PolicyConfig(ppo_epochs=2, ppo_minibatch=2))
    model = DVLARLpp(cfg, num_classes=4)
    utils.set_log_path(str(tmp_path))
    trainer = Trainer(model, args, train_loader, train_set, val_loader, val_set, store, torch.device("cpu"),
                      SummaryWriter(str(tmp_path / "tb")))
    best = trainer.fit()
    assert 0.0 <= best <= 1.0
    assert trainer.scorers_frozen and all(not p.requires_grad for p in model.scorer_parameters())
    assert os.path.exists(tmp_path / "best.pth") and os.path.exists(tmp_path / "last.pth")
    log = open(tmp_path / "train.txt").read()
    assert "epoch 1 [warmup]" in log and "epoch 2 [alternate]" in log and "policy_loss" in log


def test_cli_train_and_test(synthetic_root, bank_file, tmp_path):
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"), CUDA_VISIBLE_DEVICES="")
    common = ["--dataset", "toy", "--data_root", synthetic_root, "--semantic", bank_file, "--image_size", "64",
              "--way", "3", "--shot", "1", "--query", "2", "--num_workers", "0", "--save_dir", str(tmp_path / "save"),
              "--name", "cli", "--checkpoint", str(tmp_path / "missing.pth")]
    train = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--epoch", "2", "--warmup_epochs", "1",
             "--train_episodes", "4", "--val_episode", "2", "--policy_batch", "2", "--rep_batch", "1",
             "--ppo_epochs", "1", "--ppo_minibatch", "2"] + common
    r = subprocess.run(train, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "best validation accuracy" in r.stdout
    test = [sys.executable, os.path.join(ROOT, "scripts", "test.py"), "--episode", "3",
            "--log_dir", str(tmp_path / "logs")] + common
    r = subprocess.run(test, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "3-way 1-shot: acc =" in r.stdout
    assert os.path.exists(tmp_path / "logs" / "test_toy_1shot.txt")
