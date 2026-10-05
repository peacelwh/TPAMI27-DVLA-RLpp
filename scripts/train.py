"""Episodic meta-tuning of DVLA-RL++.

Example (5-way 1-shot miniImageNet):
    python scripts/train.py --dataset miniImageNet --shot 1 --gpu 0
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def build_parser():
    p = argparse.ArgumentParser(description="DVLA-RL++ meta-tuning")
    # experiment
    p.add_argument("--name", type=str, default="dvla_rlpp")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_dir", type=str, default=os.path.join(ROOT, "results", "save"))
    p.add_argument("--save_every", type=int, default=25)
    # data
    p.add_argument("--dataset", type=str, default="miniImageNet")
    p.add_argument("--data_root", type=str, default=os.path.join(ROOT, "data", "datasets"))
    p.add_argument("--semantic", type=str, default="", help="bank file; default data/semantic/<dataset>_banks.pth")
    p.add_argument("--checkpoint", type=str, default="", help="pre-trained backbone; default data/checkpoint/visformer-<dataset>.pth")
    p.add_argument("--reference_ckpt", type=str, default="", help="DVLA-RL checkpoint providing the frozen reference gate")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--way", type=int, default=5)
    p.add_argument("--shot", type=int, default=1)
    p.add_argument("--query", type=int, default=15)
    p.add_argument("--val_split", type=str, default="val")
    p.add_argument("--val_episode", type=int, default=600)
    p.add_argument("--train_episodes", type=int, default=0, help="episodes per epoch (0 = dataset size / episode size)")
    # optimisation
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-4, help="learning rate of the added modules")
    p.add_argument("--backbone_lr", type=float, default=1e-6)
    p.add_argument("--align_weight", type=float, default=1.0)
    p.add_argument("--policy_batch", type=int, default=32, help="episodes per policy batch")
    p.add_argument("--rep_batch", type=int, default=32, help="representation episodes between policy batches")
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--ppo_minibatch", type=int, default=16)
    p.add_argument("--policy_lr", type=float, default=1e-4)
    p.add_argument("--value_lr", type=float, default=1e-3)
    # model hyper-parameters (paper defaults)
    p.add_argument("--tau_c", type=float, default=0.2)
    p.add_argument("--tau_b", type=float, default=0.1)
    p.add_argument("--tau_s", type=float, default=0.2)
    p.add_argument("--tau_a", type=float, default=0.1)
    p.add_argument("--lam", type=float, default=0.7)
    p.add_argument("--xi0", type=float, default=0.05)
    p.add_argument("--kappa_o", type=float, default=0.1)
    p.add_argument("--max_local", type=int, default=8)
    p.add_argument("--kappa_g", type=float, default=10.0)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--eps_p", type=float, default=0.2)
    # ablations (Tables 4 and 5)
    p.add_argument("--no_csp", action="store_true", help="uniform visual support prototypes")
    p.add_argument("--no_cfg", action="store_true", help="keep the frozen reference gate")
    p.add_argument("--gate", type=str, default="policy", choices=["policy", "reference", "static"])
    p.add_argument("--uniform_allocation", action="store_true")
    p.add_argument("--positive_only", action="store_true")
    p.add_argument("--no_neutral", action="store_true")
    p.add_argument("--fixed_margin", action="store_true")
    p.add_argument("--no_intrinsic_fallback", action="store_true")
    p.add_argument("--state_baseline_only", action="store_true")
    return p


def build_config(args):
    from dvla_rlpp.config import ModelConfig, PolicyConfig, PurificationConfig
    pur = PurificationConfig(tau_b=args.tau_b, tau_s=args.tau_s, lam=args.lam, xi0=args.xi0, kappa_o=args.kappa_o,
                             tau_a=args.tau_a, max_local=args.max_local, uniform_allocation=args.uniform_allocation,
                             positive_only=args.positive_only, no_neutral=args.no_neutral,
                             fixed_margin=args.fixed_margin, no_intrinsic_fallback=args.no_intrinsic_fallback)
    pol = PolicyConfig(kappa_g=args.kappa_g, gamma=args.gamma, eta=args.eta, eps_p=args.eps_p,
                       ppo_epochs=args.ppo_epochs, ppo_minibatch=args.ppo_minibatch, policy_lr=args.policy_lr,
                       value_lr=args.value_lr, gate=args.gate, state_baseline_only=args.state_baseline_only)
    return ModelConfig(image_size=args.image_size, tau_c=args.tau_c, purification=pur, policy=pol,
                       use_csp=not args.no_csp, use_cfg=not args.no_cfg)


def resolve_paths(args):
    if not args.semantic:
        cand = os.path.join(ROOT, "data", "semantic", f"{args.dataset}_banks.pth")
        legacy = os.path.join(ROOT, "data", "semantic", f"{args.dataset}_semantic_clip_cot.pth")
        args.semantic = cand if os.path.exists(cand) else legacy
    if not args.checkpoint:
        args.checkpoint = os.path.join(ROOT, "data", "checkpoint", f"visformer-{args.dataset}.pth")
    if not args.reference_ckpt:
        cand = os.path.join(ROOT, "data", "checkpoint", f"dvla_rl-{args.dataset}-{args.shot}shot.pth")
        args.reference_ckpt = cand if os.path.exists(cand) else ""
    return args


def main():
    args = resolve_paths(build_parser().parse_args())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    from dvla_rlpp import utils
    from dvla_rlpp.data import EpisodeSampler, SemanticBankStore, build_episodic_datasets
    from dvla_rlpp.engine import Trainer
    from dvla_rlpp.models import DVLARLpp

    utils.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.work_dir = os.path.join(args.save_dir, args.dataset, f"{args.name}_{args.shot}shot")
    utils.ensure_path(args.work_dir, exist_ok=bool(args.resume))
    utils.set_log_path(args.work_dir)
    utils.log(vars(args))
    writer = SummaryWriter(os.path.join(args.work_dir, "tensorboard"))

    train_set, val_set = build_episodic_datasets(args.dataset, args.data_root, args.image_size,
                                                 eval_split=args.val_split)
    utils.log("train dataset: {} images, {} classes".format(len(train_set), len(train_set.classes)))
    utils.log("val dataset:   {} images, {} classes".format(len(val_set), len(val_set.classes)))
    n_episodes = args.train_episodes or int(len(train_set) / (args.way * (args.shot + args.query)))
    train_sampler = EpisodeSampler(train_set.targets, n_episodes, args.way, args.shot + args.query, fix_seed=False)
    val_sampler = EpisodeSampler(val_set.targets, args.val_episode, args.way, args.shot + args.query, fix_seed=True)
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

    store = SemanticBankStore(args.semantic, max_local=args.max_local, device=device)
    utils.log("semantic banks: {} classes, {} support entries ({})".format(len(store.classes), len(store.supports),
                                                                        args.semantic))

    cfg = build_config(args)
    model = DVLARLpp(cfg, num_classes=len(train_set.classes))
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        utils.log(f"resumed from {args.resume} (epoch {ckpt.get('epoch')})")
    else:
        if os.path.exists(args.checkpoint):
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            missing, unexpected = model.load_backbone(ckpt.get("state_dict", ckpt))
            utils.log(f"loaded backbone {args.checkpoint} (missing {len(missing)}, unexpected {len(unexpected)})")
        else:
            utils.log(f"WARNING: pre-trained backbone {args.checkpoint} not found, training from scratch")
        if args.reference_ckpt and os.path.exists(args.reference_ckpt):
            ref = torch.load(args.reference_ckpt, map_location="cpu")
            n = model.load_reference_from_dvla_rl(ref.get("state_dict", ref))
            utils.log(f"loaded {n} reference-gate tensors from {args.reference_ckpt}")
        else:
            utils.log("WARNING: no DVLA-RL reference checkpoint given, the frozen reference gate is randomly initialised")
    model = model.to(device)
    utils.log("num params: {}".format(utils.compute_n_params(model)))

    trainer = Trainer(model, args, train_loader, train_set, val_loader, val_set, store, device, writer)
    best = trainer.fit()
    utils.log("best validation accuracy: {:.4f}".format(best))


if __name__ == "__main__":
    main()
