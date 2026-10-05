"""Evaluate a trained DVLA-RL++ checkpoint on novel classes (2,000 episodes by default).

Examples:
    python scripts/test.py --dataset miniImageNet --shot 5 --gpu 0
    python scripts/test.py --dataset miniImageNet --test_dataset Places --shot 1   # cross-domain
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import build_config, build_parser  # noqa: E402


def main():
    parser = build_parser()
    parser.add_argument("--test_dataset", type=str, default="", help="target dataset for cross-domain evaluation")
    parser.add_argument("--test_split", type=str, default="novel")
    parser.add_argument("--episode", type=int, default=2000)
    parser.add_argument("--model_ckpt", type=str, default="", help="trained checkpoint; default <save_dir>/<dataset>/<name>_<shot>shot/best.pth")
    parser.add_argument("--log_dir", type=str, default=os.path.join(ROOT, "results", "logs"))
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    test_dataset = args.test_dataset or args.dataset
    if not args.model_ckpt:
        args.model_ckpt = os.path.join(args.save_dir, args.dataset, f"{args.name}_{args.shot}shot", "best.pth")
    if not args.semantic:
        cand = os.path.join(ROOT, "data", "semantic", f"{test_dataset}_banks.pth")
        legacy = os.path.join(ROOT, "data", "semantic", f"{test_dataset}_semantic_clip_cot.pth")
        args.semantic = cand if os.path.exists(cand) else legacy

    import torch
    from torch.utils.data import DataLoader

    from dvla_rlpp import utils
    from dvla_rlpp.data import EpisodeSampler, SemanticBankStore, build_episodic_datasets
    from dvla_rlpp.engine import evaluate, format_summary
    from dvla_rlpp.models import DVLARLpp

    utils.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.set_log_path(args.log_dir)
    log_name = f"test_{args.dataset}" + (f"_to_{test_dataset}" if test_dataset != args.dataset else "") + f"_{args.shot}shot"

    ckpt = torch.load(args.model_ckpt, map_location="cpu")
    saved = ckpt.get("args", {})
    for k, v in saved.items():                      # model hyper-parameters follow the training run
        if k in ("tau_c", "tau_b", "tau_s", "tau_a", "lam", "xi0", "kappa_o", "max_local", "kappa_g", "gamma", "eta",
                 "eps_p", "no_csp", "no_cfg", "gate", "uniform_allocation", "positive_only", "no_neutral",
                 "fixed_margin", "no_intrinsic_fallback", "state_baseline_only", "image_size"):
            setattr(args, k, v)
    cfg = build_config(args)

    train_set, test_set = build_episodic_datasets(test_dataset, args.data_root, args.image_size, eval_split=args.test_split)
    sampler = EpisodeSampler(test_set.targets, args.episode, args.way, args.shot + args.query, fix_seed=True)
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    store = SemanticBankStore(args.semantic, max_local=args.max_local, device=device)

    model = DVLARLpp(cfg, num_classes=ckpt["state_dict"]["backbone.head.weight"].shape[0])
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    utils.log("checkpoint {} (epoch {}, val acc {:.4f})".format(args.model_ckpt, ckpt.get("epoch"), ckpt.get("val_acc", 0.0)), log_name)
    utils.log("test dataset {}: {} images, {} classes".format(test_dataset, len(test_set), len(test_set.classes)), log_name)

    summary = evaluate(model, loader, test_set, test_set.idx_to_name(test_dataset), store, args.way, args.shot,
                       args.query, device, desc=f"{args.way}-way {args.shot}-shot")
    utils.log("{} {}-way {}-shot: {}".format(test_dataset, args.way, args.shot, format_summary(summary)), log_name)


if __name__ == "__main__":
    main()
