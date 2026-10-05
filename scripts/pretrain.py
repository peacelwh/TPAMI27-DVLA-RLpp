"""Backbone pre-training on the base split (whole-image classification with Mixup/CutMix).

    python scripts/pretrain.py --dataset miniImageNet --epoch 800 --gpu 0
    python scripts/pretrain.py --dataset tieredImageNet --epoch 300 --gpu 0

The resulting ``best-1shot.pth`` / ``best-5shot.pth`` is copied to
``data/checkpoint/visformer-<dataset>.pth`` for meta-tuning.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--name", default="visformer-t")
parser.add_argument("--dataset", type=str, default="miniImageNet")
parser.add_argument("--data_root", type=str, default=os.path.join(ROOT, "data", "datasets"))
parser.add_argument("--save_dir", type=str, default=os.path.join(ROOT, "results", "pretrain"))
parser.add_argument("--no_augment", action="store_true")
parser.add_argument("--no_repeat_aug", action="store_true")
parser.add_argument("--no_mixup", action="store_true")
parser.add_argument("--mixup", type=float, default=0.3)
parser.add_argument("--cutmix", type=float, default=1.0)
parser.add_argument("--mixup_prob", type=float, default=1.0)
parser.add_argument("--mixup_switch_prob", type=float, default=0.5)
parser.add_argument("--smoothing", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--ef_epoch", type=int, default=50)
parser.add_argument("--episode", type=int, default=600)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--epoch", type=int, default=800)
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--gpu", default="0")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from timm.data import Mixup  # noqa: E402
from timm.optim import AdamW  # noqa: E402
from timm.scheduler import CosineLRScheduler  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from dvla_rlpp import utils  # noqa: E402
from dvla_rlpp.data import EpisodeSampler, PretrainDataset, RepeatSampler  # noqa: E402
from dvla_rlpp.models import visformer_tiny  # noqa: E402


def main():
    utils.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(args.save_dir, args.dataset, args.name)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, "tensorboard"))
    utils.log(vars(args))

    train_set = PretrainDataset(args.dataset, args.data_root, "train", args.image_size, augment=not args.no_augment)
    if not args.no_repeat_aug:
        train_loader = DataLoader(train_set, batch_sampler=RepeatSampler(train_set, args.batch_size, 2),
                                  num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
    num_classes = len(train_set.dataset.classes)
    utils.log("train dataset: {} images, {} classes".format(len(train_set), num_classes))

    fs_set = PretrainDataset(args.dataset, args.data_root, "val", args.image_size)
    n_way, n_shots = 5, [1, 5]
    fs_loaders = [DataLoader(fs_set, batch_sampler=EpisodeSampler(fs_set.dataset.targets, args.episode, n_way, s + 15),
                             num_workers=args.num_workers) for s in n_shots]

    model = visformer_tiny(num_classes=num_classes, img_size=args.image_size).to(device)
    utils.log("num params: {}".format(utils.compute_n_params(model)))
    optimizer = AdamW(model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=args.lr, weight_decay=5e-2)
    scheduler = CosineLRScheduler(optimizer, warmup_lr_init=1e-6, t_initial=args.epoch, k_decay=0.1, warmup_t=5)
    mixup_fn = None if args.no_mixup else Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, prob=args.mixup_prob,
                                                switch_prob=args.mixup_switch_prob, mode="batch",
                                                label_smoothing=args.smoothing, num_classes=num_classes)
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    best = {1: 0.0, 5: 0.0}
    timer_used, timer_epoch = utils.Timer(), utils.Timer()
    for epoch in range(start_epoch, args.epoch + 1):
        timer_epoch.s()
        aves = {k: utils.Averager() for k in ["tl", "ta"] + [f"fsa-{s}" for s in n_shots]}
        model.train()
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            if mixup_fn is not None:
                data, label = mixup_fn(data, label)
            logits, _ = model(data)
            loss = F.cross_entropy(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            aves["tl"].add(loss.item())
            aves["ta"].add(utils.compute_acc_mix(logits, label))

        if epoch % args.ef_epoch == 0 or epoch == 1:
            model.eval()
            for i, n_shot in enumerate(n_shots):
                for episode in fs_loaders[i]:
                    image = episode[0].to(device)
                    labels = torch.arange(n_way).unsqueeze(-1).repeat(1, 15).view(-1).to(device)
                    with torch.no_grad():
                        _, feat = model(image)
                        feat = feat.view(n_way, n_shot + 15, -1)
                        sup, que = feat[:, :n_shot].mean(1), feat[:, n_shot:].reshape(n_way * 15, -1)
                        logits = F.normalize(que, dim=-1) @ F.normalize(sup, dim=-1).t()
                    aves[f"fsa-{n_shot}"].add(utils.compute_acc(logits, labels))
        scheduler.step(epoch)

        stats = {k: v.item() for k, v in aves.items()}
        writer.add_scalars("loss", {"train": stats["tl"]}, epoch)
        writer.add_scalars("acc", {"train": stats["ta"]}, epoch)
        msg = "epoch {}, train {:.4f}|{:.4f}".format(epoch, stats["tl"], stats["ta"])
        ckpt = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        if epoch % args.ef_epoch == 0 or epoch == 1:
            msg += ", fs " + " ".join("{}: {:.4f}".format(s, stats[f"fsa-{s}"]) for s in n_shots)
            for s in n_shots:
                writer.add_scalars("acc", {f"fsa-{s}": stats[f"fsa-{s}"]}, epoch)
                if stats[f"fsa-{s}"] > best[s]:
                    best[s] = stats[f"fsa-{s}"]
                    torch.save(ckpt, os.path.join(save_path, f"best-{s}shot.pth"))
        if epoch % 100 == 0:
            torch.save(ckpt, os.path.join(save_path, f"epoch-{epoch}.pth"))
        torch.save(ckpt, os.path.join(save_path, "last.pth"))
        msg += ", {} {}/{}".format(utils.time_str(timer_epoch.t()), utils.time_str(timer_used.t()),
                                   utils.time_str(timer_used.t() / epoch * args.epoch))
        utils.log(msg)
        writer.flush()
    utils.log("best 1-shot: {:.4f}, best 5-shot: {:.4f}".format(best[1], best[5]))


if __name__ == "__main__":
    main()
