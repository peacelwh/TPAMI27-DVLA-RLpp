"""Alternating representation / policy learning (Section 3.4; Appendix S5, steps 1-8).

Per epoch:
  * warm-up epochs: the representation and the anchor map are trained with the
    classification loss under the frozen reference gate, and the scoring maps are
    calibrated with the alignment loss. Afterwards the scoring maps are frozen.
  * alternating epochs: ``policy_batch`` episodes are collected with the encoder,
    scorer, text embeddings, reference policy and value baseline frozen. The
    sampled policy and the reference follow separate trajectories from the same
    episode, their paired utilities define the advantage, and only the gate
    parameters are updated with CF-PPO. The value baseline is then refitted.
    ``rep_batch`` fresh episodes update the encoder and anchor parameters with
    the classification loss under stopped mean actions. Stored trajectories are
    discarded whenever the representation changes.
"""
import os
import random
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from torch.optim import Adam, AdamW

from .. import utils
from ..models.dvla_rlpp import LAYER_KEYS, DVLARLpp
from ..models.policy import cf_ppo_loss
from .episode import unpack_episode
from .evaluator import evaluate, format_summary


class Trainer:
    def __init__(self, model: DVLARLpp, args, train_loader, train_set, val_loader, val_set, store, device, writer):
        self.model, self.args, self.device, self.writer = model, args, device, writer
        self.train_loader, self.train_set, self.val_loader, self.val_set, self.store = (
            train_loader, train_set, val_loader, val_set, store)
        self.train_names = train_set.idx_to_name(args.dataset)
        self.val_names = val_set.idx_to_name(args.dataset)
        pc = model.cfg.policy

        backbone_ids = {id(p) for p in model.backbone.parameters()}
        rep_params = model.representation_parameters()
        new_params = [p for p in rep_params if id(p) not in backbone_ids]
        self.rep_opt = AdamW([{"params": new_params, "lr": args.lr, "weight_decay": 1e-4},
                              {"params": list(model.backbone.parameters()), "lr": args.backbone_lr}],
                             weight_decay=1e-4)
        self.scorer_opt = AdamW(model.scorer_parameters(), lr=args.lr, weight_decay=1e-4)
        self.policy_opt = Adam(model.policy_parameters(), lr=pc.policy_lr)
        self.value_opt = Adam(model.value_parameters(), lr=pc.value_lr)
        self.global_step = 0
        self.scorers_frozen = False

    # ------------------------------------------------------------------ #
    def _episode(self, batch):
        a = self.args
        return unpack_episode(batch, self.train_set, self.train_names, self.store, a.way, a.shot, a.query, self.device)

    def _reward(self, res, labels):
        pc = self.model.cfg.policy
        loss_cls = self.model.classification_loss(res.logits, labels)
        return -loss_cls - pc.gamma * res.cost, loss_cls

    # ------------------------------------------------------------------ #
    # warm-up (reference gate, scorer calibration)
    # ------------------------------------------------------------------ #
    def warmup_step(self, ep):
        a = self.args
        self.model.train()
        res = self.model.run_episode(ep.support, ep.query, ep.banks, a.way, a.shot, mode="reference",
                                     record_cost=False, record_tokens=True)
        loss_cls = self.model.classification_loss(res.logits, ep.labels)
        loss_align = self.model.alignment_loss(res.support, ep.banks, a.way, a.shot)
        loss = loss_cls + a.align_weight * loss_align
        self.rep_opt.zero_grad()
        self.scorer_opt.zero_grad()
        loss.backward()
        self.rep_opt.step()
        self.scorer_opt.step()
        return {"loss": loss.item(), "loss_cls": loss_cls.item(), "loss_align": loss_align.item(),
                "acc": utils.compute_acc(res.logits, ep.labels)}

    # ------------------------------------------------------------------ #
    # policy batch (steps 2-4)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def collect(self, ep) -> Dict:
        a, pc, m = self.args, self.model.cfg.policy, self.model
        m.eval()
        ref = m.run_episode(ep.support, ep.query, ep.banks, a.way, a.shot, mode="reference")
        r0, loss0 = self._reward(ref, ep.labels)
        smp = m.run_episode(ep.support, ep.query, ep.banks, a.way, a.shot, mode="sample")
        r, loss = self._reward(smp, ep.labels)
        deviation = ((smp.support.action_vector - ref.support.action_vector) ** 2).mean()
        r = r - pc.eta * deviation                                              # Eq. (14)
        states = [smp.support.states[k] for k in LAYER_KEYS]
        actions = smp.support.action_vector.detach()
        old_log_prob = m.policy.log_prob(states, actions)
        baseline = m.value(states)                                              # fixed before the update
        paired = (r - r0).expand(len(states))
        target = r.expand(len(states)) if pc.state_baseline_only else paired
        advantage = target - baseline
        return {"states": states, "actions": actions, "old_log_prob": old_log_prob, "advantage": advantage,
                "paired": paired, "reward": r.item(), "reward_ref": r0.item(), "cost": smp.cost.item(),
                "cost_ref": ref.cost.item(), "acc": utils.compute_acc(smp.logits, ep.labels),
                "acc_ref": utils.compute_acc(ref.logits, ep.labels)}

    def update_policy(self, buffer: List[Dict]) -> Dict:
        pc, m = self.model.cfg.policy, self.model
        states = [s for b in buffer for s in b["states"]]
        actions = torch.cat([b["actions"] for b in buffer])
        old_log_prob = torch.cat([b["old_log_prob"] for b in buffer])
        advantage = torch.cat([b["advantage"] for b in buffer])
        n = len(states)
        stats = defaultdict(list)
        per_episode = len(LAYER_KEYS)
        mb = max(per_episode, pc.ppo_minibatch * per_episode)
        for _ in range(pc.ppo_epochs):
            order = list(range(0, n, per_episode))
            random.shuffle(order)
            for start in range(0, len(order), mb // per_episode):
                idx = [i + j for i in order[start: start + mb // per_episode] for j in range(per_episode)]
                new_log_prob = m.policy.log_prob([states[i] for i in idx], actions[idx])
                loss, active = cf_ppo_loss(new_log_prob, old_log_prob[idx], advantage[idx], pc.eps_p)   # Eq. (18)
                self.policy_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.policy_parameters(), 1.0)
                self.policy_opt.step()
                stats["policy_loss"].append(loss.item())
                stats["clip_active"].append(active.item())
        # value baseline fitted after the policy step, Eq. (16)
        paired = torch.cat([b["paired"] for b in buffer]).detach()
        for _ in range(pc.ppo_epochs):
            value_loss = ((m.value(states) - paired) ** 2).mean()
            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()
            stats["value_loss"].append(value_loss.item())
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out["paired_return"] = paired.mean().item()
        out["cost_increase_freq"] = float(np.mean([b["cost"] > b["cost_ref"] for b in buffer]))
        out["reward"] = float(np.mean([b["reward"] for b in buffer]))
        out["acc_sampled"] = float(np.mean([b["acc"] for b in buffer]))
        out["acc_reference"] = float(np.mean([b["acc_ref"] for b in buffer]))
        return out

    # ------------------------------------------------------------------ #
    # representation step (step 7)
    # ------------------------------------------------------------------ #
    def representation_step(self, ep) -> Dict:
        a, m = self.args, self.model
        m.train()
        res = m.run_episode(ep.support, ep.query, ep.banks, a.way, a.shot, mode=m.gate_mode(sample=False),
                            record_cost=False)
        loss = m.classification_loss(res.logits, ep.labels)
        self.rep_opt.zero_grad()
        loss.backward()
        self.rep_opt.step()
        return {"loss": loss.item(), "acc": utils.compute_acc(res.logits, ep.labels)}

    # ------------------------------------------------------------------ #
    def train_epoch(self, epoch: int) -> Dict:
        a = self.args
        warmup = epoch <= a.warmup_epochs
        policy_active = (not warmup) and self.model.cfg.use_cfg and self.model.cfg.policy.gate == "policy"
        if not warmup and not self.scorers_frozen:
            self.model.csp.freeze_scorers()
            self.scorers_frozen = True
            utils.log("scoring maps frozen after warm-up")

        aves = defaultdict(utils.Averager)
        buffer, phase, count = [], "policy" if policy_active else "rep", 0
        for batch in self.train_loader:
            ep = self._episode(batch)
            if warmup:
                out = self.warmup_step(ep)
            elif phase == "policy":
                buffer.append(self.collect(ep))
                count += 1
                out = {}
                if count >= a.policy_batch:
                    out = self.update_policy(buffer)
                    for k, v in out.items():
                        self.writer.add_scalar(f"policy/{k}", v, self.global_step)
                    buffer, phase, count = [], "rep", 0        # trajectories discarded
            else:
                out = self.representation_step(ep)
                count += 1
                if policy_active and count >= a.rep_batch:
                    phase, count = "policy", 0
            for k, v in out.items():
                aves[k].add(v)
                self.writer.add_scalar(f"train/{k}", v, self.global_step)
            self.global_step += 1
        return {k: v.item() for k, v in aves.items()}

    def validate(self) -> Dict:
        a = self.args
        return evaluate(self.model, self.val_loader, self.val_set, self.val_names, self.store, a.way, a.shot,
                        a.query, self.device, desc="val", progress=False)

    # ------------------------------------------------------------------ #
    def fit(self):
        a = self.args
        best, timer_used, timer_epoch = 0.0, utils.Timer(), utils.Timer()
        for epoch in range(1, a.epoch + 1):
            timer_epoch.s()
            train_stats = self.train_epoch(epoch)
            val = self.validate()
            self.writer.add_scalar("val/acc", val["acc"], epoch)
            msg = "epoch {} [{}] ".format(epoch, "warmup" if epoch <= a.warmup_epochs else "alternate")
            msg += ", ".join("{}={:.4f}".format(k, v) for k, v in sorted(train_stats.items()))
            msg += " | val " + format_summary(val)
            ckpt = {"epoch": epoch, "state_dict": self.model.state_dict(), "val_acc": val["acc"],
                    "args": vars(a)}
            if val["acc"] > best:
                best = val["acc"]
                torch.save(ckpt, os.path.join(a.work_dir, "best.pth"))
            if a.save_every and epoch % a.save_every == 0:
                torch.save(ckpt, os.path.join(a.work_dir, f"epoch-{epoch}.pth"))
            torch.save(ckpt, os.path.join(a.work_dir, "last.pth"))
            msg += " | best {:.4f}, {} {}/{}".format(best, utils.time_str(timer_epoch.t()), utils.time_str(timer_used.t()),
                                                    utils.time_str(timer_used.t() / epoch * a.epoch))
            utils.log(msg)
            self.writer.flush()
        return best
