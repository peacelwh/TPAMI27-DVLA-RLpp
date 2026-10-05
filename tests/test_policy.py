"""Beta gate policy, value baseline, reference gate and the CF-PPO surrogate."""
import torch

from dvla_rlpp.config import PolicyConfig
from dvla_rlpp.models.policy import (BetaGatePolicy, GateState, ReferenceGate, StaticGate, ValueBaseline,
                                     cf_ppo_loss)

DIMS = {"stage2": 32, "stage3": 48}


def _state(key, i):
    torch.manual_seed(i)
    return GateState(key, torch.randn(DIMS[key]), torch.randn(512), torch.randn(512), torch.rand(6),
                     list(DIMS).index(key))


def test_policy_mean_is_bounded_and_initially_half():
    cfg = PolicyConfig(eps_g=0.05)
    pol = BetaGatePolicy(DIMS, 512, cfg)
    s = _state("stage2", 0)
    a, logp = pol.act(s, sample=False)
    assert abs(a.item() - 0.5) < 1e-6                     # zero-initialised head
    assert torch.isfinite(logp)
    for p in pol.parameters():
        p.data.normal_(0, 5)
    for i in range(20):
        a, _ = pol.act(_state("stage3", i), sample=False)
        assert cfg.eps_g - 1e-6 <= a.item() <= 1 - cfg.eps_g + 1e-6
    a, logp = pol.act(_state("stage2", 3), sample=True)
    assert 0 < a.item() < 1 and torch.isfinite(logp)


def test_batched_log_prob_matches_single():
    pol = BetaGatePolicy(DIMS, 512, PolicyConfig())
    states = [_state("stage2", 1), _state("stage3", 2), _state("stage2", 3)]
    actions = torch.tensor([0.3, 0.6, 0.9])
    batched = pol.log_prob(states, actions)
    for i, s in enumerate(states):
        single = pol.distribution(*pol._unbatch(s)).log_prob(actions[i: i + 1]).squeeze(0)
        assert torch.allclose(batched[i], single, atol=1e-5)
    assert ValueBaseline(DIMS, 512, PolicyConfig())(states).shape == (3,)


def test_cf_ppo_surrogate_at_behaviour_parameters():
    adv = torch.tensor([0.5, -0.2, 1.0])
    old = torch.tensor([-0.1, -0.3, -0.2])
    loss, active = cf_ppo_loss(old.clone(), old, adv, 0.2)
    assert torch.allclose(loss, -adv.mean())              # ratio 1 everywhere
    assert active.item() == 0.0
    # active clipping: positive advantage with ratio above 1 + eps
    new = old + torch.tensor([0.5, 0.0, 0.0])
    loss2, active2 = cf_ppo_loss(new, old, adv, 0.2)
    assert torch.isfinite(loss2) and abs(active2.item() - 1 / 3) < 1e-6
    # clipped branch is locally constant: no gradient through the clipped sample
    new = (old + torch.tensor([0.5, 0.0, 0.0])).requires_grad_(True)
    loss3, _ = cf_ppo_loss(new, old, adv, 0.2)
    loss3.backward()
    assert new.grad[0].item() == 0.0 and new.grad[2].item() != 0.0


def test_reference_gate_mean_action_and_loading():
    ref = ReferenceGate(32, 512)
    x = torch.randn(6, 32, 4, 4)
    a = ref.mean_action(x, torch.randn(512))
    assert 0.0 <= a.item() <= 1.0 and a.ndim == 0
    sd = {"rl_gate1.policy_net.1.weight": torch.ones(3, 32, 1, 1), "rl_gate1.policy_net.1.bias": torch.zeros(3),
          "t2i.weight": torch.ones(32, 512)}
    sub = {k.split("rl_gate1.")[-1]: v for k, v in sd.items() if "policy_net" in k}
    sub["t2i.weight"] = sd["t2i.weight"]
    ref.load_state_dict(sub, strict=False)
    assert torch.all(ref.policy_net[1].weight == 1)


def test_static_gate():
    g = StaticGate(["stage2", "stage3"])
    a = g("stage2")
    assert abs(a.item() - 0.5) < 1e-6
    a.backward()
    assert g.logits["stage2"].grad is not None
