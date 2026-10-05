"""Episode-level behaviour of DVLA-RL++ (shapes, gate modes, parameter boundaries)."""
import torch

from dvla_rlpp.models.dvla_rlpp import LAYER_KEYS

WAY, SHOT, QUERY, SIZE = 3, 2, 4, 64


def _episode():
    torch.manual_seed(0)
    sup = torch.randn(WAY * SHOT, 3, SIZE, SIZE)
    que = torch.randn(WAY * QUERY, 3, SIZE, SIZE)
    labels = torch.arange(WAY).repeat_interleave(QUERY)
    return sup, que, labels


def test_all_gate_modes_produce_logits(tiny_model, episode_banks):
    sup, que, labels = _episode()
    tiny_model.train()
    for mode in ("sample", "mean", "reference", "static"):
        res = tiny_model.run_episode(sup, que, episode_banks, WAY, SHOT, mode=mode)
        assert res.logits.shape == (WAY * QUERY, WAY)
        assert set(res.actions) == set(LAYER_KEYS)
        for a in res.actions.values():
            assert 0.0 <= a.item() <= 1.0
        assert torch.isfinite(res.cost)
        assert res.purification is not None
        loss = tiny_model.classification_loss(res.logits, labels)
        assert torch.isfinite(loss)
    assert set(res.support.costs) == set(LAYER_KEYS)


def test_queries_replay_support_schedule(tiny_model, episode_banks):
    sup, que, _ = _episode()
    tiny_model.eval()
    with torch.no_grad():
        res = tiny_model.run_episode(sup, que, episode_banks, WAY, SHOT, mode="mean")
        for k in LAYER_KEYS:
            assert torch.allclose(res.query.actions[k], res.support.actions[k])
        # deterministic mean policy: identical results across calls
        res2 = tiny_model.run_episode(sup, que, episode_banks, WAY, SHOT, mode="mean")
        assert torch.allclose(res.logits, res2.logits, atol=1e-5)


def test_reference_trajectory_is_independent_of_sampled_actions(tiny_model, episode_banks):
    sup, que, _ = _episode()
    tiny_model.eval()
    with torch.no_grad():
        ref1 = tiny_model.run_episode(sup, que, episode_banks, WAY, SHOT, mode="reference")
        _ = tiny_model.run_episode(sup, que, episode_banks, WAY, SHOT, mode="sample")
        ref2 = tiny_model.run_episode(sup, que, episode_banks, WAY, SHOT, mode="reference")
    for k in LAYER_KEYS:
        assert torch.allclose(ref1.actions[k], ref2.actions[k])
    assert torch.allclose(ref1.logits, ref2.logits, atol=1e-5)


def test_representation_step_does_not_touch_policy_or_scorers(tiny_model, episode_banks):
    sup, que, labels = _episode()
    tiny_model.train()
    tiny_model.csp.freeze_scorers()
    res = tiny_model.run_episode(sup, que, episode_banks, WAY, SHOT, mode="mean", record_cost=False)
    tiny_model.classification_loss(res.logits, labels).backward()
    assert all(p.grad is None for p in tiny_model.policy_parameters())
    assert all(p.grad is None for p in tiny_model.value_parameters())
    assert all(p.grad is None for p in tiny_model.scorer_parameters())
    assert all(not p.requires_grad for p in tiny_model.reference.parameters())
    assert any(p.grad is not None for p in tiny_model.backbone.parameters())
    assert any(p.grad is not None for p in tiny_model.gated.parameters())
    assert tiny_model.csp.anchor_proj.weight.grad is not None


def test_policy_step_touches_only_policy(tiny_model, episode_banks):
    sup, que, labels = _episode()
    tiny_model.eval()
    with torch.no_grad():
        res = tiny_model.run_episode(sup, que, episode_banks, WAY, SHOT, mode="sample")
    states = [res.support.states[k] for k in LAYER_KEYS]
    logp = tiny_model.policy.log_prob(states, res.support.action_vector.detach())
    (-logp.sum()).backward()
    assert any(p.grad is not None for p in tiny_model.policy_parameters())
    assert all(p.grad is None for p in tiny_model.representation_parameters())


def test_alignment_loss_and_backbone_loading(tiny_model, episode_banks):
    sup, que, labels = _episode()
    res = tiny_model.run_episode(sup, que, episode_banks, WAY, SHOT, mode="reference", record_cost=False,
                                 record_tokens=True)
    assert set(res.support.layer_tokens) == set(LAYER_KEYS)
    loss = tiny_model.alignment_loss(res.support, episode_banks, WAY, SHOT)
    assert torch.isfinite(loss)
    sd = tiny_model.backbone.state_dict()
    missing, unexpected = tiny_model.load_backbone(sd)
    assert not missing and not unexpected
    n = tiny_model.load_reference_from_dvla_rl({"rl_gate1.policy_net.1.weight": torch.zeros(3, 192, 1, 1),
                                                "t2i2.weight": torch.zeros(384, 512)})
    assert n == 2


def test_ablation_without_purification(episode_banks):
    from dvla_rlpp.config import ModelConfig
    from dvla_rlpp.models import DVLARLpp
    sup, que, _ = _episode()
    m = DVLARLpp(ModelConfig(image_size=SIZE, use_csp=False, use_cfg=False), num_classes=4).eval()
    with torch.no_grad():
        res = m.run_episode(sup, que, episode_banks, WAY, SHOT)
    assert res.purification is None and res.logits.shape == (WAY * QUERY, WAY)
    assert m.gate_mode(sample=True) == "reference"
