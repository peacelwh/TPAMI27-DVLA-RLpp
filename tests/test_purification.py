"""Properties of complementary semantic purification (Appendix S2-S3)."""
import torch
import torch.nn.functional as F

from dvla_rlpp.config import PurificationConfig
from dvla_rlpp.models.purification import CSPModule, project_capped_simplex, size_normalised_lse


def test_projection_kkt_properties():
    torch.manual_seed(0)
    b = torch.randn(64, 20) * 2
    a = project_capped_simplex(b)
    assert (a >= 0).all()
    assert (a.sum(-1) <= 1 + 1e-5).all()
    # a_i > 0  =>  b_i > delta >= 0, i.e. retained tokens clear the margin, Eq. (S7)
    assert (b[a > 0] > 0).all()
    # rows whose positive part already has mass <= 1 are left untouched (delta = 0)
    small = b.clamp(min=0).sum(-1) <= 1
    assert torch.allclose(a[small], b[small].clamp(min=0))
    # rows with excess mass are projected onto the simplex boundary
    assert torch.allclose(a[~small].sum(-1), torch.ones((~small).sum()), atol=1e-5)


def test_projection_matches_brute_force():
    torch.manual_seed(1)
    b = torch.randn(5, 8)
    a = project_capped_simplex(b)
    for i in range(5):
        # brute force: minimise 0.5||a-b||^2 s.t. a>=0, sum<=1 via projected gradient descent
        x = torch.zeros(8, requires_grad=True)
        opt = torch.optim.SGD([x], lr=0.1)
        for _ in range(3000):
            opt.zero_grad()
            ((x - b[i]) ** 2).sum().mul(0.5).backward()
            opt.step()
            with torch.no_grad():
                x.copy_(project_capped_simplex(x))
        assert torch.allclose(a[i], x.detach(), atol=1e-3)


def test_projection_is_nonexpansive():
    torch.manual_seed(2)
    b1, b2 = torch.randn(32, 15), torch.randn(32, 15)
    a1, a2 = project_capped_simplex(b1), project_capped_simplex(b2)
    assert ((a1 - a2).norm(dim=-1) <= (b1 - b2).norm(dim=-1) + 1e-6).all()      # Eq. (S10)


def test_force_unit_mass():
    a = project_capped_simplex(torch.randn(10, 6), force_unit_mass=True)
    assert torch.allclose(a.sum(-1), torch.ones(10), atol=1e-5)


def test_size_normalised_lse_is_invariant_to_duplicated_phrases():
    torch.manual_seed(3)
    v = F.normalize(torch.randn(2, 3, 7, 16), dim=-1)
    bank = F.normalize(torch.randn(2, 4, 16), dim=-1)
    mask = torch.ones(2, 4, dtype=torch.bool)
    u1 = size_normalised_lse(v, bank, mask, 0.1)
    dup = torch.cat([bank, bank], 1)
    u2 = size_normalised_lse(v, dup, torch.ones(2, 8, dtype=torch.bool), 0.1)
    assert torch.allclose(u1, u2, atol=1e-5)
    # empty bank -> zero response
    u0 = size_normalised_lse(v, bank, torch.zeros(2, 4, dtype=torch.bool), 0.1)
    assert torch.all(u0 == 0)


def test_margin_and_purification(episode_banks):
    torch.manual_seed(4)
    csp = CSPModule({"stage2": 32, "stage3": 48, "final": 48}, 512, 48, PurificationConfig())
    xi = csp.margin(episode_banks)
    assert xi.shape == (3,)
    assert xi[2] == 0.0                                  # empty nuisance bank -> zero margin
    assert (xi[:2] >= csp.cfg.xi0).all()
    tokens = torch.randn(3, 2, 9, 48)
    out = csp.purify(tokens, episode_banks)
    assert out.prototypes.shape == (3, 48)
    assert torch.allclose(out.prototypes.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert (out.allocation >= 0).all() and (out.retained_mass <= 1 + 1e-5).all()
    # retained tokens clear the rejection margin, Eq. (S7)
    d, xi3 = out.evidence, out.margin[:, None, None]
    assert (d[out.allocation > 0] > xi3.expand_as(d)[out.allocation > 0]).all()
    assert all(k in out.diagnostics() for k in ("retention", "neutral_mass", "candidate_deficit"))


def test_complete_rejection_returns_anchor(episode_banks):
    cfg = PurificationConfig(xi0=10.0)                   # huge margin rejects every token
    csp = CSPModule({"final": 48}, 512, 48, cfg)
    tokens = torch.randn(3, 2, 9, 48)
    out = csp.purify(tokens, episode_banks)
    assert torch.all(out.allocation[:2] == 0)            # classes with a nuisance bank: all rejected
    assert torch.allclose(out.prototypes[:2], out.anchor[:2], atol=1e-5)   # w = r exactly, Eq. (8)


def test_ablation_switches(episode_banks):
    tokens = torch.randn(3, 2, 9, 48)
    uni = CSPModule({"final": 48}, 512, 48, PurificationConfig(uniform_allocation=True)).purify(tokens, episode_banks)
    assert torch.allclose(uni.allocation, torch.full_like(uni.allocation, 1.0 / 9))
    nn_ = CSPModule({"final": 48}, 512, 48, PurificationConfig(no_neutral=True)).purify(tokens, episode_banks)
    assert torch.allclose(nn_.retained_mass, torch.ones(3, 2), atol=1e-5)
    pos = CSPModule({"final": 48}, 512, 48, PurificationConfig(positive_only=True))
    assert torch.all(pos.margin(episode_banks) == 0)


def test_alignment_loss_is_finite(episode_banks):
    csp = CSPModule({"stage2": 32, "final": 48}, 512, 48, PurificationConfig())
    loss = csp.alignment_loss("stage2", torch.randn(3, 2, 9, 32), episode_banks)
    assert torch.isfinite(loss)
    loss.backward()
    assert csp.scorers["stage2"].weight.grad is not None
