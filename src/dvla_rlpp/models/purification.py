"""Complementary semantic purification (Section 3.3, Fig. 4; Appendix S1-S3).

Notation follows the paper:
    v  projected, unit-normalised image tokens used for scoring        (N, K, M, d_t)
    u+ / u-   size-normalised log-sum-exp bank responses, Eq. (4)
    d  signed evidence u+ - u-                                          (N, K, M)
    xi rejection margin xi_0 + kappa_o * overlap, Eq. (5)              (N,)
    a  sparse allocation = projection of (d - xi) / tau_s onto {a>=0, 1^T a<=1}, Eq. (6)-(7)
    w  raw support representation lambda * sum_i a_i z_i + (1 - lambda rho) r, Eq. (8)
    p  normalised class prototype, Eq. (9)
    D  candidate deficit (1/M) sum_i [xi - d_i]_+, Eq. (10)
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PurificationConfig
from ..data.semantic import BankTensors


# --------------------------------------------------------------------------- #
# Projection onto the capped simplex {a >= 0, 1^T a <= 1}  (Appendix S2)
# --------------------------------------------------------------------------- #
def project_capped_simplex(b: torch.Tensor, force_unit_mass: bool = False) -> torch.Tensor:
    """Euclidean projection of each row of ``b`` (..., M) onto {a>=0, sum a <= 1}.

    a_i = [b_i - delta]_+ with delta = 0 when sum_i [b_i]_+ <= 1, otherwise delta
    is the unique simplex threshold found by sorting. With ``force_unit_mass``
    the projection is onto the probability simplex (ablation "no neutral option").
    """
    shape = b.shape
    b2 = b.reshape(-1, shape[-1])
    pos = b2.clamp(min=0.0)
    mass = pos.sum(dim=1)
    need = mass > 1.0 if not force_unit_mass else torch.ones_like(mass, dtype=torch.bool)

    sorted_b, _ = torch.sort(b2, dim=1, descending=True)
    cssv = sorted_b.cumsum(dim=1) - 1.0
    ks = torch.arange(1, shape[-1] + 1, device=b.device, dtype=b.dtype).unsqueeze(0)
    cond = (sorted_b - cssv / ks) > 0
    rho = cond.sum(dim=1, keepdim=True).clamp(min=1)
    delta = cssv.gather(1, rho - 1) / rho.to(b.dtype)
    simplex = (b2 - delta).clamp(min=0.0)

    out = torch.where(need.unsqueeze(1), simplex, pos)
    return out.reshape(shape)


def size_normalised_lse(v: torch.Tensor, bank: torch.Tensor, mask: torch.Tensor, tau_b: float) -> torch.Tensor:
    """u = tau_b * log[(1/R) sum_r exp(v . e_r / tau_b)]  for every token, Eq. (4).

    v: (N, K, M, d), bank: (N, R, d), mask: (N, R). Empty banks return 0.
    """
    N, K, M, d = v.shape
    sims = torch.einsum("nkmd,nrd->nkmr", v, bank) / tau_b                   # (N, K, M, R)
    neg_inf = torch.finfo(sims.dtype).min
    sims = sims.masked_fill(~mask[:, None, None, :], neg_inf)
    count = mask.sum(dim=1).clamp(min=1).to(v.dtype)                          # (N,)
    lse = torch.logsumexp(sims, dim=-1) - torch.log(count)[:, None, None]     # (N, K, M)
    u = tau_b * lse
    empty = mask.sum(dim=1) == 0
    return torch.where(empty[:, None, None], torch.zeros_like(u), u)


@dataclass
class PurificationOutput:
    prototypes: torch.Tensor        # (N, d_f)
    allocation: torch.Tensor        # (N, K, M)
    evidence: torch.Tensor          # (N, K, M) signed evidence d
    margin: torch.Tensor            # (N,) xi_c
    retained_mass: torch.Tensor     # (N, K) rho
    deficit: torch.Tensor           # (N, K) D
    fallback: torch.Tensor          # (N,) bool, Eq. (9) second case
    anchor: torch.Tensor            # (N, d_f)

    def diagnostics(self) -> Dict[str, float]:
        a = self.allocation
        return {
            "retention": (a > 0).float().mean().item(),
            "neutral_mass": (1.0 - self.retained_mass).mean().item(),
            "candidate_deficit": self.deficit.mean().item(),
            "mean_evidence": self.evidence.mean().item(),
            "fallback_rate": self.fallback.float().mean().item(),
        }


class CSPModule(nn.Module):
    """Scoring projections P_l, anchor projection G and the purification rule."""

    def __init__(self, dims: Dict[str, int], text_dim: int, feat_dim: int, cfg: PurificationConfig):
        super().__init__()
        self.cfg = cfg
        self.text_dim = text_dim
        self.scorers = nn.ModuleDict({k: nn.Linear(d, text_dim, bias=False) for k, d in dims.items()})
        self.anchor_proj = nn.Linear(text_dim, feat_dim, bias=False)          # G
        self._scorers_frozen = False

    # ------------------------------------------------------------------ #
    # parameter groups
    # ------------------------------------------------------------------ #
    def scorer_parameters(self):
        return list(self.scorers.parameters())

    def anchor_parameters(self):
        return list(self.anchor_proj.parameters())

    def freeze_scorers(self):
        for p in self.scorers.parameters():
            p.requires_grad_(False)
        self._scorers_frozen = True

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #
    def level_weights(self, key: str) -> Tuple[float, float]:
        return self.cfg.level_weights.get(key, (0.5, 0.5))

    @staticmethod
    def unit(x: torch.Tensor) -> torch.Tensor:
        """unit(v) with the deterministic fallback for exactly-zero vectors (Appendix S1)."""
        norm = x.norm(dim=-1, keepdim=True)
        safe = x / norm.clamp(min=1e-12)
        zero = norm.squeeze(-1) == 0
        if zero.any():
            e = torch.zeros_like(x)
            e[..., 0] = 1.0
            safe = torch.where(zero.unsqueeze(-1), e, safe)
        return safe

    def project(self, key: str, tokens: torch.Tensor) -> torch.Tensor:
        """v = unit(P_l z) for tokens (..., d_l)."""
        return self.unit(self.scorers[key](tokens))

    def margin(self, banks: BankTensors) -> torch.Tensor:
        """xi_c = xi_0 + kappa_o o_c; zero for classes without nuisance phrases."""
        kappa = 0.0 if self.cfg.fixed_margin else self.cfg.kappa_o
        xi = self.cfg.xi0 + kappa * banks.overlap
        if self.cfg.positive_only:
            return torch.zeros_like(xi)
        return torch.where(banks.has_nuisance, xi, torch.zeros_like(xi))

    def signed_evidence(self, key: str, v: torch.Tensor, banks: BankTensors) -> torch.Tensor:
        """d = sum_h w_h (u^{+,h} - u^{-,h}) for projected tokens v (N, K, M, d_t), Eq. (4)."""
        w_loc, w_glob = self.level_weights(key)
        tau = self.cfg.tau_b
        u_pos = w_loc * size_normalised_lse(v, banks.pos_local, banks.pos_local_mask, tau) \
            + w_glob * size_normalised_lse(v, banks.pos_global, banks.pos_global_mask, tau)
        if self.cfg.positive_only:
            return u_pos
        u_neg = w_loc * size_normalised_lse(v, banks.neg_local, banks.neg_local_mask, tau) \
            + w_glob * size_normalised_lse(v, banks.neg_global, banks.neg_global_mask, tau)
        return u_pos - u_neg

    def allocate(self, d: torch.Tensor, xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sparse allocation a and retained mass rho, Eq. (6)-(7)."""
        M = d.shape[-1]
        if self.cfg.uniform_allocation:
            a = torch.full_like(d, 1.0 / M)
            return a, a.sum(-1)
        b = (d - xi[:, None, None]) / self.cfg.tau_s
        a = project_capped_simplex(b, force_unit_mass=self.cfg.no_neutral)
        return a, a.sum(-1)

    @staticmethod
    def deficit(d: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Candidate deficit D = (1/M) sum_i [xi - d_i]_+, Eq. (10); shape (N, K)."""
        return (xi[:, None, None] - d).clamp(min=0.0).mean(dim=-1)

    def evidence(self, key: str, tokens: torch.Tensor, banks: BankTensors):
        """tokens (N, K, M, d_l) -> (d, xi). Scores use image positions only."""
        v = self.project(key, tokens)
        return self.signed_evidence(key, v, banks), self.margin(banks)

    def nuisance_cost(self, key: str, tokens: torch.Tensor, banks: BankTensors) -> torch.Tensor:
        """Per-layer term of C(a): mean over classes and supports of the post-action deficit, Eq. (13)."""
        d, xi = self.evidence(key, tokens, banks)
        return self.deficit(d, xi).mean()

    # ------------------------------------------------------------------ #
    # anchor and prototypes
    # ------------------------------------------------------------------ #
    def anchor(self, banks: BankTensors) -> torch.Tensor:
        """r_c = unit(G sum_h w_{f,h} mean_r e_c^{+,h,r}), Eq. (S1)."""
        w_loc, w_glob = self.level_weights("final")
        loc = (banks.pos_local * banks.pos_local_mask[..., None]).sum(1) / banks.pos_local_mask.sum(1, keepdim=True).clamp(min=1)
        glob = (banks.pos_global * banks.pos_global_mask[..., None]).sum(1) / banks.pos_global_mask.sum(1, keepdim=True).clamp(min=1)
        return self.unit(self.anchor_proj(w_loc * loc + w_glob * glob))

    def purify(self, final_tokens: torch.Tensor, banks: BankTensors) -> PurificationOutput:
        """
        final_tokens: (N, K, M, d_f) final-layer image tokens of the supports.
        Scoring uses the separately projected copies; prototypes use the
        original (unit-normalised) visual tokens.
        """
        N, K, M, d_f = final_tokens.shape
        z = self.unit(final_tokens)
        d, xi = self.evidence("final", final_tokens, banks)
        a, rho = self.allocate(d, xi)
        D = self.deficit(d, xi)

        if self.cfg.no_intrinsic_fallback:
            r = self.unit(z.mean(dim=(1, 2)))                       # ablation: mean visual evidence
        else:
            r = self.anchor(banks)
        lam = self.cfg.lam
        visual = torch.einsum("nkm,nkmd->nkd", a, z)
        w = lam * visual + (1.0 - lam * rho)[..., None] * r[:, None, :]    # Eq. (8)
        w_bar = w.mean(dim=1)
        norm = w_bar.norm(dim=-1)
        fallback = norm <= self.cfg.eps_n
        p = torch.where(fallback[:, None], r, w_bar / norm.clamp(min=1e-12)[:, None])  # Eq. (9)
        return PurificationOutput(p, a, d, xi, rho, D, fallback, r)

    def uniform_prototypes(self, final_tokens: torch.Tensor) -> torch.Tensor:
        """Ablation without purification: average of pooled support features."""
        z = final_tokens.mean(dim=2)            # (N, K, d_f)
        return F.normalize(z.mean(dim=1), dim=-1)

    # ------------------------------------------------------------------ #
    # calibration of the scoring maps (warm-up), Eq. (S4)
    # ------------------------------------------------------------------ #
    def alignment_loss(self, key: str, tokens: torch.Tensor, banks: BankTensors) -> torch.Tensor:
        """Cross-entropy between pooled projected supports and all episode intrinsic embeddings."""
        N, K = tokens.shape[:2]
        pooled = self.unit(self.scorers[key](tokens.mean(dim=2)))                  # (N, K, d_t)
        e_bar = self.unit(banks.pos_summary)                                        # (N, d_t)
        logits = torch.einsum("nkd,cd->nkc", pooled, e_bar) / self.cfg.tau_a
        target = torch.arange(N, device=tokens.device).repeat_interleave(K)
        return F.cross_entropy(logits.reshape(N * K, N), target)
