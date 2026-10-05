"""RL-gated attention block (Section 3.2, Fig. 3).

Given image tokens ``Z`` (B, M, d) and projected intrinsic semantic tokens
``T`` (P, d) shared by every image of the episode, the block computes

    V = Attn(T W_q, Z W_k^v, Z W_v^v)        image-guided context
    S = Attn(T W_q, T W_k^t, T W_v^t)        text-guided context
    B(a) = a V + (1 - a) S                   scalar gate mixing, Eq. (1)

and returns ``Z + mean(B)`` together with ``B`` so that the caller can append
the fused semantic tokens to the transformer block input and keep only the
image positions afterwards, Eq. (2).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mha(q, k, v, num_heads: int):
    """Scaled dot-product attention with (B, P, d) queries and (B, M, d) keys/values."""
    B, P, d = q.shape
    hd = d // num_heads
    q = q.view(B, P, num_heads, hd).transpose(1, 2)
    k = k.view(B, -1, num_heads, hd).transpose(1, 2)
    v = v.view(B, -1, num_heads, hd).transpose(1, 2)
    attn = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    out = attn.softmax(dim=-1) @ v
    return out.transpose(1, 2).reshape(B, P, d)


class RLGatedAttentionBlock(nn.Module):
    def __init__(self, dim: int, text_dim: int = 512, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0
        self.dim, self.num_heads = dim, num_heads
        self.sem_proj = nn.Linear(text_dim, dim, bias=False)     # A_l, Eq. (S2)
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k_img = nn.Linear(dim, dim, bias=False)
        self.w_v_img = nn.Linear(dim, dim, bias=False)
        self.w_k_txt = nn.Linear(dim, dim, bias=False)
        self.w_v_txt = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def semantic_tokens(self, pos_local: torch.Tensor, pos_global: torch.Tensor, weights) -> torch.Tensor:
        """T_l = [w_loc E_loc A ; w_glob E_glob A] shared by all images, Eq. (S2)."""
        w_loc, w_glob = weights
        t = torch.cat([w_loc * self.sem_proj(pos_local), w_glob * self.sem_proj(pos_global)], dim=0)
        return t

    def forward(self, z: torch.Tensor, tokens: torch.Tensor, gate: torch.Tensor):
        """
        z:      (B, M, dim) image tokens
        tokens: (P, dim) projected semantic tokens
        gate:   scalar tensor a_l in [0, 1] (one action per layer and episode)
        returns z_ctx (B, M, dim), fused (B, P, dim)
        """
        B = z.shape[0]
        t = tokens.unsqueeze(0).expand(B, -1, -1)
        q = self.w_q(t)
        v_img = _mha(q, self.w_k_img(z), self.w_v_img(z), self.num_heads)
        s_txt = _mha(q, self.w_k_txt(t), self.w_v_txt(t), self.num_heads)
        fused = self.out(gate * v_img + (1.0 - gate) * s_txt)
        z_ctx = z + fused.mean(dim=1, keepdim=True)
        return z_ctx, fused


# --------------------------------------------------------------------------- #
# BCHW <-> token helpers (semantic tokens are appended as extra rows of the map,
# following the inherited Visformer implementation, and stripped afterwards)
# --------------------------------------------------------------------------- #
def map_to_tokens(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    return x.flatten(2).transpose(1, 2)            # (B, H*W, C)


def tokens_to_map(z: torch.Tensor, H: int, W: int) -> torch.Tensor:
    B, M, C = z.shape
    return z.transpose(1, 2).reshape(B, C, H, W)


def append_semantic_rows(x: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
    """Append P semantic tokens as ceil(P / W) extra rows (zero padded)."""
    B, C, H, W = x.shape
    P = fused.shape[1]
    rows = math.ceil(P / W)
    pad = rows * W - P
    sem = fused
    if pad > 0:
        sem = torch.cat([sem, sem.new_zeros(B, pad, C)], dim=1)
    sem = sem.transpose(1, 2).reshape(B, C, rows, W)
    return torch.cat([x, sem], dim=2)


def strip_semantic_rows(x: torch.Tensor, H: int) -> torch.Tensor:
    return x[:, :, :H].contiguous()
