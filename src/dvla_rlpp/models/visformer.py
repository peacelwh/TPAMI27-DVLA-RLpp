"""Visformer backbone (Chen et al., ICCV 2021) with stage-wise access.

Parameter names are identical to the DVLA-RL / VT-FSL release so that the
pre-trained ``visformer-<dataset>.pth`` checkpoints load without renaming.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .weight_init import to_2tuple, trunc_normal_

__all__ = ["Visformer", "visformer_tiny"]


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.LayerNorm):
    """LayerNorm for BCHW tensors."""

    def __init__(self, num_channels):
        super().__init__([num_channels, 1, 1])

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class BatchNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim, eps=1e-5, momentum=0.1, track_running_stats=True)

    def forward(self, x):
        return self.bn(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 group=8, spatial_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.spatial_conv = spatial_conv
        if self.spatial_conv:
            hidden_features = in_features * 5 // 6 if group < 2 else in_features * 2
        self.drop = nn.Dropout(drop)
        self.conv1 = nn.Conv2d(in_features, hidden_features, 1, bias=False)
        self.act1 = act_layer()
        if self.spatial_conv:
            self.conv2 = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=group, bias=False)
            self.act2 = act_layer()
        self.conv3 = nn.Conv2d(hidden_features, out_features, 1, bias=False)

    def forward(self, x):
        x = self.drop(self.act1(self.conv1(x)))
        if self.spatial_conv:
            x = self.act2(self.conv2(x))
        return self.drop(self.conv3(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim_ratio=1., qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = round(dim // num_heads * head_dim_ratio)
        self.head_dim = head_dim
        self.scale = head_dim ** (qk_scale if qk_scale is not None else -0.25)
        self.qkv = nn.Conv2d(dim, head_dim * num_heads * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(head_dim * num_heads, dim, 1, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = rearrange(self.qkv(x), "b (x y z) h w -> x b y (h w) z", x=3, y=self.num_heads, z=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q * self.scale) @ (k.transpose(-2, -1) * self.scale)).softmax(dim=-1)
        x = self.attn_drop(attn) @ v
        x = rearrange(x, "b y (h w) z -> b (y z) h w", h=H, w=W)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, head_dim_ratio=1., mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm, group=8,
                 attn_disabled=False, spatial_conv=False):
        super().__init__()
        self.attn_disabled = attn_disabled
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if not attn_disabled:
            self.norm1 = norm_layer(dim)
            self.attn = Attention(dim, num_heads, head_dim_ratio, qkv_bias, qk_scale, attn_drop, drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop, group=group, spatial_conv=spatial_conv)

    def forward(self, x):
        if not self.attn_disabled:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size, patch_size = to_2tuple(img_size), to_2tuple(patch_size)
        self.img_size, self.patch_size = img_size, patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm_pe = norm_layer is not None
        if self.norm_pe:
            self.norm = norm_layer(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input size ({H}*{W}) does not match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        return self.norm(x) if self.norm_pe else x


class Visformer(nn.Module):
    """Three-stage Visformer. Stage dims are (embed_dim/2, embed_dim, 2*embed_dim)."""

    def __init__(self, img_size=224, init_channels=32, num_classes=64, embed_dim=384, depth=12, num_heads=6,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.5,
                 norm_layer=LayerNorm, attn_stage="111", pos_embed=True, spatial_conv="111", group=8, pool=True,
                 conv_init=False, embedding_norm=None, small_stem=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.init_channels = init_channels
        self.img_size = img_size
        self.pool = pool
        self.conv_init = conv_init
        self.stage_dims = (embed_dim // 2, embed_dim, embed_dim * 2)

        if isinstance(depth, (list, tuple)):
            self.stage_num1, self.stage_num2, self.stage_num3 = depth
            depth_total = sum(depth)
        else:
            self.stage_num1 = self.stage_num3 = depth // 3
            self.stage_num2 = depth - self.stage_num1 - self.stage_num3
            depth_total = depth
        self.pos_embed = pos_embed
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth_total)]

        # stem + stage 1
        if not small_stem:
            self.stem = nn.Sequential(nn.Conv2d(3, init_channels, 7, stride=2, padding=3, bias=False),
                                      BatchNorm(init_channels), nn.ReLU(inplace=True))
            img_size //= 2
        else:
            self.stem = nn.Sequential(nn.Conv2d(3, init_channels, 3, padding=1, bias=False), BatchNorm(init_channels),
                                      nn.ReLU(inplace=True), nn.Conv2d(init_channels, init_channels, 3, padding=1, bias=False),
                                      BatchNorm(init_channels), nn.ReLU(inplace=True))
        self.patch_embed1 = PatchEmbed(img_size, 4, init_channels, embed_dim // 2, embedding_norm)
        img_size //= 4
        if pos_embed:
            self.pos_embed1 = nn.Parameter(torch.zeros(1, embed_dim // 2, img_size, img_size))
            self.pos_drop = nn.Dropout(p=drop_rate)
        self.stage1 = nn.ModuleList([
            Block(embed_dim // 2, num_heads, 0.5, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[i],
                  norm_layer=norm_layer, group=group, attn_disabled=(attn_stage[0] == "0"),
                  spatial_conv=(spatial_conv[0] == "1")) for i in range(self.stage_num1)])

        # stage 2
        self.patch_embed2 = PatchEmbed(img_size, 2, embed_dim // 2, embed_dim, embedding_norm)
        img_size //= 2
        if pos_embed:
            self.pos_embed2 = nn.Parameter(torch.zeros(1, embed_dim, img_size, img_size))
        self.stage2 = nn.ModuleList([
            Block(embed_dim, num_heads, 1.0, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[i],
                  norm_layer=norm_layer, group=group, attn_disabled=(attn_stage[1] == "0"),
                  spatial_conv=(spatial_conv[1] == "1")) for i in range(self.stage_num1, self.stage_num1 + self.stage_num2)])

        # stage 3
        self.patch_embed3 = PatchEmbed(img_size, 2, embed_dim, embed_dim * 2, embedding_norm)
        img_size //= 2
        if pos_embed:
            self.pos_embed3 = nn.Parameter(torch.zeros(1, embed_dim * 2, img_size, img_size))
        self.stage3 = nn.ModuleList([
            Block(embed_dim * 2, num_heads, 1.0, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[i],
                  norm_layer=norm_layer, group=group, attn_disabled=(attn_stage[2] == "0"),
                  spatial_conv=(spatial_conv[2] == "1")) for i in range(self.stage_num1 + self.stage_num2, depth_total)])

        # head
        if pool:
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.norm = norm_layer(embed_dim * 2)
        self.head = nn.Linear(embed_dim * 2, num_classes)

        if pos_embed:
            trunc_normal_(self.pos_embed1, std=0.02)
            trunc_normal_(self.pos_embed2, std=0.02)
            trunc_normal_(self.pos_embed3, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            if self.conv_init:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            else:
                trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    # ---------------------------------------------------------------- #
    # Stage-wise interface used by DVLA-RL++
    # ---------------------------------------------------------------- #
    def forward_stage1(self, x):
        x = self.stem(x)
        x = self.patch_embed1(x)
        if self.pos_embed:
            x = self.pos_drop(x + self.pos_embed1)
        for b in self.stage1:
            x = b(x)
        return x

    def embed_stage2(self, x):
        x = self.patch_embed2(x)
        if self.pos_embed:
            x = self.pos_drop(x + self.pos_embed2)
        return x

    def embed_stage3(self, x):
        x = self.patch_embed3(x)
        if self.pos_embed:
            x = self.pos_drop(x + self.pos_embed3)
        return x

    def blocks(self, stage: str):
        return {"stage2": self.stage2, "stage3": self.stage3}[stage]

    def final_tokens(self, x):
        """Normalised final feature map (B, C, H, W)."""
        return self.norm(x)

    def pooled(self, x):
        x = self.global_pooling(x) if self.pool else x[:, :, 0:1, 0:1]
        return x.flatten(1)

    def forward(self, x):
        x = self.forward_stage1(x)
        x = self.embed_stage2(x)
        for b in self.stage2:
            x = b(x)
        x = self.embed_stage3(x)
        for b in self.stage3:
            x = b(x)
        x = self.pooled(self.final_tokens(x))
        return self.head(x), x


def visformer_tiny(**kwargs):
    return Visformer(img_size=kwargs.pop("img_size", 224), init_channels=16, embed_dim=192, depth=[7, 4, 4],
                     num_heads=3, mlp_ratio=4., group=8, attn_stage="011", spatial_conv="100", norm_layer=BatchNorm,
                     conv_init=True, embedding_norm=BatchNorm, **kwargs)
