from functools import partial

import timm.models.vision_transformer as vision_transformer
import torch
import torch.nn as nn
from einops import repeat

from src.network.pos_embed import get_2d_sincos_pos_embed


class VisionTransformer(vision_transformer.VisionTransformer):
    """Vision transformer for masked image modeling.
    Uses fixed sin-cos position embeddings
    """

    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        assert self.num_prefix_tokens == 1  # Must have cls token

        # Re-initialize with fixed sin-cos position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(self.pos_embed.shape), requires_grad=False
        )
        self.init_pos_embed()

    def init_pos_embed(self):
        """Initialize sin-cos position embeddings"""
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        """Randomly mask mask_ratio patches of an image

        Args:
            x: Tensor of shape B x L x D
            mask_ratio: Ratio of patches to mask

        Return:
            x_masked: Tensor of non-masked patches
            mask: Tensor of size B x L where the positions of masked
                patches are marked by 1 and else 0
            idx_unshuffle: Tensor of size B x L with the sorting order
                to unshuffle patches back to the original order
        """
        B, L, D = x.shape

        # Number of patches to keep
        num_keep = int(L * (1 - mask_ratio))

        # Sort array of random noise
        noise = torch.rand((B, L), device=x.device)
        idx_shuffle = torch.argsort(noise, dim=1)
        idx_unshuffle = torch.argsort(idx_shuffle, dim=1)  # Undo shuffling

        # Keep indices of n_keep smallest values
        idx_keep = idx_shuffle[:, :num_keep]
        x_masked = torch.gather(x, dim=1, index=repeat(idx_keep, "b l -> b l d", d=D))

        # Generate binary mask
        mask = torch.ones((B, L), device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=idx_unshuffle)

        return x_masked, mask, idx_unshuffle

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.75):
        # Patch embed image
        x = self.patch_embed(x)

        # Add pos embed skipping cls token
        x = x + self.pos_embed[:, 1:, :]

        # Mask the image
        x, mask, idx_unshuffle = self.random_masking(x, mask_ratio)

        # Append the cls token
        cls_token = self.cls_token + self.pos_embed[:, 0, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        # Apply transformer layers
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)

        return x, mask, idx_unshuffle


def build_encoder(model: str, **kwargs):
    try:
        model_fn, patch_size = MODEL_DICT[model]
    except:
        raise ValueError(
            f"{model} is not an available encoder. Should be one of {[k for k in MODEL_DICT.keys()]}"
        )

    return model_fn(**kwargs), patch_size


def vit_tiny_patch16(**kwargs):
    return VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        weight_init="jax",
        **kwargs,
    )


def vit_small_patch16(**kwargs):
    return VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        weight_init="jax",
        **kwargs,
    )


def vit_base_patch16(**kwargs):
    return VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        weight_init="jax",
        **kwargs,
    )


def vit_large_patch16(**kwargs):
    return VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        weight_init="jax",
        **kwargs,
    )


def vit_huge_patch14(**kwargs):
    return VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        weight_init="jax",
        **kwargs,
    )


MODEL_DICT = {
    "vit_tiny_patch16": (vit_tiny_patch16, 16),
    "vit_small_patch16": (vit_small_patch16, 16),
    "vit_base_patch16": (vit_base_patch16, 16),
    "vit_large_patch16": (vit_large_patch16, 16),
    "vit_huge_patch14": (vit_huge_patch14, 14),
}


if __name__ == "__main__":
    from decoder import VitDecoder

    e = vit_base_patch16(img_size=192)
    d = VitDecoder()
    x = torch.rand(2, 3, 192, 192)
    y, mask, idx = e(x)
    print(y.size())
    y = d(y, idx)
    print(y.size())
