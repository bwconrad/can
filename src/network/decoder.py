from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from einops import repeat
from timm.models.vision_transformer import Block

from src.network.pos_embed import get_2d_sincos_pos_embed


class VitDecoder(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        num_patches: int = 196,
        in_channels: int = 3,
        depth: int = 8,
        embed_dim: int = 512,
        in_dim: int = 768,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),  # type:ignore
        act_layer: nn.Module = nn.GELU,  # type:ignore
    ):
        super().__init__()

        # Projection from encoder to decoder dim
        self.embed = nn.Linear(in_dim, embed_dim, bias=True)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, in_dim))

        # Sin-cos position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros((1, num_patches + 1, embed_dim)), requires_grad=False
        )

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,  # type:ignore
                    act_layer=act_layer,  # type:ignore
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, patch_size**2 * in_channels, bias=True)

        self.init_weights(num_patches)

    def init_weights(self, num_patches: int):
        # Initialize to sin-cos position embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # All other weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        idx_unshuffle: torch.Tensor,
        p: Optional[torch.Tensor] = None,
    ):
        # Append mask tokens to input
        L = idx_unshuffle.shape[1]
        B, L_unmasked, D = x.shape
        mask_tokens = self.mask_token.repeat(B, L + 1 - L_unmasked, 1)
        temp = torch.concat([x[:, 1:, :], mask_tokens], dim=1)  # Skip cls token

        # Unshuffle tokens
        temp = torch.gather(
            temp, dim=1, index=repeat(idx_unshuffle, "b l -> b l d", d=D)
        )

        # Add noise level embedding
        if p is not None:
            temp = temp + p[:, None, :]

        # Prepend cls token
        x = torch.cat([x[:, :1, :], temp], dim=1)

        # Project to decoder embed size
        x = self.embed(x)

        # Add pos embed
        x = x + self.pos_embed

        # Apply transformer layers
        x = self.blocks(x)

        # Predict pixel values
        x = self.head(self.norm(x))

        return x[:, 1:, :]  # Don't return cls token


def dec512d8b(patch_size: int, num_patches: int, in_dim: int, **kwargs):
    return VitDecoder(
        patch_size=patch_size,
        num_patches=num_patches,
        in_dim=in_dim,
        embed_dim=512,
        depth=8,
        num_heads=16,
        **kwargs,
    )


MODEL_DICT = {"dec512d8b": dec512d8b}


def build_decoder(model: str, **kwargs):
    try:
        model_fn = MODEL_DICT[model]
    except:
        raise ValueError(
            f"{model} is not an available decoder. Should be one of {[k for k in MODEL_DICT.keys()]}"
        )

    return model_fn(**kwargs)
