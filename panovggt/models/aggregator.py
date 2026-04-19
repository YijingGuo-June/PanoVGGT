# aggregator.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0

from functools import partial
import logging
import os
import math
from typing import List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import requests

from panovggt.dinov2.layers import Mlp, PatchEmbed
from panovggt.layers.pos_embed import RoPE2D, PositionGetter
from panovggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from panovggt.layers.block import BlockRope
from panovggt.layers.attention import FlashAttentionRope

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    Aggregator decoder module:
      1. RoPE (Rotary Position Encoding) for relative positional relationships.
      2. Layer-wise additive absolute spherical position encoding for geometric priors.
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 5,
        rope_freq: int = 100,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        qk_norm: bool = True,
        init_values: float = 0.01,
        num_dec_blk_not_to_checkpoint: int = 4,
        use_checkpoint: bool = True,
        patch_embed: str = "dinov2_vitl14_reg",
        use_pano_pos: bool = True,
        pos_mlp_hidden: int = 1024,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.dec_embed_dim = embed_dim
        self.depth = depth
        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = False

        # 1) DINO patch embedding
        self._build_patch_embed(
            patch_embed=patch_embed,
            img_size=img_size,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            embed_dim=embed_dim,
        )

        # 2) RoPE (relative position encoding within attention)
        self.rope = RoPE2D(freq=float(rope_freq)) if rope_freq and rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        # 3) Absolute spherical position encoding (additive)
        self.use_pano_pos = use_pano_pos
        if self.use_pano_pos:
            self.pano_pos_mlp = nn.Sequential(
                nn.Linear(4, pos_mlp_hidden),
                nn.GELU(),
                nn.Linear(pos_mlp_hidden, self.dec_embed_dim),
            )
            self.alpha_pos = nn.Parameter(torch.tensor(0.0))

        # 4) Decoder blocks
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=self.dec_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=init_values,
                qk_norm=qk_norm,
                attn_class=FlashAttentionRope,
                rope=self.rope,
            )
            for _ in range(depth)
        ])

        # 5) Register tokens and normalization buffers
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(
            torch.randn(1, 1, num_register_tokens, self.dec_embed_dim)
        )
        nn.init.normal_(self.register_token, std=1e-6)

        self.register_buffer(
            "_resnet_mean",
            torch.tensor(_RESNET_MEAN, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_resnet_std",
            torch.tensor(_RESNET_STD, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    # ------------------------------------------------------------------ #
    #                        Patch Embedding                               #
    # ------------------------------------------------------------------ #
    def _build_patch_embed(
        self,
        patch_embed: str,
        img_size: int,
        patch_size: int,
        num_register_tokens: int,
        embed_dim: int,
        interpolate_antialias: bool = True,
        interpolate_offset: float = 0.0,
        block_chunks: int = 0,
        init_values: float = 1.0,
    ):
        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size,
                in_chans=3, embed_dim=embed_dim,
            )
            self.patch_embed_dim = embed_dim
            self.needs_projection = False
            return

        vit_registry = {
            "dinov2_vitl14_reg": (vit_large, 1024, "dinov2_vitl14"),
            "dinov2_vitb14_reg": (vit_base, 768, "dinov2_vitb14"),
            "dinov2_vits14_reg": (vit_small, 384, "dinov2_vits14"),
            "dinov2_vitg2_reg": (vit_giant2, 1536, "dinov2_vitg14"),
        }
        vit_url_map = {
            "dinov2_vitl14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
            "dinov2_vitb14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
            "dinov2_vits14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
            "dinov2_vitg2_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
        }

        vit_fn, vit_dim, hub_name = vit_registry[patch_embed]
        self.patch_embed = vit_fn(
            img_size=518, patch_size=patch_size,
            num_register_tokens=4,
            interpolate_antialias=interpolate_antialias,
            interpolate_offset=interpolate_offset,
            block_chunks=block_chunks,
            init_values=init_values,
        )

        # Attempt to load DINOv2 pretrained weights
        self._try_load_dinov2(hub_name, vit_url_map.get(patch_embed), patch_embed)

        self.patch_embed_dim = vit_dim
        self.needs_projection = vit_dim != self.dec_embed_dim
        if self.needs_projection:
            self.patch_embed_projection = nn.Linear(vit_dim, self.dec_embed_dim)

        if hasattr(self.patch_embed, "mask_token"):
            delattr(self.patch_embed, "mask_token")

    def _try_load_dinov2(self, hub_name: str, url: Optional[str], patch_embed_key: str):
        """Try loading DINOv2 weights via torch.hub, then fallback to direct download."""
        success = False
        model_dict = self.patch_embed.state_dict()

        # Method 1: torch.hub
        try:
            logger.info(f"Loading DINOv2 weights for {hub_name} via torch.hub")
            pretrained = torch.hub.load("facebookresearch/dinov2", hub_name)
            matched = {
                k: v for k, v in pretrained.state_dict().items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            logger.info(f"Matched {len(matched)}/{len(model_dict)} layers from torch.hub")
            model_dict.update(matched)
            self.patch_embed.load_state_dict(model_dict)
            success = True
        except Exception as e:
            logger.warning(f"torch.hub load failed: {e}")

        # Method 2: Direct download
        if not success and url:
            try:
                logger.info(f"Downloading DINOv2 weights from {url}")
                weights_dir = Path(os.path.expanduser("~/.cache/panovggt/weights"))
                weights_dir.mkdir(parents=True, exist_ok=True)
                local_path = weights_dir / f"{patch_embed_key}_pretrain.pth"
                if not local_path.exists():
                    r = requests.get(url, allow_redirects=True)
                    with open(local_path, "wb") as f:
                        f.write(r.content)
                state = torch.load(local_path, map_location="cpu")
                if "teacher" in state:
                    state = state["teacher"]
                matched = {
                    k: v for k, v in state.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                model_dict.update(matched)
                self.patch_embed.load_state_dict(model_dict)
                success = True
            except Exception as e:
                logger.warning(f"Direct download failed: {e}")

        if success:
            for p in self.patch_embed.parameters():
                p.requires_grad = True
            logger.info("DINOv2 weights loaded; parameters set to trainable")
        else:
            logger.warning("Could not load DINOv2 pretrained weights; using random init")

    # ------------------------------------------------------------------ #
    #                           Decode                                     #
    # ------------------------------------------------------------------ #
    def _decode(self, hidden: torch.Tensor, B: int, S: int, H: int, W: int):
        BN, hw, C = hidden.shape
        assert BN == B * S
        Hp, Wp = H // self.patch_size, W // self.patch_size
        assert hw == Hp * Wp

        # Prepend register tokens
        reg = self.register_token.repeat(B, S, 1, 1).reshape(B * S, self.patch_start_idx, C)
        hidden = torch.cat([reg, hidden], dim=1)  # (B*S, P, C)
        P = hidden.shape[1]

        # --- RoPE position indices ---
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, Hp, Wp, device=hidden.device)
            pos = pos + 1
            pos_special = torch.zeros(
                B * S, self.patch_start_idx, 2,
                device=hidden.device, dtype=pos.dtype,
            )
            pos = torch.cat([pos_special, pos], dim=1)

        # --- Absolute spherical position encoding ---
        pos_embed_single_bs = None
        pos_embed_multi_bsp = None
        if self.use_pano_pos:
            device = hidden.device
            ys = torch.arange(Hp, device=device, dtype=torch.float32) + 0.5
            xs = torch.arange(Wp, device=device, dtype=torch.float32) + 0.5
            theta = (ys[:, None] / Hp - 0.5) * math.pi
            phi = (xs[None, :] / Wp - 0.5) * (2 * math.pi)
            theta = theta.expand(Hp, Wp)
            phi = phi.expand(Hp, Wp)

            pos_feats = torch.stack(
                [torch.sin(theta), torch.cos(theta), torch.sin(phi), torch.cos(phi)],
                dim=-1,
            ).reshape(Hp * Wp, 4)

            pos_embed_patch = self.pano_pos_mlp(pos_feats)

            # Register tokens get zero positional encoding
            zeros_reg = torch.zeros(self.patch_start_idx, C, device=device, dtype=hidden.dtype)
            pos_embed_single = torch.cat([zeros_reg, pos_embed_patch], dim=0)  # (P, C)

            pos_embed_single_bs = pos_embed_single.unsqueeze(0).expand(B * S, -1, -1)
            pos_embed_multi_bsp = (
                pos_embed_single.unsqueeze(0).unsqueeze(0)
                .expand(B, S, P, C)
                .reshape(B, S * P, C)
            )

        # --- Decoder loop ---
        last_two = []
        for i, blk in enumerate(self.decoder):
            if i % 2 == 0:
                # Single-frame branch
                h_in = hidden.reshape(B * S, P, C)
                if self.use_pano_pos:
                    h_in = h_in + self.alpha_pos * pos_embed_single_bs
                p_in = None if pos is None else pos.reshape(B * S, P, -1)
            else:
                # Multi-frame branch
                h_in = hidden.reshape(B, S * P, C)
                if self.use_pano_pos:
                    h_in = h_in + self.alpha_pos * pos_embed_multi_bsp
                p_in = None if pos is None else pos.reshape(B, S * P, -1)

            if self.training and self.use_checkpoint and i >= self.num_dec_blk_not_to_checkpoint:
                h_out = checkpoint(blk, h_in, p_in, use_reentrant=False)
            else:
                h_out = blk(h_in, xpos=p_in)

            hidden = h_out.reshape(B * S, P, C)

            if i + 1 in [self.depth - 1, self.depth]:
                last_two.append(hidden)

        last_two_cat = torch.cat(last_two, dim=-1) if len(last_two) == 2 else last_two[-1]

        pos_2d = None if pos is None else pos.reshape(B * S, P, -1)
        return last_two_cat, pos_2d

    # ------------------------------------------------------------------ #
    #                           Forward                                    #
    # ------------------------------------------------------------------ #
    def forward(
        self, images: torch.Tensor
    ) -> Tuple[List[torch.Tensor], int, Optional[torch.Tensor]]:
        """
        Args:
            images: (B, S, 3, H, W) input panoramic images.

        Returns:
            output_list: [(B, S, P, 2*C)] aggregated features.
            patch_start_idx: number of register tokens prepended.
            pos_2d: optional RoPE position embeddings.
        """
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize
        images = (images - self._resnet_mean) / self._resnet_std

        # Patch embed
        x = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(x)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        if getattr(self, "needs_projection", False):
            patch_tokens = self.patch_embed_projection(patch_tokens)

        # Decode
        hidden_cat, pos_2d = self._decode(patch_tokens, B, S, H, W)

        P = hidden_cat.shape[1]
        C2 = hidden_cat.shape[-1]
        output = hidden_cat.view(B, S, P, C2)
        return [output], self.patch_start_idx, pos_2d