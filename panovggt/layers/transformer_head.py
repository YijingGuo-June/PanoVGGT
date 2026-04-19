"""
Transformer Decoder Modules

This module provides transformer-based decoder architectures for dense prediction tasks.
It includes standard self-attention decoders and cross-attention decoders with rotary 
position embeddings (RoPE) support.
"""

from functools import partial
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import FlashAttentionRope, FlashCrossAttentionRope
from .block import BlockRope, CrossBlockRope
from ..dinov2.layers import Mlp


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder with Self-Attention.
    
    A standard transformer decoder using self-attention blocks with optional
    rotary position embeddings (RoPE) and gradient checkpointing support.
    
    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension.
        dec_embed_dim (int): Decoder embedding dimension. Default: 512.
        depth (int): Number of transformer blocks. Default: 5.
        dec_num_heads (int): Number of attention heads. Default: 8.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim. Default: 4.
        rope: Rotary position embedding module. Default: None.
        need_project (bool): Whether to project input features. Default: True.
        use_checkpoint (bool): Whether to use gradient checkpointing. Default: False.
    
    Attributes:
        dec_embed_dim (int): Decoder embedding dimension.
        dec_num_heads (int): Number of attention heads.
        depth (int): Number of transformer blocks.
        mlp_ratio (float): MLP expansion ratio.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dec_embed_dim: int = 512,
        depth: int = 5,
        dec_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        rope: Optional[nn.Module] = None,
        need_project: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        # Expose configuration for external access
        self.dec_embed_dim = dec_embed_dim
        self.dec_num_heads = dec_num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        
        # Input projection
        self.projects = (
            nn.Linear(in_dim, dec_embed_dim) if need_project 
            else nn.Identity()
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                attn_class=FlashAttentionRope,
                rope=rope,
            )
            for _ in range(depth)
        ])
        
        # Output projection
        self.linear_out = nn.Linear(dec_embed_dim, out_dim)
    
    def forward(
        self,
        hidden: torch.Tensor,
        xpos: Optional[torch.Tensor] = None,
        pos_embed: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the transformer decoder.
        
        Args:
            hidden (torch.Tensor): Input features of shape [B*S, P, in_dim].
            xpos (torch.Tensor, optional): Position features (backward compatibility).
            pos_embed (torch.Tensor, optional): Position embeddings of shape [B*S, P, dec_embed_dim].
            attn_bias (torch.Tensor, optional): Attention bias of shape [H, P, P] or broadcastable.
        
        Returns:
            torch.Tensor: Output features of shape [B*S, P, out_dim].
        """
        # Project input features
        hidden = self.projects(hidden)  # [B*S, P, dec_embed_dim]
        
        # Add position embeddings if provided
        if pos_embed is not None:
            hidden = hidden + pos_embed
        
        # Apply transformer blocks
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                # Use gradient checkpointing during training
                try:
                    hidden = checkpoint(
                        blk, hidden, xpos, attn_bias, use_reentrant=False
                    )
                except TypeError:
                    # Fallback for blocks that don't support attn_bias
                    hidden = checkpoint(blk, hidden, xpos, use_reentrant=False)
            else:
                try:
                    hidden = blk(hidden, xpos=xpos, attn_bias=attn_bias)
                except TypeError:
                    # Fallback for blocks that don't support attn_bias
                    hidden = blk(hidden, xpos=xpos)
        
        # Project to output dimension
        out = self.linear_out(hidden)
        return out


class ContextTransformerDecoder(nn.Module):
    """
    Context Transformer Decoder with Cross-Attention.
    
    A transformer decoder that uses cross-attention to incorporate context information
    from a separate context stream. Supports rotary position embeddings (RoPE).
    
    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension.
        dec_embed_dim (int): Decoder embedding dimension. Default: 512.
        depth (int): Number of transformer blocks. Default: 5.
        dec_num_heads (int): Number of attention heads. Default: 8.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim. Default: 4.
        rope: Rotary position embedding module. Default: None.
    
    Attributes:
        dec_embed_dim (int): Decoder embedding dimension.
        dec_num_heads (int): Number of attention heads.
        depth (int): Number of transformer blocks.
        mlp_ratio (float): MLP expansion ratio.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dec_embed_dim: int = 512,
        depth: int = 5,
        dec_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        # Expose configuration for external access
        self.dec_embed_dim = dec_embed_dim
        self.dec_num_heads = dec_num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        
        # Input projections for query and context
        self.projects_x = nn.Linear(in_dim, dec_embed_dim)
        self.projects_y = nn.Linear(in_dim, dec_embed_dim)
        
        # Cross-attention transformer blocks
        self.blocks = nn.ModuleList([
            CrossBlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                attn_class=FlashAttentionRope,
                cross_attn_class=FlashCrossAttentionRope,
                rope=rope,
            )
            for _ in range(depth)
        ])
        
        # Output projection
        self.linear_out = nn.Linear(dec_embed_dim, out_dim)
    
    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        xpos: Optional[torch.Tensor] = None,
        ypos: Optional[torch.Tensor] = None,
        pos_embed: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the context transformer decoder.
        
        Args:
            hidden (torch.Tensor): Query features of shape [B*S, P, in_dim].
            context (torch.Tensor): Context features of shape [B*S, P, in_dim].
            xpos (torch.Tensor, optional): Position features for queries (backward compatibility).
            ypos (torch.Tensor, optional): Position features for context (backward compatibility).
            pos_embed (torch.Tensor, optional): Position embeddings of shape [B*S, P, dec_embed_dim].
                Applied to both hidden and context.
            attn_bias (torch.Tensor, optional): Attention bias for self-attention of shape [H, P, P]
                or broadcastable. Ignored if block doesn't support it.
        
        Returns:
            torch.Tensor: Output features of shape [B*S, P, out_dim].
        """
        # Project input features
        hidden = self.projects_x(hidden)    # [B*S, P, dec_embed_dim]
        context = self.projects_y(context)  # [B*S, P, dec_embed_dim]
        
        # Add position embeddings if provided
        if pos_embed is not None:
            hidden = hidden + pos_embed
            context = context + pos_embed
        
        # Apply cross-attention transformer blocks
        for blk in self.blocks:
            try:
                # Try to pass attn_bias
                hidden = blk(
                    hidden, context, xpos=xpos, ypos=ypos, attn_bias=attn_bias
                )
            except TypeError:
                # Fallback for blocks that don't support attn_bias
                hidden = blk(hidden, context, xpos=xpos, ypos=ypos)
        
        # Project to output dimension
        out = self.linear_out(hidden)
        return out


class LinearPts3d(nn.Module):
    """
    Linear Head for 3D Point Prediction.
    
    A simple linear projection head that outputs dense 3D points from patch tokens.
    Each token is projected to a patch_size × patch_size grid of 3D coordinates.
    Designed for dust3r-style dense 3D reconstruction.
    
    Args:
        patch_size (int): Size of each patch (e.g., 16 for 16×16 patches).
        dec_embed_dim (int): Input feature dimension from decoder.
        output_dim (int): Output dimension per point (e.g., 3 for XYZ). Default: 3.
    
    Attributes:
        patch_size (int): Patch size for upsampling.
    """
    
    def __init__(
        self,
        patch_size: int,
        dec_embed_dim: int,
        output_dim: int = 3
    ):
        super().__init__()
        self.patch_size = patch_size
        
        # Project each token to patch_size^2 points with output_dim coordinates
        self.proj = nn.Linear(
            dec_embed_dim,
            output_dim * self.patch_size ** 2
        )
    
    def forward(
        self,
        decout: List[torch.Tensor],
        img_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Forward pass to generate dense 3D points.
        
        Args:
            decout (List[torch.Tensor]): List of decoder outputs. Uses the last one.
            img_shape (Tuple[int, int]): Original image shape (H, W).
        
        Returns:
            torch.Tensor: Dense 3D points of shape [B, H, W, output_dim].
        """
        H, W = img_shape
        tokens = decout[-1]  # Use last decoder output
        B, S, D = tokens.shape
        
        # Project tokens to 3D points
        feat = self.proj(tokens)  # [B, S, output_dim * patch_size^2]
        
        # Reshape and upsample to image resolution
        feat = feat.transpose(-1, -2).view(
            B, -1, H // self.patch_size, W // self.patch_size
        )
        feat = F.pixel_shuffle(feat, self.patch_size)  # [B, output_dim, H, W]
        
        # Permute to [B, H, W, output_dim]
        return feat.permute(0, 2, 3, 1)

