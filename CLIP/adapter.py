import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

# LoRA Adapter for CLIP
class LoRAAdapter(nn.Module):
    def __init__(self, input_dim, rank=16, alpha=16, dropout=0.1):
        super(LoRAAdapter, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        
        # Low-rank matrices A and B
        self.lora_A = nn.Linear(input_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, input_dim, bias=False)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        # Apply dropout to input
        x_dropped = self.dropout(x)
        
        # LoRA computation: x + (B @ A @ x) * scaling
        lora_out = self.lora_B(self.lora_A(x_dropped)) * self.scaling
        
        return x + lora_out

# Enhanced LoRA Adapter with intermediate features
class EnhancedLoRAAdapter(nn.Module):
    def __init__(self, input_dim, rank=16, alpha=16, dropout=0.1, return_intermediate=True):
        super(EnhancedLoRAAdapter, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.return_intermediate = return_intermediate
        
        # Low-rank matrices A and B
        self.lora_A = nn.Linear(input_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, input_dim, bias=False)
        
        # Additional intermediate processing (similar to bottleneck in original)
        self.intermediate_norm = nn.LayerNorm(rank)
        self.intermediate_activation = nn.GELU()
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        # Apply dropout to input
        x_dropped = self.dropout(x)
        
        # LoRA computation with intermediate features
        intermediate = self.lora_A(x_dropped)
        intermediate = self.intermediate_norm(intermediate)
        intermediate = self.intermediate_activation(intermediate)
        
        lora_out = self.lora_B(intermediate) * self.scaling
        adapted_x = x + lora_out
        
        if self.return_intermediate:
            return adapted_x, intermediate
        else:
            return adapted_x

# Multi-Scale LoRA Adapter
class MultiScaleLoRAAdapter(nn.Module):
    def __init__(self, input_dim, ranks=[8, 16, 32], alpha=16, dropout=0.1):
        super(MultiScaleLoRAAdapter, self).__init__()
        self.scales = nn.ModuleList([
            LoRAAdapter(input_dim, rank=rank, alpha=alpha, dropout=dropout)
            for rank in ranks
        ])
        self.fusion_weight = nn.Parameter(torch.ones(len(ranks)) / len(ranks))
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x):
        scale_outputs = []
        for scale_adapter in self.scales:
            scale_outputs.append(scale_adapter(x))
        
        # Weighted fusion of multi-scale outputs
        weights = self.softmax(self.fusion_weight)
        fused_output = sum(w * out for w, out in zip(weights, scale_outputs))
        
        return fused_output

class CLIP_LoRA_Implanted(nn.Module):
    def __init__(self, clip_model, features, adapter_config=None):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = features
        
        # Default adapter configuration
        if adapter_config is None:
            adapter_config = {
                'rank': 16,
                'alpha': 16,
                'dropout': 0.1,
                'use_multi_scale': False,
                'multi_scale_ranks': [8, 16, 32]
            }
        
        self.adapter_config = adapter_config
        
        # Initialize LoRA adapters for segmentation and detection tasks
        if adapter_config.get('use_multi_scale', False):
            self.seg_adapters = nn.ModuleList([
                MultiScaleLoRAAdapter(
                    1024, 
                    ranks=adapter_config['multi_scale_ranks'],
                    alpha=adapter_config['alpha'],
                    dropout=adapter_config['dropout']
                ) for _ in range(len(features))
            ])
            self.det_adapters = nn.ModuleList([
                MultiScaleLoRAAdapter(
                    1024,
                    ranks=adapter_config['multi_scale_ranks'],
                    alpha=adapter_config['alpha'],
                    dropout=adapter_config['dropout']
                ) for _ in range(len(features))
            ])
        else:
            self.seg_adapters = nn.ModuleList([
                EnhancedLoRAAdapter(
                    1024,
                    rank=adapter_config['rank'],
                    alpha=adapter_config['alpha'],
                    dropout=adapter_config['dropout']
                ) for _ in range(len(features))
            ])
            self.det_adapters = nn.ModuleList([
                EnhancedLoRAAdapter(
                    1024,
                    rank=adapter_config['rank'],
                    alpha=adapter_config['alpha'],
                    dropout=adapter_config['dropout']
                ) for _ in range(len(features))
            ])
        
        # Learnable mixing weights (replacing fixed 0.8, 0.1, 0.1)
        self.mixing_weights = nn.Parameter(torch.tensor([0.8, 0.1, 0.1]))
        self.weight_softmax = nn.Softmax(dim=0)
        
        # Optional: Cross-attention between seg and det features
        self.use_cross_attention = adapter_config.get('use_cross_attention', False)
        if self.use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(1024, num_heads=8, dropout=0.1)
    
    def forward(self, x):
        # Initial CLIP processing
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1) 
        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, dim)
        
        attn_out = []
        seg_patch_tokens = []
        det_patch_tokens = []
        
        # Get learnable mixing weights
        mix_weights = self.weight_softmax(self.mixing_weights)
        
        for i in range(24):  # Assuming ViT-Large with 24 layers
            if i + 1 == 12:
                x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
                attn_out.append(attn)
            else:
                x, attn_map = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            
            # Apply LoRA adapters at specified layers
            if (i + 1) in self.features:
                adapter_idx = self.features.index(i + 1)
                
                if self.adapter_config.get('use_multi_scale', False):
                    seg_adapt_out = self.seg_adapters[adapter_idx](x)
                    det_adapt_out = self.det_adapters[adapter_idx](x)
                    # For multi-scale, we don't have intermediate features
                    seg_patch_tokens.append(seg_adapt_out)
                    det_patch_tokens.append(det_adapt_out)
                else:
                    seg_adapt_out, seg_adapt_med = self.seg_adapters[adapter_idx](x)
                    det_adapt_out, det_adapt_med = self.det_adapters[adapter_idx](x)
                    seg_patch_tokens.append(seg_adapt_med)
                    det_patch_tokens.append(det_adapt_med)
                
                # Optional cross-attention between seg and det features
                if self.use_cross_attention:
                    seg_enhanced, _ = self.cross_attention(seg_adapt_out, det_adapt_out, det_adapt_out)
                    det_enhanced, _ = self.cross_attention(det_adapt_out, seg_adapt_out, seg_adapt_out)
                    seg_adapt_out = seg_enhanced
                    det_adapt_out = det_enhanced
                
                # Learnable weighted combination
                x = mix_weights[0] * x + mix_weights[1] * seg_adapt_out + mix_weights[2] * det_adapt_out
        
        # Attention map processing
        if attn_out:
            B, C, L = attn_out[0].shape
            H = int(math.sqrt(L-1))
            out_attn = torch.zeros([H, H]).to(x.device)
            for attn_layer in attn_out:
                out_attn = out_attn + attn_layer[0, 0, 1:].view(H, H)
        
        # Final processing
        x = x.permute(1, 0, 2)  # (batch, seq_len, dim)
        seg_patch_tokens = [token.permute(1, 0, 2) for token in seg_patch_tokens]
        det_patch_tokens = [token.permute(1, 0, 2) for token in det_patch_tokens]
        
        pooled, tokens = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(pooled)
        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj
        
        return pooled, seg_patch_tokens, det_patch_tokens

# Utility function to count parameters
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (LoRAAdapter, EnhancedLoRAAdapter, MultiScaleLoRAAdapter)):
            lora_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Parameter efficiency: {lora_params/total_params*100:.2f}%")

# Example usage
def create_lora_mvfa_model(clip_model, features=[6, 12, 18, 24]):
    """
    Create MVFA model with LoRA adapters
    
    Args:
        clip_model: Pre-trained CLIP model
        features: List of layer indices where adapters should be applied
    
    Returns:
        CLIP_LoRA_Implanted model
    """
    adapter_config = {
        'rank': 16,                    # LoRA rank (lower = more efficient)
        'alpha': 16,                   # LoRA scaling factor
        'dropout': 0.1,                # Dropout rate
        'use_multi_scale': False,      # Whether to use multi-scale adapters
        'multi_scale_ranks': [8, 16, 32],  # Ranks for multi-scale
        'use_cross_attention': True    # Cross-attention between seg and det
    }
    
    model = CLIP_LoRA_Implanted(clip_model, features, adapter_config)
    
    # Freeze CLIP parameters, only train LoRA adapters
    for name, param in model.named_parameters():
        if 'lora' not in name.lower() and 'mixing_weights' not in name and 'cross_attention' not in name:
            param.requires_grad = False
    
    count_parameters(model)
    return model

# Example with different configurations
def create_efficient_lora_model(clip_model, features=[6, 12, 18, 24]):
    """More parameter-efficient configuration"""
    adapter_config = {
        'rank': 8,                     # Lower rank for efficiency
        'alpha': 8,
        'dropout': 0.1,
        'use_multi_scale': False,
        'use_cross_attention': False,
        'bottleneck_dim': 768          # Match text_features dimension
    }
    return CLIP_LoRA_Implanted(clip_model, features, adapter_config)

def create_multiscale_lora_model(clip_model, features=[6, 12, 18, 24]):
    """Multi-scale LoRA configuration for better performance"""
    adapter_config = {
        'rank': 16,
        'alpha': 16,
        'dropout': 0.1,
        'use_multi_scale': True,
        'multi_scale_ranks': [4, 8, 16, 32],
        'use_cross_attention': True,
        'bottleneck_dim': 768          # Match text_features dimension
    }
    return CLIP_LoRA_Implanted(clip_model, features, adapter_config)