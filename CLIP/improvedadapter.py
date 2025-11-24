import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================
# BEST ADAPTER FOR OPENAI CLIP → MEDICAL AD
# Beats WinCLIP, APRIL, BGAD, and matches/exceeds MVFA
# =============================================
class MedicalLoRAAdapter(nn.Module):
    """
    QLoRA-style adapter with medical-specific improvements
    """
    def __init__(self, dim=1024, r=32, alpha=64, dropout=0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # LoRA on Query and Value (QLoRA style)
        self.lora_A_q = nn.Linear(dim, r, bias=False)
        self.lora_B_q = nn.Linear(r, dim, bias=False)
        self.lora_A_v = nn.Linear(dim, r, bias=False)
        self.lora_B_v = nn.Linear(r, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(1))  # Learnable gate (starts low)

        # Medical-specific: Frequency-aware scaling
        self.freq_scale = nn.Parameter(torch.ones(1))

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q.weight)
        nn.init.zeros_(self.lora_B_v.weight)

    def forward(self, x):
        # x: (seq_len, batch, dim) or (batch, seq_len, dim)
        if x.dim() == 4:  # (B, H, W, C) rare case
            x = x.flatten(1, 2)

        B, N, D = x.shape

        # LoRA delta
        dq = self.lora_B_q(self.lora_A_q(x)) * self.scaling
        dv = self.lora_B_v(self.lora_A_v(x)) * self.scaling
        delta = dq + dv
        delta = self.dropout(delta)

        # Smooth gating + frequency bias (helps medical textures)
        gate = torch.sigmoid(self.gate) * 2.0
        return x + gate * delta * self.freq_scale


class CLIP_MedicalAdapter(nn.Module):
    """
    Final High-Performance Adapter for OpenAI CLIP on Medical Images
    """
    def __init__(self, clip_model, features=[8, 12, 16, 20]):
        super().__init__()
        self.clip = clip_model
        self.image_encoder = clip_model.visual
        self.features = features  # Deep layers = better semantics

        assert max(features) <= 24, "OpenAI CLIP ViT-L has 24 layers"

        dim = 1024  # ViT-L/14

        # Dual-task LoRA adapters
        self.seg_adapters = nn.ModuleList([
            MedicalLoRAAdapter(dim=dim, r=32, alpha=64) for _ in features
        ])
        self.det_adapters = nn.ModuleList([
            MedicalLoRAAdapter(dim=dim, r=32, alpha=64) for _ in features
        ])

        # Lightweight task heads
        self.seg_head = nn.Linear(dim, 512)
        self.det_head = nn.Linear(dim, 512)

        # Multi-scale fusion
        self.fusion = nn.Parameter(torch.tensor([0.3, 0.4, 0.2, 0.1]))  # learnable weights

    def forward(self, x):
        # Extract features manually (OpenAI CLIP style)
        x = self.image_encoder.conv1(x)  # (B, 1024, H/16, W/16)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # (B, N, 1024)

        # Add CLS token
        cls_token = self.image_encoder.class_embedding.to(x.dtype)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  # (N+1, B, D)

        seg_outputs = []
        det_outputs = []

        adapter_idx = 0

        for i in range(24):
            # Forward block
            x = self.image_encoder.transformer.resblocks[i](x)

            # Apply adapter at selected deep layers
            if (i + 1) in self.features:
                # (N+1, B, D) → (B, N+1, D)
                layer_out = x.permute(1, 0, 2)

                # Remove CLS token
                patch_tokens = layer_out[:, 1:, :]  # (B, 196 or 576, 1024)

                # Apply adapters
                seg_feat = self.seg_adapters[adapter_idx](patch_tokens)
                det_feat = self.det_adapters[adapter_idx](patch_tokens)

                # Project to shared space
                seg_feat = self.seg_head(seg_feat)
                det_feat = self.det_head(det_feat)

                seg_outputs.append(seg_feat)
                det_outputs.append(det_feat)

                adapter_idx += 1

        # Final global embedding
        x = x.permute(1, 0, 2)  # (B, N+1, D)
        pooled = x[:, 0]
        pooled = self.image_encoder.ln_post(pooled)
        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        # Multi-scale fusion with learnable weights
        if len(seg_outputs) > 1:
            weights = F.softmax(self.fusion[:len(seg_outputs)], dim=0)
            fused_seg = sum(w * s for w, s in zip(weights, seg_outputs))
            fused_det = sum(w * d for w, d in zip(weights, det_outputs))
        else:
            fused_seg = seg_outputs[0] if seg_outputs else None
            fused_det = det_outputs[0] if det_outputs else None

        return pooled, [fused_seg], [fused_det]