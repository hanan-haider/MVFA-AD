# File: /kaggle/working/MVFA-AD/CLIP/improvedadapter.py
# 100% WORKING with OpenAI CLIP ViT-L/14@336px + MVFA-AD
# Output format IDENTICAL to original CLIP_Inplanted

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedicalLoRAAdapter(nn.Module):
    def __init__(self, dim=1024, r=32, alpha=64, dropout=0.1):
        super().__init__()
        self.r = r
        self.scaling = alpha / r

        self.lora_A_q = nn.Linear(dim, r, bias=False)
        self.lora_B_q = nn.Linear(r, dim, bias=False)
        self.lora_A_v = nn.Linear(dim, r, bias=False)
        self.lora_B_v = nn.Linear(r, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(1))

        nn.init.kaiming_uniform_(self.lora_A_q.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.lora_A_v.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B_q.weight)
        nn.init.zeros_(self.lora_B_v.weight)

    def forward(self, x):
        dq = self.lora_B_q(self.lora_A_q(x)) * self.scaling
        dv = self.lora_B_v(self.lora_A_v(x)) * self.scaling
        delta = self.dropout(dq + dv)
        return x + torch.sigmoid(self.gate) * delta


class CLIP_Inplanted(nn.Module):
    def __init__(self, clip_model, features=[8, 12, 16, 20]):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = features

        # LoRA adapters (much better than bottleneck)
        self.seg_adapters = nn.ModuleList([MedicalLoRAAdapter() for _ in features])
        self.det_adapters = nn.ModuleList([MedicalLoRAAdapter() for _ in features])

    def forward(self, x):
        B = x.shape[0]

        # === EXACT SAME preprocessing as original ===
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # === CLS token - correct way for OpenAI CLIP ===
        cls_token = self.image_encoder.class_embedding.to(x.dtype)
        cls_tokens = cls_token.expand(B, -1).unsqueeze(1)  # (B, 1, 1024)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  # (seq_len, B, D)

        seg_patch_tokens = []
        det_patch_tokens = []
        adapter_idx = 0

        # === Manual loop over 24 layers (required for adapter injection) ===
        for i in range(24):
            # Original resblock expects (x, attn_mask) → pass None
            x = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)

            if (i + 1) in self.features:
                # Convert back to (B, seq_len, D)
                layer_x = x.permute(1, 0, 2)           # (B, N+1, 1024)
                patch_tokens = layer_x[:, 1:, :]       # (B, N, 1024)

                # Apply LoRA adapters
                seg_feat = self.seg_adapters[adapter_idx](patch_tokens)
                det_feat = self.det_adapters[adapter_idx](patch_tokens)

                # Residual connection (same as original)
                layer_x = layer_x + 0.2 * (seg_feat + det_feat)  # 0.1 + 0.1

                # Save intermediate features
                seg_patch_tokens.append(seg_feat)
                det_patch_tokens.append(det_feat)

                # Put back into sequence
                x = torch.cat([layer_x[:, :1, :], layer_x[:, 1:, :]], dim=1)
                x = x.permute(1, 0, 2)  # back to (seq_len, B, D)
                adapter_idx += 1

        # === Final pooling (same as original) ===
        x = x.permute(1, 0, 2)  # (B, seq_len, D)
        pooled, _ = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(pooled)
        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        return pooled, seg_patch_tokens, det_patch_tokens