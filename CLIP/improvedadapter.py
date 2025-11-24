# File: /kaggle/working/MVFA-AD/CLIP/improvedadapter.py
# 100% compatible with MVFA-AD + OpenAI CLIP ViT-L/14 and ViT-L/14@336px
# Fixed cls_token bug + maximum performance

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
        # x: (B, N, D)
        dq = self.lora_B_q(self.lora_A_q(x)) * self.scaling
        dv = self.lora_B_v(self.lora_A_v(x)) * self.scaling
        delta = self.dropout(dq + dv)
        return x + torch.sigmoid(self.gate) * delta


class CLIP_MedicalAdapter(nn.Module):
    def __init__(self, clip_model, features=[8, 12, 16, 20]):
        super().__init__()
        self.clip = clip_model
        self.visual = clip_model.visual
        self.features = features

        # LoRA adapters
        self.seg_adapters = nn.ModuleList([MedicalLoRAAdapter() for _ in features])
        self.det_adapters = nn.ModuleList([MedicalLoRAAdapter() for _ in features])

        # Task-specific heads
        self.seg_head = nn.Linear(1024, 512)
        self.det_head = nn.Linear(1024, 512)

        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(len(features)) * 0.25)

    def forward(self, x):
        B = x.shape[0]

        # === Exact same preprocessing as original CLIP ===
        x = self.visual.conv1(x)                     # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)    # shape = [*, width, grid]
        x = x.permute(0, 2, 1)                       # shape = [*, grid, width]

        # === CLS token - THIS IS THE FIX ===
        # class_embedding is already (1, 1, 1024) → just repeat
        cls_tokens = self.visual.class_embedding.expand(B, -1, -1)  # (B, 1, 1024)
        x = torch.cat([cls_tokens, x], dim=1)                    # (B, 1 + grid, 1024)

        x = x + self.visual.positional_embedding
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # (1 + grid, B, 1024)

        seg_outputs = []
        det_outputs = []
        adapter_idx = 0

        # === Transformer blocks ===
        for i in range(len(self.visual.transformer.resblocks)):
            x = self.visual.transformer.resblocks[i](x)

            if (i + 1) in self.features:
                layer_x = x.permute(1, 0, 2)           # (B, 1+grid, 1024)
                patch_tokens = layer_x[:, 1:, :]        # (B, grid, 1024)

                seg_feat = self.seg_adapters[adapter_idx](patch_tokens)
                det_feat = self.det_adapters[adapter_idx](patch_tokens)

                seg_feat = self.seg_head(seg_feat)
                det_feat = self.det_head(det_feat)

                seg_outputs.append(seg_feat)
                det_outputs.append(det_feat)
                adapter_idx += 1

        # === Final global embedding ===
        x = x.permute(1, 0, 2)          # (B, 1+grid, 1024)
        x = self.visual.ln_post(x[:, 0, :])
        if self.visual.proj is not None:
            x = x @ self.visual.proj

        # === Multi-scale fusion ===
        weights = F.softmax(self.fusion_weights, dim=0)
        fused_seg = sum(w * s for w, s in zip(weights, seg_outputs))
        fused_det = sum(w * d for w, d in zip(weights, det_outputs))

        return x, [fused_seg], [fused_det]