import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAAdapter(nn.Module):
    """
    Efficient LoRA Adapter for Vision Transformers (Q-LoRA style)
    Applied to both Q and V projections in attention
    """
    def __init__(self, hidden_dim=768, r=16, dropout=0.1, alpha=32):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # LoRA for Query
        self.lora_A_q = nn.Linear(hidden_dim, r, bias=False)
        self.lora_B_q = nn.Linear(r, hidden_dim, bias=False)
        
        # LoRA for Value
        self.lora_A_v = nn.Linear(hidden_dim, r, bias=False)
        self.lora_B_v = nn.Linear(r, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.ones(1) * 0.1)  # Learnable gating

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A_q.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.lora_A_v.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B_q.weight)
        nn.init.zeros_(self.lora_B_v.weight)

    def forward(self, x):
        # x: (B, N, D)
        delta_q = self.lora_B_q(self.lora_A_q(x)) * self.scaling
        delta_v = self.lora_B_v(self.lora_A_v(x)) * self.scaling
        delta = delta_q + delta_v
        delta = self.dropout(delta)
        return x + self.gate * delta


class EnhancedCLIPAdapter(nn.Module):
    """
    State-of-the-art adapter for BioMedCLIP Anomaly Detection & Segmentation
    """
    def __init__(self, clip_model, feature_layers=[3, 6, 9, 12], r=16, alpha=32):
        super().__init__()
        self.clipmodel = clip_model
        self.visual = clip_model.visual.trunk
        self.proj = clip_model.visual.proj
        self.hidden_dim = 768  # BioMedCLIP ViT-B/16

        self.feature_layers = feature_layers
        self.num_layers = len(self.visual.blocks)

        # Dual adapters: one for detection, one for segmentation
        self.seg_adapters = nn.ModuleList([
            LoRAAdapter(self.hidden_dim, r=r, alpha=alpha) for _ in feature_layers
        ])
        self.det_adapters = nn.ModuleList([
            LoRAAdapter(self.hidden_dim, r=r, alpha=alpha) for _ in feature_layers
        ])

        # Optional: lightweight MLP heads on top of adapted tokens
        self.seg_head = nn.Linear(self.hidden_dim, 512) if len(feature_layers) > 0 else None
        self.det_head = nn.Linear(self.hidden_dim, 512) if len(feature_layers) > 0 else None

        # Register forward hooks to capture intermediate features
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        self.features = {}
        
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook

        for i, layer_idx in enumerate(self.feature_layers):
            if layer_idx <= self.num_layers:
                hook = self.visual.blocks[layer_idx - 1].register_forward_hook(
                    get_hook(f'layer{layer_idx}')
                )
                self.hooks.append(hook)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def forward(self, x):
        B = x.shape[0]
        self.features.clear()

        # Forward through the frozen BioMedCLIP backbone
        # This triggers hooks automatically
        _ = self.clipmodel.visual(x)  # We only use this to populate self.features

        # Extract features from hooked layers
        seg_tokens = []
        det_tokens = []

        for i, layer_idx in enumerate(self.feature_layers):
            key = f'layer{layer_idx}'
            if key not in self.features:
                continue
            feats = self.features[key]  # (B, N+1, 768)

            # Remove CLS token and apply spatial tokens only
            patch_tokens = feats[:, 1:, :]  # (B, 196, 768)

            # Apply adapters
            seg_adapted = self.seg_adapters[i](patch_tokens)
            det_adapted = self.det_adapters[i](patch_tokens)

            # Optional lightweight head
            if self.seg_head is not None:
                seg_adapted = self.seg_head(seg_adapted)
            if self.det_head is not None:
                det_adapted = self.det_head(det_adapted)

            seg_tokens.append(seg_adapted)
            det_tokens.append(det_adapted)

        # Global image embedding (from original model)
        with torch.no_grad():
            image_embeds = self.clipmodel.encode_image(x).float()  # (B, 512)

        return image_embeds, seg_tokens, det_tokens

    def train(self, mode=True):
        super().train(mode)
        # Keep backbone frozen
        self.clipmodel.eval()
        for param in self.clipmodel.parameters():
            param.requires_grad = False
        return self