import torch
from collections import OrderedDict

def remap_biomedclip_to_openai_keys(biomed_state):
    """
    Convert BiomedCLIP vision keys:
      visual.trunk.*  -> visual.*
      visual.head.*   -> visual.*
    into OpenAI CLIP-style keys expected by CustomTextCLIP:
      visual.conv1, visual.positional_embedding, visual.transformer.resblocks.*, etc.
    Text keys already follow a BERT-style layout, so we leave them unchanged.
    """
    new_state = OrderedDict()

    for k, v in biomed_state.items():
        new_k = k

        # --- vision: patch embed / pos embed / cls token ---
        if k.startswith("visual.trunk.patch_embed.proj."):
            # weight/bias
            new_k = k.replace("visual.trunk.patch_embed.proj.", "visual.conv1.")
        elif k == "visual.trunk.pos_embed":
            # (1, 197, 768) -> positional_embedding (197, 768)
            new_k = "visual.positional_embedding"
            v = v.squeeze(0)  # (1, L, C) -> (L, C)
        elif k == "visual.trunk.cls_token":
            # (1, 1, 768) -> class_embedding (768,)
            new_k = "visual.class_embedding"
            v = v.squeeze(0).squeeze(0)

        # --- vision: blocks -> transformer.resblocks ---
        elif k.startswith("visual.trunk.blocks."):
            # example: visual.trunk.blocks.0.norm1.weight
            parts = k.split(".")  # ['visual','trunk','blocks','0','norm1','weight']
            block_idx = parts[3]
            suffix = ".".join(parts[4:])  # 'norm1.weight', 'attn.qkv.weight', etc.

            # base prefix in OpenAI CLIP
            prefix = f"visual.transformer.resblocks.{block_idx}."

            if suffix.startswith("norm1.weight"):
                new_k = prefix + "ln_1.weight"
            elif suffix.startswith("norm1.bias"):
                new_k = prefix + "ln_1.bias"
            elif suffix.startswith("norm2.weight"):
                new_k = prefix + "ln_2.weight"
            elif suffix.startswith("norm2.bias"):
                new_k = prefix + "ln_2.bias"
            elif suffix.startswith("attn.qkv.weight"):
                new_k = prefix + "attn.in_proj_weight"
            elif suffix.startswith("attn.qkv.bias"):
                new_k = prefix + "attn.in_proj_bias"
            elif suffix.startswith("attn.proj.weight"):
                new_k = prefix + "attn.out_proj.weight"
            elif suffix.startswith("attn.proj.bias"):
                new_k = prefix + "attn.out_proj.bias"
            elif suffix.startswith("mlp.fc1.weight"):
                new_k = prefix + "mlp.c_fc.weight"
            elif suffix.startswith("mlp.fc1.bias"):
                new_k = prefix + "mlp.c_fc.bias"
            elif suffix.startswith("mlp.fc2.weight"):
                new_k = prefix + "mlp.c_proj.weight"
            elif suffix.startswith("mlp.fc2.bias"):
                new_k = prefix + "mlp.c_proj.bias"
            else:
                # if some unexpected suffix, just keep original key
                new_k = k

        # --- vision: final norm + proj ---
        elif k.startswith("visual.trunk.norm."):
            if k.endswith("weight"):
                new_k = "visual.ln_post.weight"
            elif k.endswith("bias"):
                new_k = "visual.ln_post.bias"
        elif k.startswith("visual.head.proj."):
            # visual.head.proj.weight -> visual.proj
            if k.endswith("weight"):
                new_k = "visual.proj"
            else:
                new_k = k  # there is no bias in CLIP proj

        # --- everything else (text tower etc.) stays the same ---
        new_state[new_k] = v

    return new_state
