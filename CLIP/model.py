""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from dataclasses import dataclass
import logging
import math
from typing import Optional, Tuple, Union
from itertools import repeat
import collections.abc
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from .modified_resnet import ModifiedResNet
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer
from dataclasses import dataclass
from typing import Optional
from .hf_model import HFTextEncoder

@dataclass
class BioMedCLIPVisionCfg:
    image_size: int = 224
    layers: int = 12
    width: int = 768
    patch_size: int = 16
    heads: int = 12
    embed_dim: int = 512
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None
    patch_dropout: float = 0.0
    input_patchnorm: bool = False
    global_average_pool: bool = False
    attentional_pool: bool = False
    n_queries: int = 256
    attn_pooler_heads: int = 8
    timm_model_name: str = None
    timm_model_pretrained: bool = False
    timm_pool: str = ''
    timm_proj: str = 'linear'
    timm_proj_bias: bool = False
    timm_drop: float = 0.0
    timm_drop_path: Optional[float] = None
    output_tokens: bool = True
    cast_dtype: Optional[str] = None


@dataclass
class BioMedCLIPTextCfg:
    context_length: int = 256          # HF model id uses 256 max length
    vocab_size: int = 30522            # BERT-base vocab
    width: int = 768
    heads: int = 12
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None
    output_dim: int = 512              # matches vision embed_dim
    embed_cls: bool = True
    hf_model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    hf_tokenizer_name: Optional[str] = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    hf_proj_type: str = "mlp"
    hf_pooler_type: str = "cls_last_hidden_state_pooler"
    hf_model_pretrained: bool = True
    pad_id: int = 0
    output_tokens: bool = True


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: BioMedCLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU
    if isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.heads
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: BioMedCLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,

):
    print("\n Building text tower...")
    print("Here is the text config:", text_cfg,"\n")

    if isinstance(text_cfg, dict):
        text_cfg = BioMedCLIPTextCfg(**text_cfg)
    
    print("\nText config for text tower:", text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj_type=text_cfg.hf_proj_type,
            pooler_type=text_cfg.hf_pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )




class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: BioMedCLIPVisionCfg,
            text_cfg: BioMedCLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def build_model_from_biomedclip_state_dict(
    state_dict: dict,
    cast_dtype=torch.float16,
):
    """
    Build BiomedCLIP model from official Microsoft checkpoint state_dict
    (e.g. microsoft/BiomedCLIP-PubMedBERT-ViT-B-16)
    """

    print("Building BiomedCLIP model from state_dict...")
    print(f"   Total parameters in state_dict: {len(state_dict)}")

    # === Vision Tower (ViT-B/16) ===
    print("Parsing vision tower (ViT-B/16)...")
    vision_width = state_dict["visual.trunk.patch_embed.proj.weight"].shape[0]  # 768
    vision_layers = len([
        k for k in state_dict.keys()
        if k.startswith("visual.trunk.blocks.") and k.endswith(".attn.qkv.weight")
    ])
    vision_patch_size = state_dict["visual.trunk.patch_embed.proj.weight"].shape[-1]
    pos_embed_shape = state_dict["visual.trunk.pos_embed"].shape
    grid_size = int((pos_embed_shape[1] - 1) ** 0.5)
    image_size = vision_patch_size * grid_size

    print(f"   Vision: width={vision_width}, layers={vision_layers}, patch_size={vision_patch_size}, image_size={image_size}")

    # === Text Tower (PubMedBERT) ===
    print("Parsing text tower (PubMedBERT)...")
    transformer_width = state_dict["text.transformer.embeddings.word_embeddings.weight"].shape[1]
    vocab_size = state_dict["text.transformer.embeddings.word_embeddings.weight"].shape[0]
    context_length = state_dict["text.transformer.embeddings.position_embeddings.weight"].shape[0]

    transformer_layers = len([
        k for k in state_dict.keys()
        if k.startswith("text.transformer.encoder.layer.") and k.endswith(".attention.self.query.weight")
    ])
    transformer_heads = transformer_width // 64

    embed_dim = state_dict["visual.head.proj.weight"].shape[0]  # 512

    print(f"   Text: width={transformer_width}, heads={transformer_heads}, layers={transformer_layers}, "
          f"vocab={vocab_size}, ctx_len={context_length}, embed_dim={embed_dim}")

    # === Configs ===
    vision_cfg = BioMedCLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size   

    )
    print("\nvision configs inside build model from state dict:", vision_cfg)

    text_cfg = BioMedCLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
  
    )
    #print("\ntext configs inside build model from state dict:", text_cfg)

    print("Creating CustomTextCLIP model instance...")
    model = CustomTextCLIP(
        embed_dim=embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=False,
        cast_dtype=cast_dtype,
    )


    # One key ('text.transformer.embeddings.position_ids') in your state_dict is unused
    state_dict = {
        k: v for k, v in state_dict.items()
        if k in model.state_dict().keys()
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    return model 


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed