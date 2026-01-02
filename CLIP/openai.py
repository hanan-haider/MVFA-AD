"""BiomedCLIP pretrained model functions

Adapted from https://github.com/openai/CLIP (MIT License).
"""

import os
import warnings
from typing import Optional, Union

import torch

#from .model import build_model_from_biomedclip_state_dict, get_cast_dtype
# from .model import convert_weights_to_lp  # optional

__all__ = ["load_biomedclip_model"]


def load_biomedclip_model(
        name: str,
        precision: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        jit: bool = False,
        cache_dir: Optional[str] = None,  # kept for API compatibility but unused
):
    """
    Load a BiomedCLIP model from a local .bin checkpoint (state_dict).

    Parameters
    ----------
    name : str
        Path to a model checkpoint containing the state_dict (.bin).
    precision: str
        'fp16', 'fp32', 'bf16', or AMP; if None defaults to 'fp32' on CPU, 'fp16' on CUDA.
    device : Union[str, torch.device]
        Device to put the loaded model on.
    jit : bool
        Ignored for BiomedCLIP .bin (always loads non-JIT nn.Module).
    cache_dir : Optional[str]
        Unused; kept only to match the original function signature.

    Returns
    -------
    model : torch.nn.Module
        The BiomedCLIP model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if precision is None:
        precision = "fp32" if device == "cpu" else "fp16"

    if os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found")

    # For BiomedCLIP .bin, we treat it as a plain state_dict, not a JIT archive.
    jit = False

    try:
        # This will almost always fail for .bin, and we fall back to state_dict path.
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
      
        state_dict = None
    except RuntimeError:
        if jit:
            warnings.warn(
                f"File {model_path} is not a JIT archive. Loading as a state dict instead"
            )
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")
     
    
    if not jit:
        cast_dtype = get_cast_dtype(precision)

        # Build BiomedCLIP model from state_dict
        try:
            model = build_model_from_biomedclip_state_dict(
                state_dict or model.state_dict(),
                cast_dtype=cast_dtype,
            )
            print("here is the state dict of model",model)
        except Exception:
            # Common pattern: checkpoint["state_dict"] with "module." prefixes
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                sd = state_dict["state_dict"]
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
                model = build_model_from_biomedclip_state_dict(sd, cast_dtype=cast_dtype)
                print("here is the state dict of model 2",model)
            else:
                raise

        model = model.to(device)
        if precision.startswith("amp") or precision == "fp32":
            model.float()
        # elif precision == "bf16":
        #     convert_weights_to_lp(model, dtype=torch.bfloat16)
        # elif precision == "fp16":
        #     convert_weights_to_lp(model, dtype=torch.float16)

        # If needed, set image_size explicitly for downstream code
        # e.g., model.visual.image_size = 224

        return model
