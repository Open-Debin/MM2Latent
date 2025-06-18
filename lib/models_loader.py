import os
import clip
import torch


def load_clip(clip_type, device, subdir='../models'):
    """
    Loads a specified CLIP or FaRL model using a dictionary-based approach.
    """
    
    supported_types = ['vit16', 'vit32', 'farl_ep16', 'farl_ep64']
    if clip_type not in supported_types:
        raise ValueError(f"Unsupported CLIP type: {clip_type}. Supported types are {supported_types}.")

    # Handle standard CLIP models first
    if clip_type == 'vit32':
        return clip.load("ViT-B/32", device=device)
    if clip_type == 'vit16':
        return clip.load("ViT-B/16", device=device)

    # --- Handle FaRL models ---
    clip_model, preprocess = clip.load("ViT-B/16", device=device)

    # Use a dictionary to map type to filename
    farl_model_files = {
        'farl_ep16': 'FaRL-Base-Patch16-LAIONFace20M-ep16.pth',
        'farl_ep64': 'FaRL-Base-Patch16-LAIONFace20M-ep64.pth',
    }
    model_name = farl_model_files[clip_type]
    
    # Load the FaRL weights
    this_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(this_dir, subdir, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' does not exist!")

    farl_state = torch.load(model_path, map_location=device)
    clip_model.load_state_dict(farl_state["state_dict"], strict=False)

    return clip_model, preprocess

