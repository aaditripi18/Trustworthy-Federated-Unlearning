"""Vision Transformer model factory.

Builds either a small ViT sized for the tiny synthetic inputs (fast bring-up)
or a named timm DeiT for real-scale runs. Both share one interface, so the rest
of the pipeline is agnostic to which is used. torch/timm are imported lazily so
the rest of the repo (data, metrics) stays importable without them.
"""
from __future__ import annotations


def build_model(model_cfg: dict, image_size: int, num_classes: int):
    """Return an nn.Module mapping (B, in_chans, H, W) -> (B, num_classes) logits."""
    import timm

    name = model_cfg.get("name", "small_vit")
    in_chans = model_cfg.get("in_chans", 1)

    if name == "small_vit":
        from timm.models.vision_transformer import VisionTransformer

        return VisionTransformer(
            img_size=image_size,
            patch_size=model_cfg.get("patch_size", 4),
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=model_cfg.get("embed_dim", 64),
            depth=model_cfg.get("depth", 4),
            num_heads=model_cfg.get("num_heads", 4),
            mlp_ratio=2.0,
        )

    # Named timm model (e.g., deit_small_patch16_224) for real-scale experiments.
    return timm.create_model(
        name,
        pretrained=model_cfg.get("pretrained", False),
        num_classes=num_classes,
        in_chans=in_chans,
        img_size=image_size,
    )


def resolve_device(pref: str = "auto") -> str:
    import torch

    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return "cuda"
    return "cpu"
