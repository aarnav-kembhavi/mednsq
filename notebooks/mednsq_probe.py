from typing import List, Dict, Any

import torch


def _get_backbone(model):
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "transformer"):
        return model.transformer
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox
    return model


def _get_layers(model):
    backbone = _get_backbone(model)
    if hasattr(backbone, "layers"):
        return backbone.layers
    if hasattr(backbone, "h"):
        return backbone.h
    if hasattr(backbone, "decoder") and hasattr(backbone.decoder, "layers"):
        return backbone.decoder.layers
    raise ValueError("Unsupported architecture: cannot find transformer layers")


def _get_mlp_down_proj(layer):
    """Resolve MLP output projection (down_proj) for Gemma / LLaMA / Mistral-style blocks."""
    module = None
    # LLaMA-style: layer.mlp.down_proj
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
        module = layer.mlp.down_proj
    elif hasattr(layer, "mlp"):
        mlp = layer.mlp
        if hasattr(mlp, "down_proj"):
            module = mlp.down_proj
        elif hasattr(mlp, "fc2"):
            module = mlp.fc2
    if module is None and hasattr(layer, "feed_forward"):
        if hasattr(layer.feed_forward, "w2"):
            module = layer.feed_forward.w2
    if module is None:
        raise RuntimeError(
            "Unknown MLP structure: cannot find down_proj / fc2 / w2 on this layer"
        )
    if not hasattr(module, "weight"):
        raise RuntimeError("Invalid down_proj module: no weight")
    if module.weight.dim() != 2:
        raise RuntimeError("Invalid down_proj weight shape")
    return module


class MedNSQProbe:

    def __init__(self, model):
        # Fix: typo modele -> model
        self.model = model

        # Cache architecture once (centralized dimension inference)
        self.layers = _get_layers(model)
        sample_layer = self.layers[0]
        down_proj = _get_mlp_down_proj(sample_layer)

        self.hidden_size = down_proj.weight.shape[0]
        self.intermediate_size = down_proj.weight.shape[1]

        attn = sample_layer.self_attn

        self.num_heads = getattr(model.config, "num_attention_heads", None)
        if self.num_heads is None:
            text_config = getattr(model.config, "text_config", None)
            if text_config is not None:
                self.num_heads = getattr(text_config, "num_attention_heads", None)
        if self.num_heads is None:
            raise ValueError("Cannot infer num_attention_heads")

        self.head_dim = self.hidden_size // self.num_heads

        print("=== Model Dimensions ===")
        print("hidden_size:", self.hidden_size)
        print("intermediate_size:", self.intermediate_size)
        print("num_heads:", self.num_heads)
        print("head_dim:", self.head_dim)

    def get_layer_weight(self, layer_idx: int) -> torch.Tensor:
        return _get_mlp_down_proj(self.layers[layer_idx]).weight

    def _final_token_logits(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits[:, -1, :]

    def compute_contrastive_jacobian(self, layer_idx, adv_pairs):
        """Directional Taylor scoring over a simulated 1-bit column crush.

        This method intentionally retains the existing directional Taylor logic.
        """

        original_requires_grad = {}

        for name, p in self.model.named_parameters():
            original_requires_grad[name] = p.requires_grad
            p.requires_grad = False

        target_weight = self.get_layer_weight(layer_idx)
        target_weight.requires_grad = True

        grad_accum = torch.zeros_like(target_weight, dtype=torch.float32)

        try:
            for pair in adv_pairs:

                self.model.zero_grad(set_to_none=True)

                logits = self._final_token_logits(
                    pair["input_ids"], pair["attention_mask"]
                )

                margin = logits[0, pair["pos_id"]] - logits[0, pair["neg_id"]]
                margin.backward()

                if target_weight.grad is not None:
                    grad_accum += target_weight.grad.detach().to(torch.float32)

        finally:
            for name, p in self.model.named_parameters():
                p.requires_grad = original_requires_grad[name]

        with torch.no_grad():
            weight = self.get_layer_weight(layer_idx)
            original_weight = weight.detach().to(torch.float32)

            scale_per_col = original_weight.abs().mean(dim=0, keepdim=True)
            crushed_weight = original_weight.sign() * scale_per_col
            delta = crushed_weight - original_weight

            col_scores = torch.abs((grad_accum * delta).sum(dim=0))

        return col_scores

    def simulate_column_crush(self, layer_idx: int, col_idx: int):
        weight = self.get_layer_weight(layer_idx)
        original_col = weight[:, col_idx].clone()

        scale = original_col.abs().mean()
        crushed = original_col.sign() * scale

        with torch.no_grad():
            weight[:, col_idx] = crushed.to(weight.dtype)

        return original_col

    def restore_column(self, layer_idx: int, col_idx: int, original_col):
        weight = self.get_layer_weight(layer_idx)
        with torch.no_grad():
            weight[:, col_idx] = original_col.to(weight.dtype)

    def compute_per_sample_margins(self, adv_pairs: List[Dict[str, Any]]) -> torch.Tensor:
        """Compute baseline margins for each calibration sample individually.

        This is used to cache baseline margins once and reuse them across
        simulated crush evaluations.
        """
        margins: List[float] = []
        with torch.no_grad():
            for pair in adv_pairs:
                logits = self._final_token_logits(
                    pair["input_ids"], pair["attention_mask"]
                )
                margin = (
                    logits[0, pair["pos_id"]] - logits[0, pair["neg_id"]]
                ).item()
                margins.append(margin)

        if not margins:
            return torch.zeros(0, dtype=torch.float32)
        return torch.tensor(margins, dtype=torch.float32)

    def average_margin(self, adv_pairs):
        margins = self.compute_per_sample_margins(adv_pairs)
        if margins.numel() == 0:
            return 0.0
        return float(margins.mean().item())
