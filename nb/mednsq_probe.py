"""
MedNSQProbe: column-crush intervention + forward-difference Taylor scoring.

Single source of truth for the intervention. Both discovery and ablation use
`simulate_column_crush` and `restore_column` from this module.
"""

from typing import Any, Dict, List, Tuple

import torch


def _get_backbone(model):
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "transformer"):
        return model.transformer
    return model


def _get_layers(model):
    bb = _get_backbone(model)
    if hasattr(bb, "layers"):
        return bb.layers
    if hasattr(bb, "h"):
        return bb.h
    raise ValueError("Cannot locate transformer layers.")


def _get_mlp_down_proj(layer):
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
        return layer.mlp.down_proj
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "fc2"):
        return layer.mlp.fc2
    raise RuntimeError("Cannot locate MLP down projection.")


def _get_mlp_input_module(layer):
    """Return the module whose input is the post-SiLU intermediate vector
    (i.e., the down_proj). Hooking input here gives [B, T, intermediate]."""
    return _get_mlp_down_proj(layer)


class MedNSQProbe:
    def __init__(self, model):
        self.model = model
        self.layers = _get_layers(model)

        sample = self.layers[0]
        down = _get_mlp_down_proj(sample)
        # down_proj.weight: [hidden, intermediate]
        self.hidden_size = down.weight.shape[0]
        self.intermediate_size = down.weight.shape[1]

        cfg = model.config
        self.num_heads = getattr(cfg, "num_attention_heads", None)
        if self.num_heads is None and hasattr(cfg, "text_config"):
            self.num_heads = getattr(cfg.text_config, "num_attention_heads", None)
        self.head_dim = self.hidden_size // self.num_heads if self.num_heads else None

        print(f"[Probe] hidden={self.hidden_size} intermediate={self.intermediate_size} "
              f"layers={len(self.layers)} heads={self.num_heads}")

    def get_layer_weight(self, layer_idx: int) -> torch.Tensor:
        return _get_mlp_down_proj(self.layers[layer_idx]).weight

    # ------------------------------------------------------------------
    # Intervention: 1-bit column crush (sign * mean magnitude)
    # ------------------------------------------------------------------
    def simulate_column_crush(self, layer_idx: int, col_idx: int) -> torch.Tensor:
        weight = self.get_layer_weight(layer_idx)
        original_col = weight[:, col_idx].detach().clone()
        scale = original_col.abs().mean()
        crushed = original_col.sign() * scale
        with torch.no_grad():
            weight[:, col_idx] = crushed.to(weight.dtype)
        return original_col

    def restore_column(self, layer_idx: int, col_idx: int, original_col: torch.Tensor) -> None:
        weight = self.get_layer_weight(layer_idx)
        with torch.no_grad():
            weight[:, col_idx] = original_col.to(weight.dtype).to(weight.device)

    # ------------------------------------------------------------------
    # Batched margin evaluation on adversarial pairs
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_per_sample_margins(
        self,
        adv_pairs: List[Dict[str, Any]],
        batch_size: int = 32,
        pad_id: int = 0,
    ) -> torch.Tensor:
        if not adv_pairs:
            return torch.zeros(0, dtype=torch.float32)
        device = next(self.model.parameters()).device
        margins: List[float] = []

        for start in range(0, len(adv_pairs), batch_size):
            batch = adv_pairs[start: start + batch_size]
            seq_lens = [p["input_ids"].shape[1] for p in batch]
            max_len = max(seq_lens)
            ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
            mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
            for j, p in enumerate(batch):
                L = seq_lens[j]
                ids[j, :L] = p["input_ids"][0]
                mask[j, :L] = p["attention_mask"][0]

            out = self.model(input_ids=ids, attention_mask=mask)
            logits = out.logits

            for j, p in enumerate(batch):
                L = seq_lens[j]
                last_logits = logits[j, L - 1, :]
                m = (last_logits[p["pos_id"]] - last_logits[p["neg_id"]]).item()
                margins.append(m)

        return torch.tensor(margins, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Forward-only Taylor proxy: rank columns by how much zeroing each
    # column's contribution to the residual moves the (pos - neg) margin.
    #
    # For a single forward pass, we capture the down_proj input activation
    # `a` of shape [B, T, intermediate]. The contribution of neuron c to
    # residual is W[:, c] * a[..., c]. Zeroing it shifts the final-token
    # residual by -W[:, c] * a_last[c]. Projecting this through the rest
    # of the network is non-linear, but a first-order proxy is to estimate
    # the change in (pos_logit - neg_logit) by:
    #
    #   delta_margin_c ≈ (W_unembed[pos] - W_unembed[neg]) · (residual_shift_c)
    #
    # which factorizes as:
    #   delta_margin_c ≈ a_last[c] * ((W[:, c]) · (W_U[pos] - W_U[neg]))
    #
    # This ignores subsequent layer-norms and attention/MLP blocks but
    # gives a cheap, well-correlated ranking of columns to send to Stage 1.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_taylor_scores(
        self,
        layer_idx: int,
        adv_pairs: List[Dict[str, Any]],
        batch_size: int = 16,
        pad_id: int = 0,
        n_pairs: int = 200,
    ) -> torch.Tensor:
        """Returns a [intermediate_size] tensor of absolute mean Taylor proxy scores."""
        device = next(self.model.parameters()).device
        adv_pairs = adv_pairs[:n_pairs]
        if not adv_pairs:
            return torch.zeros(self.intermediate_size, dtype=torch.float32)

        # Get unembedding rows for pos/neg directions per pair
        # The unembedding is model.lm_head.weight: [vocab, hidden]
        lm_head = self.model.get_output_embeddings()
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise RuntimeError("Model has no lm_head with .weight; cannot compute Taylor proxy.")
        W_U = lm_head.weight  # [vocab, hidden]

        down_proj = _get_mlp_down_proj(self.layers[layer_idx])
        W_down = down_proj.weight  # [hidden, intermediate]

        score_accum = torch.zeros(self.intermediate_size, dtype=torch.float32, device=device)
        n_used = 0

        # Capture down_proj input via forward pre-hook
        captured: Dict[int, torch.Tensor] = {}

        def hook(module, args):
            captured["a"] = args[0].detach()  # [B, T, intermediate]

        handle = down_proj.register_forward_pre_hook(hook)
        try:
            for start in range(0, len(adv_pairs), batch_size):
                batch = adv_pairs[start: start + batch_size]
                seq_lens = [p["input_ids"].shape[1] for p in batch]
                max_len = max(seq_lens)
                ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
                mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
                for j, p in enumerate(batch):
                    L = seq_lens[j]
                    ids[j, :L] = p["input_ids"][0]
                    mask[j, :L] = p["attention_mask"][0]

                self.model(input_ids=ids, attention_mask=mask)
                a = captured["a"]  # [B, T, intermediate]

                # For each pair, take the activation at its real last token
                # and the (pos - neg) unembedding direction
                for j, p in enumerate(batch):
                    L = seq_lens[j]
                    a_last = a[j, L - 1, :].float()  # [intermediate]
                    u = (W_U[p["pos_id"]] - W_U[p["neg_id"]]).float()  # [hidden]
                    # column scores = a_last[c] * (W_down[:, c] · u)
                    col_dot = (W_down.float().t() @ u)  # [intermediate]
                    score = a_last * col_dot
                    score_accum += score.abs()
                    n_used += 1
        finally:
            handle.remove()

        return (score_accum / max(n_used, 1)).cpu()