from typing import List, Dict, Any

import torch


class MedNSQProbe:

    def __init__(self, model):
        self.model = model

    def get_layer_weight(self, layer_idx: int) -> torch.Tensor:
        layer_stack = self.model.model.layers
        return layer_stack[layer_idx].mlp.down_proj.weight

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
