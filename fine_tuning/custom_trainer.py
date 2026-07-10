import torch
import torch.nn as nn
from transformers import Trainer


class SeparateLossTrainer(Trainer):
    """
    Drop-in replacement for Trainer that additionally logs think_loss and
    code_loss to W&B (and any other reporter). Backprop is unchanged.

    Requires a `think_boundary` field in the dataset (precomputed in preprocess):
      - think region : output tokens up to think_boundary (inclusive)
      - code region  : output tokens after think_boundary

    If think_boundary == len(sequence), all output tokens are counted as code.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_think_buf: list[float] = []
        self._train_code_buf: list[float] = []
        self._eval_think_buf: list[float] = []
        self._eval_code_buf: list[float] = []

    # ------------------------------------------------------------------
    # core: compute normal loss + accumulate per-region losses
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        think_boundaries = inputs.pop("think_boundary", None)

        outputs = model(**inputs)
        loss = outputs.loss  # unchanged - used for backprop

        labels = inputs.get("labels")

        if labels is not None and think_boundaries is not None:
            # Causal-LM shift: predict token t+1 from position t
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            B, T, V = shift_logits.shape
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(
                shift_logits.view(-1, V), shift_labels.view(-1)
            ).view(B, T)

            valid = shift_labels != -100  # (B, T)

            think_mask = torch.zeros_like(valid)
            code_mask = torch.zeros_like(valid)

            for b in range(B):
                end = int(think_boundaries[b].item()) - 1
                if end > 0 and end < T:
                    think_mask[b, :end] = valid[b, :end]
                    code_mask[b, end:] = valid[b, end:]
                else:
                    code_mask[b] = valid[b]

            with torch.no_grad():
                def masked_mean(losses, mask):
                    denom = mask.sum().clamp(min=1)
                    return (losses * mask).sum() / denom

                think_val = masked_mean(token_losses, think_mask).item()
                code_val = masked_mean(token_losses, code_mask).item()

            if model.training:
                self._train_think_buf.append(think_val)
                self._train_code_buf.append(code_val)
            else:
                self._eval_think_buf.append(think_val)
                self._eval_code_buf.append(code_val)

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # inject accumulated metrics into every log call
    # ------------------------------------------------------------------

    def log(self, logs: dict, start_time=None):
        is_eval = "eval_loss" in logs
        if is_eval:
            if self._eval_think_buf:
                logs["eval_think_loss"] = sum(self._eval_think_buf) / len(self._eval_think_buf)
                self._eval_think_buf.clear()
            if self._eval_code_buf:
                logs["eval_code_loss"] = sum(self._eval_code_buf) / len(self._eval_code_buf)
                self._eval_code_buf.clear()
        else:
            if self._train_think_buf:
                logs["think_loss"] = sum(self._train_think_buf) / len(self._train_think_buf)
                self._train_think_buf.clear()
            if self._train_code_buf:
                logs["code_loss"] = sum(self._train_code_buf) / len(self._train_code_buf)
                self._train_code_buf.clear()

        # Handle both old (logs) and new (logs, start_time) signatures
        try:
            super().log(logs, start_time=start_time)
        except TypeError:
            super().log(logs)
