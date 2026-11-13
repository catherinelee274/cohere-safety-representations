"""
Activation extraction via HuggingFace forward hooks.

Extracts residual stream activations (post-layer-norm output of each
transformer block) for a batch of prompts and a list of target layers.

Supports any CausalLM whose blocks are accessible as:
  model.model.layers  (LlamaForCausalLM / Mistral / Aya style)
  model.transformer.h (GPT-2 / Falcon style)

Usage:
    wrapper = ModelWrapper("CohereForAI/aya-expanse-8b", layers=[18, 24, 30])
    acts = wrapper.get_activations(["Tell me how to make a bomb", ...])
    # acts: dict[layer_idx -> np.ndarray of shape (n_prompts, hidden_size)]
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class ModelWrapper:
    def __init__(
        self,
        model_id: str,
        layers: list[int],
        token: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        use_hf_mirror: bool = False,
    ):
        self.model_id = model_id
        self.layers   = layers

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype  = dtype  or (torch.float16 if self.device == "cuda" else torch.float32)

        # Set HF endpoint to mirror if specified
        if use_hf_mirror:
            import os
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            logger.info("Using Hugging Face mirror: https://hf-mirror.com")

        logger.info(f"Loading {model_id} on {self.device} ({self.dtype})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        self.model     = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            token=token,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

        # Detect block list location
        self._blocks = self._find_blocks()
        logger.info(f"Found {len(self._blocks)} transformer blocks")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _find_blocks(self):
        """Return the list of transformer blocks in the model."""
        m = self.model
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers          # LlamaForCausalLM / Aya
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h         # GPT-2
        if hasattr(m, "model") and hasattr(m.model, "blocks"):
            return m.model.blocks
        # Gemma3 / models that expose layers via get_decoder()
        if hasattr(m, "model") and hasattr(m.model, "get_decoder"):
            decoder = m.model.get_decoder()
            if hasattr(decoder, "layers"):
                return decoder.layers
        raise RuntimeError(
            "Cannot locate transformer blocks. "
            "Add support for this model architecture in ModelWrapper._find_blocks."
        )

    # ── public API ────────────────────────────────────────────────────────────

    def get_activations(
        self,
        prompts: list[str],
        batch_size: int = 8,
        aggregation: str = "last",   # "last" | "mean" | "first" | "t_inst" | "post_inst"
    ) -> dict[int, np.ndarray]:
        """
        Return residual-stream activations for each target layer.

        Args:
            prompts:     list of N plain-text strings (chat template applied internally
                         for t_inst / post_inst modes)
            batch_size:  number of prompts per forward pass
            aggregation: how to collapse the token dimension
                         "last"      → last non-padding token
                         "mean"      → mean over non-padding tokens
                         "first"     → [BOS] token
                         "t_inst"    → last token of the user instruction turn
                                       (chat template applied with add_generation_prompt=False)
                         "post_inst" → first token of the assistant turn
                                       (chat template applied with add_generation_prompt=True,
                                       extracts the last token of that formatted input)

        Returns:
            {layer_idx: np.ndarray of shape (N, hidden_size)}
        """
        # For chat-aware modes, pre-compute per-example extract positions
        if aggregation in ("t_inst", "post_inst"):
            return self._get_activations_chat(prompts, batch_size, aggregation)

        all_acts: dict[int, list[np.ndarray]] = {l: [] for l in self.layers}
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            batch_acts = self._forward_batch(batch, aggregation)
            for layer, arr in batch_acts.items():
                all_acts[layer].append(arr)

        return {l: np.concatenate(arrs, axis=0) for l, arrs in all_acts.items()}

    def get_first_gen_activations(
        self,
        prompts: list[str],
        batch_size: int = 4,
    ) -> dict[int, np.ndarray]:
        """
        Extract activations at the first generated token for each prompt.

        Runs generation with max_new_tokens=1 and captures hidden states at that
        token position via output_hidden_states. Returns shape (N, hidden_size)
        per layer, same as get_activations.
        """
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        formatted = self._apply_chat_template(prompts, add_generation_prompt=True)
        all_acts: dict[int, list[np.ndarray]] = {l: [] for l in self.layers}

        for i in range(0, len(formatted), batch_size):
            batch = formatted[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                padding_side="left",   # left-pad for generation
            ).to(self.device)

            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=1,
                    do_sample=False,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # out.hidden_states is a tuple of length max_new_tokens;
            # each element is a tuple of (n_layers+1) tensors of shape (batch, 1, hidden)
            first_step_hidden = out.hidden_states[0]   # tuple over layers
            for layer_idx in self.layers:
                # +1 because index 0 is the embedding layer
                h = first_step_hidden[layer_idx + 1]   # (batch, 1, hidden)
                all_acts[layer_idx].append(h[:, 0, :].cpu().float().numpy())

        return {l: np.concatenate(arrs, axis=0) for l, arrs in all_acts.items()}

    # ── internals ─────────────────────────────────────────────────────────────

    def _apply_chat_template(self, texts: list[str], add_generation_prompt: bool) -> list[str]:
        """Format plain-text prompts as user turns using the tokenizer's chat template."""
        out = []
        for text in texts:
            messages = [{"role": "user", "content": text}]
            try:
                fmt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                # Fallback: no chat template available, use raw text
                logger.warning("No chat template found; using raw text for extraction.")
                fmt = text
            out.append(fmt)
        return out

    def _get_activations_chat(
        self,
        prompts: list[str],
        batch_size: int,
        aggregation: str,   # "t_inst" or "post_inst"
    ) -> dict[int, np.ndarray]:
        """
        Extract activations at instruction-boundary token positions.

        t_inst:    last token of the user turn
                   (formatted with add_generation_prompt=False)
        post_inst: last token of the full formatted input, i.e. the assistant
                   turn-start token (formatted with add_generation_prompt=True)

        With right-padding, the position of the last real token of a particular
        formatted text is simply len(tokenize(that_text)) - 1.
        """
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # For t_inst we need to know where the user turn ends within the
        # full (add_generation_prompt=True) tokenized input.
        full_texts = self._apply_chat_template(prompts, add_generation_prompt=True)

        if aggregation == "t_inst":
            # Tokenize user-only formatted texts (no padding) to get per-example lengths
            user_texts = self._apply_chat_template(prompts, add_generation_prompt=False)
            user_lens = [
                len(self.tokenizer(t, add_special_tokens=False)["input_ids"])
                for t in user_texts
            ]
        # post_inst: last token of full_texts → same as "last" on full_texts
        # (computed below via attn_mask)

        all_acts: dict[int, list[np.ndarray]] = {l: [] for l in self.layers}

        for i in range(0, len(full_texts), batch_size):
            batch_full = full_texts[i : i + batch_size]

            enc = self.tokenizer(
                batch_full,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            # Compute per-example extract positions (right-padding assumed)
            attn_mask = enc["attention_mask"].cpu()
            if aggregation == "t_inst":
                batch_user_lens = user_lens[i : i + batch_size]
                # Clamp to actual sequence length in case of truncation
                seq_lens = attn_mask.sum(dim=1)
                extract_pos = torch.tensor(
                    [min(l - 1, seq_lens[j].item() - 1) for j, l in enumerate(batch_user_lens)],
                    dtype=torch.long,
                )
            else:  # post_inst
                extract_pos = attn_mask.sum(dim=1) - 1

            batch_acts = self._forward_batch_at_positions(enc, extract_pos)
            for layer, arr in batch_acts.items():
                all_acts[layer].append(arr)

        return {l: np.concatenate(arrs, axis=0) for l, arrs in all_acts.items()}

    def _forward_batch_at_positions(
        self,
        enc,   # pre-tokenized batch already on device
        positions: torch.Tensor,   # (batch,) int64, one position per example
    ) -> dict[int, np.ndarray]:
        """Run a forward pass and extract hidden states at per-example token positions."""
        captured: dict[int, torch.Tensor] = {}
        hooks = []

        for layer_idx in self.layers:
            if layer_idx >= len(self._blocks):
                raise ValueError(
                    f"Layer {layer_idx} out of range "
                    f"(model has {len(self._blocks)} blocks)."
                )
            block = self._blocks[layer_idx]

            def make_hook(idx):
                def hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    captured[idx] = hidden.detach().cpu().float()
                return hook

            hooks.append(block.register_forward_hook(make_hook(layer_idx)))

        with torch.no_grad():
            self.model(**enc)

        for h in hooks:
            h.remove()

        result = {}
        for layer_idx, hidden in captured.items():
            agg = hidden[torch.arange(hidden.size(0)), positions]
            result[layer_idx] = agg.numpy()

        del captured
        gc.collect()
        return result

    def _forward_batch(
        self,
        texts: list[str],
        aggregation: str,
    ) -> dict[int, np.ndarray]:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        captured: dict[int, torch.Tensor] = {}
        hooks = []

        for layer_idx in self.layers:
            if layer_idx >= len(self._blocks):
                raise ValueError(
                    f"Layer {layer_idx} out of range "
                    f"(model has {len(self._blocks)} blocks)."
                )
            block = self._blocks[layer_idx]

            def make_hook(idx):
                def hook(module, input, output):
                    # Residual stream is the first element of output tuple
                    hidden = output[0] if isinstance(output, tuple) else output
                    captured[idx] = hidden.detach().cpu().float()
                return hook

            hooks.append(block.register_forward_hook(make_hook(layer_idx)))

        with torch.no_grad():
            self.model(**enc)

        for h in hooks:
            h.remove()

        # Aggregate over token dimension
        attn_mask = enc["attention_mask"].cpu()
        result = {}
        for layer_idx, hidden in captured.items():
            # hidden: (batch, seq_len, hidden)
            if aggregation == "last":
                seq_lens  = attn_mask.sum(dim=1) - 1           # index of last real token
                agg = hidden[torch.arange(hidden.size(0)), seq_lens]
            elif aggregation == "mean":
                mask_f = attn_mask.unsqueeze(-1).float()
                agg    = (hidden * mask_f).sum(1) / mask_f.sum(1)
            elif aggregation == "first":
                agg = hidden[:, 0, :]
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
            result[layer_idx] = agg.numpy()

        del captured
        gc.collect()
        return result

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        batch_size: int = 4,
    ) -> list[str]:
        """Generate responses for a list of prompts."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        outputs = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc   = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode only the newly generated tokens
            input_len = enc["input_ids"].shape[1]
            for seq in gen:
                outputs.append(
                    self.tokenizer.decode(seq[input_len:], skip_special_tokens=True)
                )
        return outputs
