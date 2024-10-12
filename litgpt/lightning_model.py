from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from lightning import LightningModule
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import nn
from torch import Tensor

from litgpt.args import TrainArgs
from litgpt.config import Config
from litgpt.model import batched_index_select
from litgpt.model import Block
from litgpt.model import build_mask_cache
from litgpt.model import build_rope_cache
from litgpt.utils import chunked_cross_entropy


class LightningGPT(LightningModule):
    def __init__(self, config: Config, training_args: TrainArgs) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config
        self.model = None
        self.training_args = training_args

    def configure_model(self):
        if self.model is not None:
            return
        self.lm_head = nn.Linear(self.config.n_embd, self.config.padded_vocab_size, bias=self.config.lm_head_bias)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.config.padded_vocab_size, self.config.n_embd),
                h=nn.ModuleList(Block(self.config, block_idx) for block_idx in range(self.config.n_layer)),
                ln_f=self.config.norm_class(self.config.n_embd, eps=self.config.norm_eps),
            )
        )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

    def configure_optimizers(self) -> Dict[str, Any]:  # type: ignore
        """
        Return type is incorrectly hinted upstream. We return
        **Dictionary**, with an ``"optimizer"`` key, and (optionally) a ``"lr_scheduler"``
              key whose value is a single LR scheduler or ``lr_scheduler_config``.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.training_args.learning_rate)
        scheduler1 = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min(1.0, step / self.training_args.lr_warmup_steps)
        )
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(self.training_args.max_steps - self.training_args.lr_warmup_steps)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[scheduler1, scheduler2], milestones=[self.training_args.lr_warmup_steps]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # or "epoch" depending on your scheduler
                "monitor": "val_loss",  # Optional, if you use ReduceLROnPlateau or similar
            },
        }

    def forward(
        self,
        input_ids: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        T = input_ids.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # Use the KV cache
            cos = batched_index_select(self.cos, 0, input_pos)
            sin = batched_index_select(self.sin, 0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `set_kv_cache()`")
            mask = batched_index_select(self.mask_cache, 2, input_pos)
            if mask.dim() > 4:
                # The mask cache has a batch dim of 1 in addition to the one
                # we get if input_pos has a batch dimension
                mask = mask.squeeze(1)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.model.wte(input_ids)  # Token embeddings of shape (b, t, n_embd)
        if self.config.scale_embeddings:
            x = x * (self.config.n_embd**0.5)

        for block in self.model.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.model.ln_f(x)
        x = self.lm_head(x)  # (b, t, vocab_size)
        if self.config.final_logit_softcapping is not None:
            x = x.tanh() / self.config.final_logit_softcapping * self.config.final_logit_softcapping
        return x

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        input_ids = batch["input_ids"]
        targets = batch["labels"]

        logits = self.forward(input_ids)

        # Shift logits and targets to align output tokens with next input token
        logits = logits[..., :-1, :].contiguous()
        targets = targets[..., 1:].contiguous()

        loss = chunked_cross_entropy(logits, targets, chunk_size=128, ignore_index=-100)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        input_ids = batch["input_ids"]
        targets = batch["labels"]

        logits = self.forward(input_ids)

        # Shift logits and targets
        logits = logits[..., :-1, :].contiguous()
        targets = targets[..., 1:].contiguous()

        loss = chunked_cross_entropy(logits, targets, chunk_size=128, ignore_index=-100)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(
                f"Cannot attend to {value}, block size is only {self.config.block_size}."
                " This is likely because the input text exceeds the supported context length of this model."
            )
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache(device=self.cos.device)

    def rope_cache(self, device: Optional[torch_device] = None) -> Tuple[Tensor, Tensor]:

        if self.config.rope_adjustments is None:
            extra_config = None

        else:
            adjusted_params_required = ["factor", "low_freq_factor", "high_freq_factor", "original_max_seq_len"]
            params_present = [param in self.config.rope_adjustments for param in adjusted_params_required]
            num_params_present = sum(params_present)

            if num_params_present == 0:
                extra_config = None  # uses standard RoPE
            elif num_params_present == 4:
                # These parameters should always be used together so that we don't interfere with standard rope
                extra_config = {
                    "original_max_seq_len": self.config.rope_adjustments["original_max_seq_len"],
                    "factor": self.config.rope_adjustments["factor"],
                    "low_freq_factor": self.config.rope_adjustments["low_freq_factor"],
                    "high_freq_factor": self.config.rope_adjustments["high_freq_factor"],
                }
            else:
                # Some but not all parameters are specified; raise an error
                missing_params = [
                    param for param, present in zip(adjusted_params_required, params_present) if not present
                ]
                raise ValueError(
                    f"The following adjusted RoPE parameters are missing in rope_adjustments: {', '.join(missing_params)}. "
                    "All adjusted RoPE parameters must be specified together."
                )

        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
            extra_config=extra_config,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        max_seq_length: Optional[int] = None,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch_device] = None,
        dtype: Optional[torch_dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)

        if max_seq_length is None:
            max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.model.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size,
                max_seq_length,
                rope_cache_length,
                device,
                dtype,
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.model.h:
            block.attn.kv_cache = None
