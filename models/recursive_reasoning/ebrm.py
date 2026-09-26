"""Alternating energy descent with TRM's original ACT wrapper and loss head.

Use arch=ebrm with DISABLE_COMPILE=1. Evaluation enables state gradients
locally, including when the trainer calls the model under inference_mode.
"""

from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from models.layers import CastedLinear
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1 as TRM,
    TinyRecursiveReasoningModel_ACTV1Config as TRMConfig,
    TinyRecursiveReasoningModel_ACTV1InnerCarry as InnerCarry,
    TinyRecursiveReasoningModel_ACTV1_Inner as TRMInner,
)


class EBRMConfig(TRMConfig):
    H_cycles: int = Field(default=3, ge=1)
    L_cycles: int = Field(default=6, ge=1)
    H_steps: int = Field(default=1, ge=1)
    alpha_L: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    alpha_H: float = Field(default=0.01, gt=0, allow_inf_nan=False)
    use_z: bool = True  # False: no z_L latent, descend on z_H (y) only; L_cycles/alpha_L unused.
    no_ACT_continue: Literal[True] = True  # TRM's Q-halt-only loss.


class EBRMInner(TRMInner):
    def __init__(self, config: EBRMConfig):
        super().__init__(config)
        self.y_embed = CastedLinear(config.vocab_size, config.hidden_size, bias=False)
        # Gaussian states replace TRM's shared, fixed initial states.
        del self.H_init
        del self.L_init

    def empty_carry(self, batch_size: int):
        device = self.embed_tokens.embedding_weight.device
        return InnerCarry(
            # z_H is the output logits y; z_L is the hidden latent z (absent when use_z=False).
            z_H=torch.randn(
                batch_size, self.config.seq_len, self.config.vocab_size,
                device=device, dtype=torch.float32,
            ),
            z_L=torch.randn(
                batch_size, self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size, device=device, dtype=torch.float32,
            ) if self.config.use_z else None,
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: InnerCarry):
        reset = reset_flag.view(-1, 1, 1)
        return InnerCarry(
            z_H=torch.where(reset, torch.randn_like(carry.z_H), carry.z_H),
            z_L=torch.where(reset, torch.randn_like(carry.z_L), carry.z_L) if self.config.use_z else None,
        )

    def _features(self, input_embeddings, z_H, z_L, **seq_info):
        y_embedding = self.y_embed(z_H.to(self.forward_dtype))
        if self.puzzle_emb_len:
            y_embedding = F.pad(y_embedding, (0, 0, self.puzzle_emb_len, 0))

        # Math attention supports differentiation through the energy gradients.
        # This context affects only EBRM, leaving original TRM unchanged.
        with sdpa_kernel(SDPBackend.MATH):
            if self.config.use_z:
                return self.L_level(
                    z_L.to(self.forward_dtype), input_embeddings + y_embedding,
                    **seq_info,
                )
            # Without the latent, y itself is the state the network refines and x is the injection.
            return self.L_level(y_embedding, input_embeddings, **seq_info)

    def get_energy(self, input_embeddings, z_H, z_L, **seq_info):
        hidden = self._features(input_embeddings, z_H, z_L, **seq_info)
        residual = self.lm_head(hidden)[:, self.puzzle_emb_len:].float()
        return residual.sum(dim=(1, 2))

    def run_cycle(self, input_embeddings, z_H, z_L, create_graph: bool, **seq_info):
        for _ in range(self.config.L_cycles if self.config.use_z else 0):
            # A fresh coordinate makes this a partial derivative holding y
            # fixed, while preserving the outer training graph through both.
            z_L_current = z_L.clone()
            energy = self.get_energy(input_embeddings, z_H, z_L_current, **seq_info)
            grad_L = torch.autograd.grad(
                energy.sum(), z_L_current, create_graph=create_graph,
            )[0]
            z_L = z_L_current - self.config.alpha_L * grad_L
            if not create_graph:
                z_L = z_L.detach().requires_grad_(True)

        for _ in range(self.config.H_steps):
            # Hold the newly refined z fixed for each partial y derivative.
            z_H_current = z_H.clone()
            energy = self.get_energy(input_embeddings, z_H_current, z_L, **seq_info)
            grad_H = torch.autograd.grad(
                energy.sum(), z_H_current, create_graph=create_graph,
            )[0]
            z_H = z_H_current - self.config.alpha_H * grad_H
            if not create_graph:
                z_H = z_H.detach().requires_grad_(True)

        return z_H, z_L

    def forward(self, carry: InnerCarry, batch):
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        input_embeddings = self._input_embeddings(
            batch["inputs"], batch["puzzle_identifiers"],
        )
        z_H = carry.z_H.detach().requires_grad_(True)
        z_L = carry.z_L.detach().requires_grad_(True) if self.config.use_z else None

        # Keep TRM's cycle schedule; H_cycles=1 skips all warm-up cycles.
        for _ in range(self.config.H_cycles - 1):
            z_H, z_L = self.run_cycle(
                input_embeddings.detach(), z_H, z_L, create_graph=False, **seq_info,
            )

        z_H = z_H.detach().requires_grad_(True)
        z_L = z_L.detach().requires_grad_(True) if self.config.use_z else None
        z_H, z_L = self.run_cycle(
            input_embeddings, z_H, z_L, create_graph=self.training, **seq_info,
        )

        # Read the final contextual features for TRM's Q-head. Its auxiliary
        # loss trains the backbone as well as the head, just as in TRM.
        with torch.set_grad_enabled(self.training):
            hidden = self._features(input_embeddings, z_H, z_L, **seq_info)
            q_logits = self.q_head(hidden[:, 0]).float()

        new_carry = InnerCarry(z_H=z_H.detach(), z_L=z_L.detach() if z_L is not None else None)
        logits = z_H if self.training else z_H.detach()
        return new_carry, logits, (q_logits[..., 0], q_logits[..., 1])


class EBRM(TRM):
    """Use alternating z/y GD inside the unmodified TRM ACT wrapper."""

    def __init__(self, config_dict: dict):
        nn.Module.__init__(self)
        self.config = EBRMConfig(**config_dict)
        self.inner = EBRMInner(self.config)

    @torch.inference_mode(False)
    @torch.enable_grad()
    def forward(self, carry, batch):
        # The inherited wrapper performs replacements before the inner model,
        # producing ordinary tensors even when evaluation supplies inference
        # tensors. Keep its Q-based halting and exploration unchanged.
        return super().forward(carry, batch)
