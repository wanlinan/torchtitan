# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.deepseek_v4 import DeepSeekV4Model

from ..simple_fsdp import disable_active_parametrization


class GraphTrainerDeepSeekV4Model(DeepSeekV4Model):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4Model.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        # Native DSV4 has no indexer auxiliary loss yet. Its integer top-k
        # output disconnects these parameters from the loss. Keep their
        # values in the state dict, but exclude them from autograd.grad.
        for layer in self.layers.values():
            if hasattr(layer.attention, "indexer"):
                layer.attention.indexer.requires_grad_(False)
        self._token_alignments = sorted(
            {ratio for ratio in config.compress_ratios if ratio > 1}
            | {
                layer.attention.inner_attention.block_size
                if isinstance(layer.attention.inner_attention.block_size, int)
                else layer.attention.inner_attention.block_size[0]
                for layer in config.layers
            },
        )

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        padding_mask: torch.Tensor | None = None,
    ):
        # EP graph chunking marks the token count unbacked. Record the DSA
        # alignment contract before attention/compressor Python shape checks.
        for alignment in self._token_alignments:
            torch._check(tokens.size(0) % alignment == 0)
        return super().forward(
            tokens,
            positions=positions,
            attention_masks=attention_masks,
            padding_mask=padding_mask,
        )

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        with disable_active_parametrization():
            super().init_states(buffer_device=buffer_device)
