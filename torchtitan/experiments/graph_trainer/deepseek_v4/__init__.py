# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields, replace

from torchtitan.experiments.graph_trainer.deepseek_v3.parallelize import (
    parallelize_deepseekv3,
)
from torchtitan.models.deepseek_v4 import model_registry as deepseek_v4_model_registry
from torchtitan.protocols.model_spec import ModelSpec

from .model import GraphTrainerDeepSeekV4Model


def model_registry(flavor: str, *, seq_len: int | None = None) -> ModelSpec:
    base = deepseek_v4_model_registry(flavor, seq_len=seq_len)
    model = GraphTrainerDeepSeekV4Model.Config(
        **{f.name: getattr(base.model, f.name) for f in fields(base.model)}
    )
    # DSV4 uses the same MoE annotations and simple FSDP setup as DSV3.
    # Pipeline splitting needs a DSV4-specific treatment of the HC head.
    return replace(
        base,
        name="graph_trainer/deepseek_v4",
        model=model,
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=None,
    )
