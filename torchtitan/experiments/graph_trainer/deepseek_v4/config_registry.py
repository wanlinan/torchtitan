# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.graph_trainer.configs import (
    EpOverlapConfig,
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.deepseek_v4.config_registry import deepseek_v4_debugmodel

from . import model_registry


def graph_trainer_deepseek_v4_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        deepseek_v4_debugmodel(seq_len=512), model_registry
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.disable_cuda_graphs = True
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        disable_passes=[
            "joint_transformer_block_bucketing_reordering_pass",
            "cudagraph_pass",
        ],
    )
    return config


def graph_trainer_deepseek_v4_debugmodel_ep_overlap() -> GraphTrainer.Config:
    config = graph_trainer_deepseek_v4_debugmodel()
    config.parallelism.expert_parallel_degree = 4
    config.compile.memory_policy = "full"
    config.compile.ep_overlap = EpOverlapConfig(
        enabled=True,
        strategy="graph",
        chunk_dim="seq",
        module_fqn="layers.*.moe",
    )
    return config
