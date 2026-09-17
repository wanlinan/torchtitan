# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DSV4 GraphTrainer entry, numerics, and distributed FX transformation checks."""

import logging
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import flex_attention

from torchtitan.config import ConfigManager
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.utils import get_spmd_context
from torchtitan.experiments.graph_trainer.deepseek_v4.config_registry import (
    graph_trainer_deepseek_v4_debugmodel,
    graph_trainer_deepseek_v4_debugmodel_ep_overlap,
)
from torchtitan.experiments.graph_trainer.deepseek_v4.model import (
    GraphTrainerDeepSeekV4Model,
)
from torchtitan.experiments.graph_trainer.ep_chunk_pass import (
    prepare_ep_overlap_trace_call_inputs,
    prepare_ep_overlap_trace_inputs,
)
from torchtitan.experiments.graph_trainer.ep_pass_utils import collect_chunked_regions
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
from torchtitan.experiments.graph_trainer.passes import (
    apply_graph_passes,
    construct_default_graph_passes,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer, make_fwd_bwd_step
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.deepseek_v4.config_registry import deepseek_v4_debugmodel


@pytest.fixture
def deterministic_cpu():
    num_threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            yield
    finally:
        torch.set_num_threads(num_threads)
        torch.use_deterministic_algorithms(deterministic)


def test_cli_entry():
    config = ConfigManager().parse_args(
        [
            "--module",
            "graph_trainer.deepseek_v4",
            "--config",
            "graph_trainer_deepseek_v4_debugmodel_ep_overlap",
            "--parallelism.data_parallel_shard_degree",
            "4",
            "--training.steps",
            "5",
        ]
    )
    assert isinstance(config, GraphTrainer.Config)
    assert isinstance(config.model_spec.model, GraphTrainerDeepSeekV4Model.Config)
    assert config.model_spec.model.compress_ratios == (0, 0, 4, 128)
    assert config.model_spec.model.n_mtp_layers == 0
    assert config.compile.ep_overlap.enabled
    assert config.compile.ep_overlap.module_fqn == "layers.*.moe"
    assert config.compile.ep_overlap.strategy == "graph"
    assert config.parallelism.expert_parallel_degree == 4
    assert config.training.disable_cuda_graphs
    assert "cudagraph_pass" in config.compile.disable_passes


def _build_initialized(config):
    config.model_spec.model.update_from_config(config=config)
    model = config.model_spec.model.build()
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cpu"))
    return model


def test_native_model_parity_and_unused_indexer(deterministic_cpu):
    baseline = _build_initialized(deepseek_v4_debugmodel(seq_len=512))
    model = _build_initialized(graph_trainer_deepseek_v4_debugmodel())
    # strict=True also verifies checkpoint names and shapes are unchanged.
    model.load_state_dict(baseline.state_dict(), strict=True)
    tokens = torch.randint(0, 2048, (512,))
    labels = torch.randint(0, 2048, (512,))
    positions = torch.arange(512, dtype=torch.int32)
    # Execute Flex's mathematical reference on CPU in both models. This tests
    # model equivalence, not CUDA kernel numerics or performance.
    with (
        patch.object(FlexInnerAttention, "_compiled_flex_attn", flex_attention),
        patch("torch.nn.attention.flex_attention._validate_device"),
    ):
        baseline_output = baseline(tokens, positions=positions)
        output = model(tokens, positions=positions)
        torch.testing.assert_close(output, baseline_output, rtol=0, atol=0)
        baseline_loss = torch.nn.functional.cross_entropy(baseline_output, labels)
        loss = torch.nn.functional.cross_entropy(output, labels)
        torch.testing.assert_close(loss, baseline_loss, rtol=0, atol=0)
        baseline_loss.backward()
        loss.backward()

    frozen = set()
    for (name, parameter), (base_name, base_parameter) in zip(
        model.named_parameters(), baseline.named_parameters(), strict=True
    ):
        assert name == base_name
        if base_parameter.grad is None:
            assert ".indexer." in name
            assert not parameter.requires_grad
            assert parameter.grad is None
            frozen.add(name)
        else:
            assert parameter.requires_grad
            torch.testing.assert_close(
                parameter.grad, base_parameter.grad, rtol=0, atol=0
            )
    assert frozen
    # Frozen indexer weights must also have the same optimizer behavior as
    # the native parameters with grad=None, including weight decay.
    torch.optim.AdamW(baseline.parameters(), lr=8e-4).step()
    torch.optim.AdamW(model.parameters(), lr=8e-4).step()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, baseline.state_dict()[name], rtol=0, atol=0)


def test_ep_graph_traces_and_schedules_all_dsv4_layers(deterministic_cpu, caplog):
    config = graph_trainer_deepseek_v4_debugmodel_ep_overlap()
    config.parallelism.data_parallel_shard_degree = 4
    # Only CUDA kernel compilation is omitted; rematerialization, chunking,
    # process-group isolation, and EP scheduling all run unchanged.
    config.compile.disable_passes += [
        "annotate_flex_attention_for_regional_inductor_pass",
        "regional_inductor_pass",
    ]
    dist.init_process_group("fake", store=dist.HashStore(), rank=0, world_size=4)
    try:
        with patch("torchtitan.distributed.parallel_dims.device_type", "cpu"):
            dims = ParallelDims(
                dp_replicate=1, dp_shard=4, cp=1, tp=1, pp=1, ep=4, world_size=4
            )
            dims.build_mesh()
        # Initialize before sharding: DTensor RNG initialization needs an
        # accelerator. This test only executes graph transformations.
        model = _build_initialized(config)
        model = config.model_spec.parallelize_fn(
            model,
            parallel_dims=dims,
            training=config.training,
            parallelism=config.parallelism,
            compile_config=config.compile,
            ac_config=config.activation_checkpoint,
            dump_folder="unused",
        )
        loss_fn = config.loss.build(compile_config=config.compile)
        loss_fn.set_lm_head(model.lm_head)
        model._skip_lm_head = True
        args = (
            torch.randint(0, 2048, (512,)),
            torch.randint(0, 2048, (512,)),
            torch.tensor(2048),
            {
                "positions": torch.arange(512, dtype=torch.int32),
                "attention_masks": None,
            },
        )
        # make_fx uses fake tensors here. Bypass only Flex's CPU backward
        # device check; retain the real attention HOP and its backward graph.
        with (
            patch("torch.nn.attention.flex_attention._validate_device"),
            get_spmd_context(parallel_dims=dims, spmd_typechecking=False)(),
        ):
            traced = minimal_fx_tracer(
                make_fwd_bwd_step(model, loss_fn),
                module=model,
                prepare_inputs=lambda args, kwargs: prepare_ep_overlap_trace_inputs(
                    config.compile, args, kwargs
                ),
                prepare_call_inputs=lambda args, kwargs: prepare_ep_overlap_trace_call_inputs(
                    config.compile, args, kwargs
                ),
            )(*args)
        assert any(
            n.target == torch.ops.higher_order.flex_attention
            for n in traced.gm.graph.nodes
        )
        with caplog.at_level(logging.INFO):
            gm = apply_graph_passes(
                traced.gm,
                traced.example_inputs,
                construct_default_graph_passes(traced, config, parallel_dims=dims),
                compile_config=config.compile,
            )
        gm.graph.lint()
        regions = collect_chunked_regions(gm, module_pattern="layers.*.moe")
        assert {(r.root_fqn, r.is_backward) for r in regions} == {
            (f"layers.{i}.moe", backward)
            for i in range(4)
            for backward in (False, True)
        }
        for region in regions:
            assert set(region.bodies_by_chunk) == {0, 1}
            for body in region.bodies_by_chunk.values():
                assert any(
                    n.target == torch.ops._c10d_functional.all_to_all_single.default
                    for n in body.nodes
                )
        assert "Applied ep_overlap scheduling to 8 chunked region(s)" in caplog.text
    finally:
        dist.destroy_process_group()
