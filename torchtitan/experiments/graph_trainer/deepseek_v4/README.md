# DSV4 debug GraphTrainer with EP overlap

This entry uses the native GPU DeepSeek-V4 debug model: four layers, hidden
size 256, four routed experts, top-k 3, two hash-routing layers, HC width 4,
and attention compression ratios `(0, 0, 4, 128)`. MTP is disabled. It does
not import NPU patches or change the native token dispatcher or router-score
placement, so enabling it alone does not guarantee reproduction of an NPU FX
failure.

The smoke-test recipe uses one 512-token microbatch per DP rank and the checked-in
`tests/assets/tokenizer` and `c4_test` data. No model checkpoint or DSV4 tokenizer
download is required. Use the GPU environment that already runs this checkout's
GraphTrainer DSV3 entry; the dependencies in `requirements.txt` still apply.

## Run on four GPUs

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 \
  bash torchtitan/experiments/graph_trainer/deepseek_v4/run_debug_ep_overlap.sh
```

The script selects:

```bash
MODULE=graph_trainer.deepseek_v4 \
CONFIG=graph_trainer_deepseek_v4_debugmodel_ep_overlap \
CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 \
  bash run_train.sh \
    --training.steps 5 \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.tensor_parallel_degree 1 \
    --parallelism.expert_parallel_degree 4 \
    --debug.seed 42 \
    --compile.debug_graph_passes
```

The config enables `aot_fx_trace`, regional Inductor compilation, full
rematerialization, and graph EP overlap with `chunk_dim=seq` and
`module_fqn=layers.*.moe`. TP, CP, and PP are 1. Only the MoE regions are split;
DSV4 sparse attention remains over the full token stream.

CUDA graphs are disabled both in the trainer and in the graph-pass list because
the standard AllToAll dispatcher synchronizes token counts with the CPU. The
`joint_transformer_block_bucketing_reordering_pass` is also disabled to avoid
the previously observed collective-bucketing topological-sort failure. The EP
chunk, process-group isolation, and overlap scheduling passes remain enabled.

Expected evidence is `Applied ep_overlap scheduling to 8 chunked region(s)`
(four forward and four backward regions), followed by five training steps and
`Training completed`. Merely getting past config parsing is not a successful
validation. This is a functional smoke test, not a performance measurement.

For four GPUs with EP2, set `EP_DEGREE=2` on the script. For eight GPUs, use
`NGPU=8 EP_DEGREE=4` and expose eight CUDA devices. The debug model has only four
experts, so EP8 is invalid. Additional CLI flags are forwarded; for example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 \
  bash torchtitan/experiments/graph_trainer/deepseek_v4/run_debug_ep_overlap.sh \
    --training.steps 10 --compile.memory_policy default
```

`graph_trainer_deepseek_v4_debugmodel` is the corresponding EP-overlap-off recipe.
The initial integration targets the debug model with TP1/CP1/PP1 and graph MoE
chunking; full-layer or eager chunking, MTP, and pipeline splitting are not
covered by this entry.

## Implementation and validation

The wrapper declares DSA token-alignment constraints with `torch._check` so the
native attention/compressor shape checks accept EP's unbacked token dimension.
It also freezes the indexer parameters: native DSV4 currently has no indexer
auxiliary loss, and integer top-k indices disconnect those weights from the
training loss. Their state-dict entries and forward values are retained.
`simple_fsdp` preserves `requires_grad` when replacing a parameter with its
sharded form. Initializing the model uses GraphTrainer's existing disabled
parametrization context.

CPU checks:

```bash
pytest -q tests/unit_tests/cpu/test_graph_trainer_deepseek_v4.py \
  torchtitan/experiments/graph_trainer/tests/test_simple_fsdp.py
```

They cover the CLI entry, native-model loss/gradient/update parity with seed 42,
checkpoint compatibility, and full forward/backward FX tracing with Fake PG
DP4/EP4 through rematerialization and EP scheduling. CPU tests use Flex's eager
reference or fake tensors and bypass its CPU backward device check; they do not
validate CUDA kernels or NCCL execution.

The four-GPU integration test is registered in the GraphTrainer and H100 suites:

```bash
python -m torchtitan.experiments.graph_trainer.tests.integration_tests \
  outputs/dsv4_graph_integration \
  --test_name aot_fx_trace_deepseek_v4_ep_overlap --ngpu 4
```

The integration runner requires a new or empty output directory.
