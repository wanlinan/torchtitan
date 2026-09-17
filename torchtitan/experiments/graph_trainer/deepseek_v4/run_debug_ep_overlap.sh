#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$repo_root"

export NGPU=${NGPU:-4}
export LOG_RANK=${LOG_RANK:-0}
export MODULE=graph_trainer.deepseek_v4
export CONFIG=graph_trainer_deepseek_v4_debugmodel_ep_overlap
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# The native debug model has four routed experts. EP must divide that count.
ep_degree=${EP_DEGREE:-4}
if [[ "$ep_degree" != 2 && "$ep_degree" != 4 ]]; then
    echo "EP_DEGREE must be 2 or 4 for the DSV4 debug model." >&2
    exit 1
fi
if (( NGPU % ep_degree != 0 )); then
    echo "NGPU must be divisible by EP_DEGREE." >&2
    exit 1
fi

exec bash run_train.sh \
    --training.steps 5 \
    --parallelism.data_parallel_shard_degree "$NGPU" \
    --parallelism.tensor_parallel_degree 1 \
    --parallelism.expert_parallel_degree "$ep_degree" \
    --debug.seed 42 \
    --compile.debug_graph_passes \
    "$@"
