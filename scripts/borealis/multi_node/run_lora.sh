#!/bin/bash

set -euo pipefail

: "${model_path:?ERROR: model_path is not set}"
: "${output_dir:?ERROR: output_dir is not set}"
: "${conda_env_name:?ERROR: Conda environment name is not set}"
: "${username:?ERROR: username is not set; Enter your username}"
: "${WW:?ERROR: work week is not set}"
: "${model_config:?ERROR: model_config(eg: 8b, 70b... is not set)}"
: "${trl_repo_path:?ERROR: trl repo path not set}"

: "${PBS_NODEFILE:?ERROR: PBS_NODEFILE is not set (must run via qsub)}"
: "${PBS_JOBID:?ERROR: PBS_JOBID is not set (must run via qsub)}"

mkdir -p "${output_dir}/${WW}/${model_config}/multi_node"
mkdir -p "${output_dir}/${WW}/${model_config}/multi_node/lora_logs"

export model_path output_dir WW model_config

unset FI_TCP_IFACE
export FI_PROVIDER="tcp"
unset CCL_KVS_MODE
export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_OFI_PROVIDER=tcp
export CCL_PROCESS_LAUNCHER=hydra
export PYTHONUNBUFFERED=1
export CCL_OP_SYNC=1
export CCL_WORKER_AFFINITY="5,13,21,29,37,45,57,65,73,81,89,97"
export CCL_ZE_DISABLE_PORT_CHECK=1

if [[ -n "${libfabric_root:-}" ]]; then
  export FI_PROVIDER_PATH="${libfabric_root}/lib/prov"
fi

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PYTHONUNBUFFERED=1
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200
export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-WARNING}"

export PYTHON
PYTHON=$(which python)
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200

export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

NNODES="$(sort -u "$PBS_NODEFILE" | wc -l)"
export NNODES

MASTER_NODE_RAW=$(head -n1 "$PBS_NODEFILE")
MASTER_NODE_SHORT=${MASTER_NODE_RAW%%.*}
MASTER_ADDR_RESOLVED=$(getent hosts "$MASTER_NODE_SHORT" | awk '{print $1; exit}')
export MASTER_ADDR="${MASTER_ADDR_RESOLVED:-$MASTER_NODE_SHORT}"
export MASTER_PORT="${MASTER_PORT:-29500}"

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
export WORLD_SIZE

echo "[INFO] NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE} WORLD_SIZE=${WORLD_SIZE}"
echo "[INFO] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[INFO] model_path=${model_path}"
echo "[INFO] output_dir=${output_dir}/${WW}/${model_config}/multi_node"
echo "[INFO] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "[INFO] PBS_NODEFILE=${PBS_NODEFILE}"

export HF_HOME=/home/msalopan/peft/data
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export FSDP_CPU_RAM_EFFICIENT_LOADING=1

start_sec=$(date +%s)

mpiexec --envall -n ${WORLD_SIZE} -ppn ${GPUS_PER_NODE} \
  bash -lc '
    export MASTER_ADDR='"$MASTER_ADDR"'
    export MASTER_PORT='"$MASTER_PORT"'
    export RANK=${PMI_RANK:-${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}

    PHYS_LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-${PMI_LOCAL_RANK:-$(( RANK % GPUS_PER_NODE ))}}}
    export ZE_AFFINITY_MASK=${PHYS_LOCAL_RANK}
    export LOCAL_RANK=0

    NODELOG="'"${output_dir}"'/'"${WW}"'/'"${model_config}"'/multi_node/lora_logs/run_$(hostname -s)_r${RANK}.log"
    mkdir -p "$(dirname "$NODELOG")"

    source /home/${username}/miniforge3/etc/profile.d/conda.sh
    conda activate ${conda_env_name}

    echo "[LAUNCH] host=$(hostname -s) RANK=$RANK PHYS_LOCAL_RANK=$PHYS_LOCAL_RANK LOCAL_RANK=$LOCAL_RANK WORLD_SIZE='"$WORLD_SIZE"' GPUS_PER_NODE='"$GPUS_PER_NODE"' ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK" | tee -a "$NODELOG"
    env | grep -E "PMI|PMIX|OMPI|CCL_|FI_|ZE_" | tee -a "$NODELOG"

    python -c "import torch; print(\"xpu_count=\", torch.xpu.device_count() if hasattr(torch,\"xpu\") else 0)" | tee -a "$NODELOG"

    $PYTHON -m trl.scripts.sft --fsdp "full_shard auto_wrap" \
        --fsdp_config "'"${trl_repo_path}"'/trl/examples/accelerate_configs/fsdp_config.json" \
        --model_name_or_path '"${model_path}"' \
        --max_steps 10 \
        --max_seq_len 2048 \
        --dataset_name trl-lib/Capybara \
        --learning_rate 2.0e-4 \
        --num_train_epochs 1 \
        --packing \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --gradient_checkpointing \
        --eval_strategy no \
        --eval_steps 10 \
        --save_strategy no \
        --lora_r 32 \
        --lora_alpha 16 \
        --use_peft \
        --use_4bit_quantization True \
        --output_dir "'"${output_dir}"'/'"${WW}"'/'"${model_config}"'/multi_node/lora_logs" \
    |& tee -a "${NODELOG}"
  '

end_sec=$(date +%s)
dur=$((end_sec - start_sec))
printf "Duration: %02d:%02d:%02d\n" $((dur/3600)) $(((dur%3600)/60)) $((dur%60))
