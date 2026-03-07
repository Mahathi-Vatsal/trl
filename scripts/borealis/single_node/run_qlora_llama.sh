#!/bin/bash


set -euo pipefail

: "${model_path:?ERROR: model_path is not set}"
: "${output_dir:?ERROR: output_dir is not set}"
: "${WW:?ERROR: work week is not set}"
: "${model_config:?ERROR: model_config(eg: 8b, 70b... is not set)}"
: "${trl_repo_path:?ERROR: trl repo path not set}"

mkdir -p ${output_dir}/${WW}
mkdir -p ${output_dir}/${WW}/${model_config}
mkdir -p ${output_dir}/${WW}/${model_config}/single_node

export HF_HOME=/home/msalopan/peft/data
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_ENABLE_HF_TRANSFER=0

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

accelerate launch --num_processes 6 --config_file=${trl_repo_path}/trl/examples/accelerate_configs/fsdp2_lora.yaml \
	-m trl.scripts.sft \
	--model_name_or_path "${model_path}" \
	--max_steps 10 \
	--max_seq_len 512 \
	--dataset_name trl-lib/Capybara \
	--learning_rate 2.0e-4 \
	--num_train_epochs 1 \
	--packing \
	--per_device_train_batch_size 2 \
	--gradient_accumulation_steps 8 \
	--gradient_checkpointing \
	--eval_strategy steps \
	--eval_steps 10 \
	--packing False \
	--include_tokens_per_second True \
	--include_num_input_tokens_seen True \
	--use_peft \
	--bf16 True \
	--lora_r 32 \
	--lora_alpha 16 2>&1 | tee ${output_dir}/${WW}/${model_config}/single_node/qlora.log
