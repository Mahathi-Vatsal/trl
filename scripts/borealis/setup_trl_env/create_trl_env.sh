#!/bin/bash


: "${conda_env_name:?ERROR: Conda environment name is not set}"
: "${username:?ERROR: username is not set; Enter your username}"

source /home/${username}/miniforge3/etc/profile.d/conda.sh
conda create -y --name ${conda_env_name} python=3.10
conda activate ${conda_env_name}

cd /home/msalopan/peft/trl 
pip install e .
cd -
pip install -U transformers accelerate datasets safetensors
pip install -U peft pillow
pip freeze | grep -i '^nvidia-' | cut -d= -f1 | xargs -r pip uninstall -y
pip freeze | grep -iE '^(cuda-|cudnn-|nccl-)' | cut -d= -f1 | xargs -r pip uninstall -y
pip install -U "datasets>=2.19.0" "huggingface_hub>=1.0.0"
pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo transformers timm
pip uninstall -y torch
pip install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu  --no-deps
