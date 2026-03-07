#!/bin/bash

: "${conda_env_name:?ERROR: Conda environment name is not set}"
: "${username:?ERROR: username is not set; Enter your username}"
: "${oneapi_base:?ERROR: oneAPI path is not set}"

source /home/${username}/miniforge3/etc/profile.d/conda.sh
conda activate ${conda_env_name}

module unload oneapi mpich intel_compute_runtime 2>/dev/null || true
module use /home/cchannui/software/graphics-compute-runtime/modulefiles
module load graphics-compute-runtime/hotfix_agama-ci-devel-1146.12

source ${oneapi_base}/compiler/latest/env/vars.sh
source ${oneapi_base}/mkl/latest/env/vars.sh
source ${oneapi_base}/ccl/latest/env/vars.sh --ccl-bundled-mpi=no
source ${oneapi_base}/pti/latest/env/vars.sh  
source ${oneapi_base}/umf/latest/env/vars.sh
source ${oneapi_base}/mpi/latest/env/vars.sh
