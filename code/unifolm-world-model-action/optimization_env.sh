#!/usr/bin/env bash

# Reproduce the validated ASC26 inference configuration.  Source this file
# from the repository root when explicit environment settings are preferred;
# the same values are also defaults in world_model_interaction.py.
_wma_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export HF_HUB_OFFLINE=1
if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]]; then
    _wma_libstdcxx="${CONDA_PREFIX}/lib/libstdc++.so.6"
    case ":${LD_PRELOAD:-}:" in
        *":${_wma_libstdcxx}:"*) ;;
        *) export LD_PRELOAD="${_wma_libstdcxx}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
    esac
fi
export WMA_MATH_MODE=fp16
export WMA_FP16_FULL_REDUCTION=1
export WMA_TORCH_COMPILE=1
export WMA_TORCH_COMPILE_BACKEND=inductor
export WMA_TORCH_COMPILE_MODE=default
export WMA_TORCH_COMPILE_FULLGRAPH=1
export WMA_TORCH_COMPILE_FX_GRAPH_CACHE=1
export WMA_TORCH_COMPILE_VALIDATE=1
export WMA_CACHE_OBSERVATION_ENCODER=1
export WMA_PHASE_HEAD_PRUNING=1
export WMA_GENERIC_PHASE_HEAD=1
export WMA_SDPA_ACTION_ATTENTION=1
export WMA_SKIP_OPENCLIP_PRETRAINED=1
export WMA_SKIP_INFERENCE_EMA=1
export WMA_CHECKPOINT_MMAP=1
export WMA_CHECKPOINT_ASSIGN=1
export WMA_META_INIT=1
export WMA_FINAL_ONLY_OUTPUT=1
export WMA_AOT_INDUCTOR=1
export WMA_AOT_INDUCTOR_BUILD=0
export WMA_AOT_INDUCTOR_ARTIFACT="${_wma_repo_root}/aot_package/model.so"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/torchinductor_unifolm_wma_fx_v1}"
export WMA_BATCH_VAE=0
export WMA_CUDA_CAPTURE=0
export WMA_INDUCTOR_FUSION=0
export WMA_INDUCTOR_PRECAST_WEIGHTS=0
export WMA_CONV3D_CHANNELS_LAST=0
export WMA_NVTX=0
export WMA_STAGE_TIMING=0
export WMA_STARTUP_TIMING=0

unset _wma_repo_root
unset _wma_libstdcxx
