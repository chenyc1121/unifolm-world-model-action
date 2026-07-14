# Embodied World Model Submission

The parent archive follows the ASC26 Task 3 layout: `code/`,
`proposal_file/`, and `results/`. This file documents the project under
`code/unifolm-world-model-action/`.

## Inputs intentionally excluded

The archive does not contain checkpoints, reference videos, prompt images,
HDF5 transitions, normalization statistics, or compiler/model caches. Before
evaluation, place the official checkpoint at
`ckpts/unifolm_wma_dual.ckpt` and restore the
official case input trees. Restore the shared normalization dataset at
`examples/world_model_interaction_prompts`, or set `WMA_DATA_DIR` to its
absolute path.

## Run

Use the tested Python 3.10.20 environment with PyTorch 2.3.1+cu121 on an
NVIDIA A100 (SM80). From `code/unifolm-world-model-action` run an official
case script unchanged, for example:

```bash
pip install -e .
source optimization_env.sh  # recommended; also selects Conda libstdc++ when available
bash unitree_g1_pack_camera/case1/run_world_model_interaction.sh
```

The validated optimization settings are also the submission defaults, so the
official case script itself remains byte-identical to the supplied file. The
included AOTInductor package is bound to PyTorch 2.3.1+cu121, CUDA 12.1, SM80,
and FP16 full-reduction mode. On an incompatible platform, set
`WMA_AOT_INDUCTOR=0` to use the slower TorchInductor JIT fallback.

## Results and proposal

The archive-root `results/` directory contains the required 20 generated MP4
files, 20 inference logs, and `summary.json`. These are output artifacts
required by the rules, not input data. Archive-root
`proposal_file/project_report.pdf` is the English task report.
