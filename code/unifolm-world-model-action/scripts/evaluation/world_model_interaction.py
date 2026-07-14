import argparse, os, glob
import contextlib
import hashlib
import json
import time

# Submission defaults.  Every option remains overrideable through the
# environment, but the official case scripts can use the validated optimized
# path without being modified.
_REPOSITORY_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import pandas as pd
import random
import torch
import torchvision
import h5py
import numpy as np
import logging
import einops
import warnings
import imageio
from concurrent.futures import ThreadPoolExecutor

from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict, defaultdict
from torch import nn
from eval_utils import populate_queues, log_to_tensorboard
from collections import deque
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from unifolm_wma.models.samplers.ddim import DDIMSampler
from unifolm_wma.utils.utils import instantiate_from_config


_STAGE_TIMING = os.environ.get("WMA_STAGE_TIMING", "0") == "1"
_NVTX_PROFILING = os.environ.get("WMA_NVTX", "0") == "1"
_CUDA_CAPTURE = os.environ.get("WMA_CUDA_CAPTURE", "0") == "1"
_BATCH_VAE = os.environ.get("WMA_BATCH_VAE", "0") == "1"
_MATH_MODE = os.environ.get("WMA_MATH_MODE", "fp16").lower()
_FP16_FULL_REDUCTION = os.environ.get("WMA_FP16_FULL_REDUCTION", "1") == "1"
_ASYNC_OUTPUT = os.environ.get("WMA_ASYNC_OUTPUT", "0") == "1"
_ASYNC_OUTPUT_MAX_PENDING = int(
    os.environ.get("WMA_ASYNC_OUTPUT_MAX_PENDING", "2"))
_STARTUP_TIMING = os.environ.get("WMA_STARTUP_TIMING", "0") == "1"
_SKIP_OPENCLIP_PRETRAINED = (
    os.environ.get("WMA_SKIP_OPENCLIP_PRETRAINED", "1") == "1")
_SKIP_INFERENCE_EMA = (
    os.environ.get("WMA_SKIP_INFERENCE_EMA", "1") == "1")
_CHECKPOINT_MMAP = os.environ.get("WMA_CHECKPOINT_MMAP", "1") == "1"
_CHECKPOINT_ASSIGN = os.environ.get("WMA_CHECKPOINT_ASSIGN", "1") == "1"
_TORCH_COMPILE = os.environ.get("WMA_TORCH_COMPILE", "1") == "1"
_COMPILE_BACKEND = os.environ.get("WMA_TORCH_COMPILE_BACKEND", "inductor")
_COMPILE_MODE = os.environ.get("WMA_TORCH_COMPILE_MODE", "default")
_COMPILE_FULLGRAPH = os.environ.get("WMA_TORCH_COMPILE_FULLGRAPH", "1") == "1"
_COMPILE_VALIDATE = os.environ.get("WMA_TORCH_COMPILE_VALIDATE", "1") == "1"
_COMPILE_FX_GRAPH_CACHE = (
    os.environ.get("WMA_TORCH_COMPILE_FX_GRAPH_CACHE", "1") == "1")
_INDUCTOR_FUSION = os.environ.get("WMA_INDUCTOR_FUSION", "0") == "1"
_INDUCTOR_MAX_FUSION_SIZE = int(
    os.environ.get("WMA_INDUCTOR_MAX_FUSION_SIZE", "128"))
_INDUCTOR_PRECAST_WEIGHTS = (
    os.environ.get("WMA_INDUCTOR_PRECAST_WEIGHTS", "0") == "1")
_CONV3D_CHANNELS_LAST = (
    os.environ.get("WMA_CONV3D_CHANNELS_LAST", "0") == "1")
_CACHE_OBSERVATION_ENCODER = (
    os.environ.get("WMA_CACHE_OBSERVATION_ENCODER", "1") == "1")
_PHASE_HEAD_PRUNING = (
    os.environ.get("WMA_PHASE_HEAD_PRUNING", "1") == "1")
_GENERIC_PHASE_HEAD = (
    os.environ.get("WMA_GENERIC_PHASE_HEAD", "1") == "1")
_AOT_INDUCTOR = os.environ.get("WMA_AOT_INDUCTOR", "1") == "1"
_AOT_INDUCTOR_BUILD = (
    os.environ.get("WMA_AOT_INDUCTOR_BUILD", "0") == "1")
_AOT_INDUCTOR_ARTIFACT = os.environ.get(
    "WMA_AOT_INDUCTOR_ARTIFACT",
    os.path.join(_REPOSITORY_ROOT, "aot_package", "model.so"))
_AOT_SIGNATURE_DUMP = os.environ.get("WMA_AOT_SIGNATURE_DUMP", "")
_META_INIT = os.environ.get("WMA_META_INIT", "1") == "1"
_FINAL_ONLY_OUTPUT = os.environ.get("WMA_FINAL_ONLY_OUTPUT", "1") == "1"
_SDPA_ACTION_ATTENTION = (
    _TORCH_COMPILE or
    os.environ.get("WMA_SDPA_ACTION_ATTENTION", "0") == "1")
_CUDA_CAPTURE_DONE = False
_STAGE_TOTALS = defaultdict(float)
_STAGE_COUNTS = defaultdict(int)
_COMPILE_STATS = {}
_STARTUP_STATS = OrderedDict()


@contextlib.contextmanager
def profile_stage(name: str):
    """Optionally add an NVTX range and synchronized wall-clock timing."""
    global _CUDA_CAPTURE_DONE
    capture_this_stage = (_CUDA_CAPTURE and name == "ddim_sampling"
                          and not _CUDA_CAPTURE_DONE)
    if not (_STAGE_TIMING or _NVTX_PROFILING or capture_this_stage):
        yield
        return

    if _STAGE_TIMING and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    if _NVTX_PROFILING:
        torch.cuda.nvtx.range_push(name)
    if capture_this_stage:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        yield
    finally:
        if capture_this_stage:
            torch.cuda.cudart().cudaProfilerStop()
            _CUDA_CAPTURE_DONE = True
        if _NVTX_PROFILING:
            torch.cuda.nvtx.range_pop()
        if _STAGE_TIMING and torch.cuda.is_available():
            torch.cuda.synchronize()
        if _STAGE_TIMING:
            _STAGE_TOTALS[name] += time.perf_counter() - start
            _STAGE_COUNTS[name] += 1


def print_stage_profile() -> None:
    if not _STAGE_TIMING:
        return
    print(">>> Stage timing summary (s)")
    for name, total in sorted(_STAGE_TOTALS.items()):
        count = _STAGE_COUNTS[name]
        print(f">>> PROFILE {name}: total={total:.3f}, count={count}, avg={total / count:.3f}")


@contextlib.contextmanager
def profile_startup_stage(name: str, synchronize_cuda: bool = False):
    """Measure startup work without imposing overhead unless explicitly enabled."""
    if not _STARTUP_TIMING:
        yield
        return
    if synchronize_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if synchronize_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        _STARTUP_STATS[name] = time.perf_counter() - start


def print_startup_profile() -> None:
    if not _STARTUP_TIMING:
        return
    print(">>> Startup timing summary (s)")
    for name, elapsed in _STARTUP_STATS.items():
        print(f">>> STARTUP {name}: {elapsed:.3f}")


def precision_context():
    if _MATH_MODE == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if _MATH_MODE == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


class FixedHybridDenoiser(nn.Module):
    """Pure-Tensor adapter for the fixed-shape hybrid WMA inference path."""

    def __init__(self, diffusion_model: nn.Module, active_head="both"):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.active_head = active_head

    def forward(self, x, x_action, x_state, timestep, concat_condition,
                cross_attention, observation_image, observation_state, fs):
        model_input = torch.cat((x, concat_condition), dim=1)
        return self.diffusion_model(
            model_input,
            x_action,
            x_state,
            timestep,
            context=cross_attention,
            context_action=(observation_image, observation_state),
            active_head=self.active_head,
            fs=fs)


class CachedObservationHybridDenoiser(nn.Module):
    """Fixed hybrid adapter with sampler-level observation encodings."""

    def __init__(self, diffusion_model: nn.Module, active_head="both"):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.active_head = active_head

    def forward(self, x, x_action, x_state, timestep, concat_condition,
                cross_attention, observation_image, observation_state,
                action_observation_cond, state_observation_cond, fs):
        model_input = torch.cat((x, concat_condition), dim=1)
        return self.diffusion_model(
            model_input,
            x_action,
            x_state,
            timestep,
            context=cross_attention,
            context_action=(observation_image, observation_state),
            active_head=self.active_head,
            action_observation_cond=action_observation_cond,
            state_observation_cond=state_observation_cond,
            fs=fs)


class GenericPhaseHybridDenoiser(nn.Module):
    """Action-shaped template whose lifted head weights are phase-selectable."""

    def __init__(self, diffusion_model: nn.Module):
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(self, x, x_head, timestep, concat_condition,
                cross_attention, observation_image, observation_state, fs):
        model_input = torch.cat((x, concat_condition), dim=1)
        video, head, _ = self.diffusion_model(
            model_input,
            x_head,
            x_head,
            timestep,
            context=cross_attention,
            context_action=(observation_image, observation_state),
            active_head="action",
            fs=fs)
        return video, head


class CachedGenericPhaseHybridDenoiser(nn.Module):
    """Generic phase template using one precomputed observation condition."""

    def __init__(self, diffusion_model: nn.Module):
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(self, x, x_head, timestep, concat_condition,
                cross_attention, observation_image, observation_state,
                observation_cond, fs):
        model_input = torch.cat((x, concat_condition), dim=1)
        video, head, _ = self.diffusion_model(
            model_input,
            x_head,
            x_head,
            timestep,
            context=cross_attention,
            context_action=(observation_image, observation_state),
            active_head="action",
            action_observation_cond=observation_cond,
            fs=fs)
        return video, head


def _tensor_metadata(tensor: torch.Tensor):
    return (tuple(tensor.shape), tensor.dtype, tuple(tensor.stride()),
            tensor.requires_grad)


def _alias_topology(named_tensors):
    aliases = defaultdict(list)
    for name, tensor in named_tensors.items():
        aliases[id(tensor)].append(name)
    return sorted(tuple(names) for names in aliases.values() if len(names) > 1)


def validate_generic_head_compatibility(diffusion_model: nn.Module) -> None:
    """Reject generic execution unless both heads have identical tensor ABI."""
    action_head = diffusion_model.action_unet
    state_head = diffusion_model.state_unet
    total_tensors = 0
    for label, iterator in (
            ("parameter", nn.Module.named_parameters),
            ("buffer", nn.Module.named_buffers)):
        action_tensors = dict(
            iterator(action_head, recurse=True, remove_duplicate=False))
        state_tensors = dict(
            iterator(state_head, recurse=True, remove_duplicate=False))
        if action_tensors.keys() != state_tensors.keys():
            missing_action = sorted(state_tensors.keys() - action_tensors.keys())
            missing_state = sorted(action_tensors.keys() - state_tensors.keys())
            raise RuntimeError(
                f"Generic head {label} names differ: "
                f"missing_action={missing_action[:4]}, "
                f"missing_state={missing_state[:4]}")
        for name, action_tensor in action_tensors.items():
            state_tensor = state_tensors[name]
            if _tensor_metadata(action_tensor) != _tensor_metadata(state_tensor):
                raise RuntimeError(
                    f"Generic head {label} metadata differs at {name}: "
                    f"action={_tensor_metadata(action_tensor)}, "
                    f"state={_tensor_metadata(state_tensor)}")
        if _alias_topology(action_tensors) != _alias_topology(state_tensors):
            raise RuntimeError(
                f"Generic head {label} tied-tensor topology differs.")
        total_tensors += len(action_tensors)
    _COMPILE_STATS["generic_head_abi_tensors"] = total_tensors
    print(f">>> Generic action/state head ABI validated: tensors={total_tensors}")


class ExportedGenericPhaseRuntime:
    """Compile one exported action-shaped graph with explicit phase weights."""

    _CACHE_VERSION = 1
    _AOT_CACHE_VERSION = 3
    _ACTION_PREFIX = "diffusion_model.action_unet."
    _STATE_PREFIX = "diffusion_model.state_unet."

    def __init__(self, adapter: nn.Module, example_inputs, compile_options):
        from torch.export.graph_signature import InputKind

        self.adapter = adapter
        self.input_kind = InputKind
        self.exported_program = None
        self.constants = {}
        cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        cache_name = (
            f"wma_exported_generic_v{self._CACHE_VERSION}_"
            f"{type(adapter).__name__}_{_MATH_MODE}.pt")
        if _AOT_INDUCTOR and _AOT_INDUCTOR_ARTIFACT:
            # AOT deployment is bound to the exact serialized FX graph used
            # for compilation. Keep it beside the shared library so moving or
            # clearing TORCHINDUCTOR_CACHE_DIR cannot silently change the ABI.
            self.cache_path = (
                f"{os.path.abspath(_AOT_INDUCTOR_ARTIFACT)}.graph.pt")
        else:
            self.cache_path = (os.path.join(cache_dir, cache_name)
                               if cache_dir else None)
        aot_load_only = _AOT_INDUCTOR and not _AOT_INDUCTOR_BUILD
        if aot_load_only:
            if not _AOT_INDUCTOR_ARTIFACT:
                raise ValueError(
                    "WMA_AOT_INDUCTOR load-only mode requires "
                    "WMA_AOT_INDUCTOR_ARTIFACT.")
            metadata_path = (
                f"{os.path.abspath(_AOT_INDUCTOR_ARTIFACT)}.json")
            if (not self.cache_path or
                    not os.path.isfile(self.cache_path) or
                    not os.path.isfile(metadata_path)):
                raise FileNotFoundError(
                    "AOTInductor graph sidecar or metadata is missing; "
                    "load-only mode never regenerates package files.")
            with open(metadata_path, "r", encoding="utf-8") as handle:
                package_metadata = json.load(handle)
            expected_graph_digest = package_metadata.get(
                "graph_cache_sha256")
            actual_graph_digest = self._graph_cache_sha256()
            if (not expected_graph_digest or
                    actual_graph_digest != expected_graph_digest):
                raise RuntimeError(
                    "AOTInductor graph sidecar failed its package SHA-256 "
                    "check; rebuild the package instead of loading it.")
        input_signature = tuple(
            (tuple(value.shape), str(value.dtype), tuple(value.stride()))
            for value in example_inputs)
        cache_loaded = False
        cache_start = time.perf_counter()
        # A requested rebuild must export current source instead of silently
        # compiling a stale graph sidecar left by an older package.
        may_restore_graph = not (_AOT_INDUCTOR and _AOT_INDUCTOR_BUILD)
        if (may_restore_graph and self.cache_path and
                os.path.isfile(self.cache_path)):
            payload = torch.load(
                self.cache_path, map_location="cpu", weights_only=False)
            if (payload.get("version") == self._CACHE_VERSION and
                    payload.get("torch_version") == torch.__version__ and
                    payload.get("adapter") == type(adapter).__name__ and
                    payload.get("input_signature") == input_signature):
                self.graph_module = payload["graph_module"]
                self.input_specs = tuple(payload["input_specs"])
                device = example_inputs[0].device
                self.constants = {
                    # Exported scalar constants intentionally remain on CPU;
                    # the graph contains their explicit device transfers.
                    # Moving them to CUDA changes the AOT input ABI.
                    name: value
                    for name, value in payload["constants"].items()
                }
                self.graph_module.to(device=device)
                cache_loaded = True
        if aot_load_only and not cache_loaded:
            raise RuntimeError(
                "AOTInductor graph sidecar is incompatible with the current "
                "PyTorch version, adapter, or input signature. Load-only "
                "mode will not overwrite the package; rebuild it explicitly.")
        _COMPILE_STATS["generic_graph_cache_load_seconds"] = (
            time.perf_counter() - cache_start)

        if not cache_loaded:
            export_start = time.perf_counter()
            self.exported_program = torch.export.export(
                adapter, tuple(example_inputs), strict=True)
            _COMPILE_STATS["generic_export_seconds"] = (
                time.perf_counter() - export_start)
            self.graph_module = self.exported_program.graph_module
            self.input_specs = tuple(
                (spec.kind.name, spec.target)
                for spec in self.exported_program.graph_signature.input_specs)
            self.constants = dict(self.exported_program.constants)
            if self.cache_path:
                self._save_graph_cache(input_signature)
        else:
            _COMPILE_STATS["generic_export_seconds"] = 0.0
        _COMPILE_STATS["generic_graph_cache_hit"] = int(cache_loaded)
        self.user_input_count = sum(
            kind_name == InputKind.USER_INPUT.name
            for kind_name, _ in self.input_specs)
        if self.user_input_count != len(example_inputs):
            raise RuntimeError(
                "Exported generic graph changed the user-input arity: "
                f"expected={len(example_inputs)}, actual={self.user_input_count}")
        self.action_layout = self._build_static_layout("action")
        self.state_layout = self._build_static_layout("state")
        if _AOT_INDUCTOR:
            action_graph_inputs = self.inputs("action", example_inputs)
            self.compiled = self._load_or_build_aot(action_graph_inputs)
        else:
            self.compiled = torch.compile(self.graph_module,
                                          **compile_options)
        _COMPILE_STATS["generic_export_inputs"] = len(self.input_specs)
        print(
            ">>> Exported generic phase graph prepared: "
            f"inputs={len(self.input_specs)}, "
            f"user_inputs={self.user_input_count}, "
            f"graph_cache_hit={cache_loaded}, "
            f"export_seconds={_COMPILE_STATS['generic_export_seconds']:.3f}")

    def _graph_cache_sha256(self):
        if not self.cache_path or not os.path.isfile(self.cache_path):
            raise FileNotFoundError(
                "AOTInductor requires its serialized graph sidecar: "
                f"{self.cache_path}")
        digest = hashlib.sha256()
        with open(self.cache_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _aot_signature(self, graph_inputs):
        input_signature = tuple(
            (tuple(value.shape), str(value.dtype), tuple(value.stride()),
             value.device.type)
            for value in graph_inputs)
        device = graph_inputs[0].device
        signature = {
            "version": self._AOT_CACHE_VERSION,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_capability": list(torch.cuda.get_device_capability(device)),
            "math_mode": _MATH_MODE,
            "fp16_full_reduction": _FP16_FULL_REDUCTION,
            "inductor_fusion": _INDUCTOR_FUSION,
            "inductor_max_fusion_size": _INDUCTOR_MAX_FUSION_SIZE,
            "inductor_precast_weights": _INDUCTOR_PRECAST_WEIGHTS,
            "conv3d_channels_last": _CONV3D_CHANNELS_LAST,
            "cache_observation_encoder": _CACHE_OBSERVATION_ENCODER,
            "phase_head_pruning": _PHASE_HEAD_PRUNING,
            "generic_phase_head": _GENERIC_PHASE_HEAD,
            "sdpa_action_attention": _SDPA_ACTION_ATTENTION,
            "adapter": type(self.adapter).__name__,
            "graph_cache_sha256": self._graph_cache_sha256(),
            "input_count": len(graph_inputs),
            "output_count": 2,
            "input_signature_sha256": hashlib.sha256(
                repr(input_signature).encode("utf-8")).hexdigest(),
        }
        if _AOT_SIGNATURE_DUMP:
            dump_path = os.path.abspath(_AOT_SIGNATURE_DUMP)
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            detailed_inputs = []
            for index, ((kind, target), value) in enumerate(
                    zip(self.input_specs, graph_inputs)):
                detailed_inputs.append({
                    "index": index,
                    "kind": kind,
                    "target": str(target),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "stride": list(value.stride()),
                    "device": value.device.type,
                })
            with open(dump_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "signature": signature,
                    "inputs": detailed_inputs,
                    "graph_code": self.graph_module.code,
                }, handle, indent=2, sort_keys=True)
        return signature

    def _load_or_build_aot(self, graph_inputs):
        """Build or load one AOTInductor shared library for both phases."""
        if not _AOT_INDUCTOR_ARTIFACT:
            raise ValueError(
                "WMA_AOT_INDUCTOR requires WMA_AOT_INDUCTOR_ARTIFACT to "
                "name the precompiled shared library.")
        artifact = os.path.abspath(_AOT_INDUCTOR_ARTIFACT)
        if not artifact.endswith(".so"):
            raise ValueError("WMA_AOT_INDUCTOR_ARTIFACT must end in .so")
        metadata_path = f"{artifact}.json"
        expected_signature = self._aot_signature(graph_inputs)

        if _AOT_INDUCTOR_BUILD:
            from torch._inductor import aot_compile
            from torch.utils import _pytree as pytree

            os.makedirs(os.path.dirname(artifact), exist_ok=True)
            # The graph loaded from our lightweight FX cache uses ordinary
            # CodeGen rather than PyTreeCodeGen. PyTorch 2.3 therefore needs
            # explicit call specs embedded in the AOT shared library.
            self.graph_module._in_spec = pytree.tree_flatten(
                (tuple(graph_inputs), {}))[1]
            self.graph_module._out_spec = pytree.tree_flatten(
                (torch.empty(0), torch.empty(0)))[1]
            build_start = time.perf_counter()
            built_artifact = aot_compile(
                self.graph_module,
                list(graph_inputs),
                {"aot_inductor.output_path": artifact})
            torch.cuda.synchronize()
            _COMPILE_STATS["aot_build_seconds"] = (
                time.perf_counter() - build_start)
            if os.path.realpath(built_artifact) != os.path.realpath(artifact):
                raise RuntimeError(
                    "AOTInductor ignored the requested output path: "
                    f"requested={artifact}, built={built_artifact}")
            if not os.path.isfile(artifact):
                raise RuntimeError(
                    f"AOTInductor did not create the artifact: {artifact}")
            # CUDA kernels are emitted as sibling .cubin files and the wrapper
            # records their paths. Compile directly into the deployment
            # directory; copying only the .so would make the package invalid.
            temporary_metadata = f"{metadata_path}.{os.getpid()}.tmp"
            with open(temporary_metadata, "w", encoding="utf-8") as handle:
                json.dump(expected_signature, handle, indent=2, sort_keys=True)
            os.replace(temporary_metadata, metadata_path)
            print(
                ">>> AOTInductor artifact built: "
                f"seconds={_COMPILE_STATS['aot_build_seconds']:.3f}, "
                f"path={artifact}")
        else:
            if not os.path.isfile(artifact) or not os.path.isfile(
                    metadata_path):
                raise FileNotFoundError(
                    "AOTInductor artifact or metadata is missing. Run once "
                    "with WMA_AOT_INDUCTOR_BUILD=1.")
            with open(metadata_path, "r", encoding="utf-8") as handle:
                actual_signature = json.load(handle)
            if actual_signature != expected_signature:
                differing_fields = {
                    key: {
                        "artifact": actual_signature.get(key),
                        "current": expected_signature.get(key),
                    }
                    for key in sorted(set(actual_signature) |
                                      set(expected_signature))
                    if actual_signature.get(key) != expected_signature.get(key)
                }
                raise RuntimeError(
                    "AOTInductor artifact signature does not match the "
                    "current graph, inputs, PyTorch, CUDA, or math mode: "
                    f"{differing_fields}")

        load_start = time.perf_counter()
        # torch._export.aot_load in PyTorch 2.3 does not pass cubin_dir to the
        # CUDA runner.  Generated wrappers would therefore retain build-host
        # absolute paths.  Supply the package directory explicitly so the
        # .so and its sibling .cubin files remain relocatable as one unit.
        from torch.export._tree_utils import reorder_kwargs
        from torch.utils import _pytree as pytree

        device = str(graph_inputs[0].device)
        runner = torch._C._aoti.AOTIModelContainerRunnerCuda(
            artifact, 1, device, os.path.dirname(artifact))

        def loaded(*args, **kwargs):
            call_spec = runner.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = pytree.tree_flatten(
                (args, reorder_kwargs(kwargs, in_spec)))[0]
            flat_outputs = runner.run(flat_inputs)
            return pytree.tree_unflatten(flat_outputs, out_spec)

        _COMPILE_STATS["aot_load_seconds"] = time.perf_counter() - load_start
        _COMPILE_STATS["aot_artifact_mib"] = (
            os.path.getsize(artifact) / (1024**2))
        print(
            ">>> AOTInductor precompiled graph loaded: "
            f"seconds={_COMPILE_STATS['aot_load_seconds']:.3f}, "
            f"size_mib={_COMPILE_STATS['aot_artifact_mib']:.3f}")
        return loaded

    def _save_graph_cache(self, input_signature) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        # Export installs a private tracer that cannot be reconstructed by the
        # PyTorch 2.3 GraphModule unpickler. The generic FX tracer is sufficient
        # because the cached module is already a pure-ATen graph.
        self.graph_module._tracer_cls = torch.fx.Tracer
        self.graph_module._tracer_extras = {}
        self.graph_module.graph._tracer_cls = torch.fx.Tracer
        self.graph_module.graph._tracer_extras = {}
        for node in self.graph_module.graph.nodes:
            node.meta.clear()
        cpu_constants = {
            name: value.detach().cpu()
            for name, value in self.constants.items()
        }
        payload = {
            "version": self._CACHE_VERSION,
            "torch_version": torch.__version__,
            "adapter": type(self.adapter).__name__,
            "input_signature": input_signature,
            "graph_module": self.graph_module,
            "input_specs": self.input_specs,
            "constants": cpu_constants,
        }
        temporary_path = f"{self.cache_path}.{os.getpid()}.tmp"
        save_start = time.perf_counter()
        torch.save(payload, temporary_path)
        os.replace(temporary_path, self.cache_path)
        _COMPILE_STATS["generic_graph_cache_save_seconds"] = (
            time.perf_counter() - save_start)
        _COMPILE_STATS["generic_graph_cache_mib"] = (
            os.path.getsize(self.cache_path) / (1024**2))

    def _phase_target(self, target: str, active_head: str) -> str:
        if active_head == "state" and target.startswith(self._ACTION_PREFIX):
            return self._STATE_PREFIX + target[len(self._ACTION_PREFIX):]
        return target

    def _build_static_layout(self, active_head: str):
        layout = []
        for kind_name, original_target in self.input_specs:
            kind = self.input_kind[kind_name]
            if kind == self.input_kind.USER_INPUT:
                layout.append(None)
                continue
            target = self._phase_target(original_target, active_head)
            if kind == self.input_kind.PARAMETER:
                value = self.adapter.get_parameter(target)
            elif kind == self.input_kind.BUFFER:
                value = self.adapter.get_buffer(target)
            elif kind == self.input_kind.CONSTANT_TENSOR:
                if (active_head == "state" and
                        original_target.startswith(self._ACTION_PREFIX)):
                    raise RuntimeError(
                        "Cannot remap an action-head exported tensor constant.")
                value = self.constants[original_target]
            else:
                raise RuntimeError(
                    f"Unsupported exported generic input kind: {kind}")
            layout.append(value)
        return tuple(layout)

    def inputs(self, active_head: str, user_inputs):
        if active_head not in ("action", "state"):
            raise ValueError(f"Unsupported generic active head: {active_head}")
        layout = (self.action_layout if active_head == "action" else
                  self.state_layout)
        user_iterator = iter(user_inputs)
        values = tuple(
            next(user_iterator) if value is None else value
            for value in layout)
        try:
            next(user_iterator)
        except StopIteration:
            return values
        raise RuntimeError("Too many user inputs for exported generic graph.")


def _validate_compiled_outputs(eager_outputs,
                               compiled_outputs,
                               stat_prefix="",
                               output_label="Compiled",
                               output_indices=None) -> None:
    output_names = ("video", "action", "state")
    if output_indices is None:
        output_indices = range(len(output_names))
    for index in output_indices:
        name = output_names[index]
        eager = eager_outputs[index]
        compiled = compiled_outputs[index]
        if eager.shape != compiled.shape or eager.dtype != compiled.dtype:
            raise RuntimeError(
                f"{output_label} {name} metadata mismatch: "
                f"eager={eager.shape}/{eager.dtype}, "
                f"compiled={compiled.shape}/{compiled.dtype}")
        if not torch.isfinite(compiled).all():
            raise RuntimeError(
                f"{output_label} {name} output contains NaN or Inf.")
        difference = (eager.float() - compiled.float()).abs()
        max_abs = difference.max().item()
        mean_abs = difference.mean().item()
        relative_l2 = (
            torch.linalg.vector_norm(difference) /
            torch.linalg.vector_norm(eager.float()).clamp_min(1e-12)).item()
        _COMPILE_STATS[f"{stat_prefix}{name}_max_abs"] = max_abs
        _COMPILE_STATS[f"{stat_prefix}{name}_mean_abs"] = mean_abs
        _COMPILE_STATS[f"{stat_prefix}{name}_relative_l2"] = relative_l2
        torch.testing.assert_close(compiled,
                                   eager,
                                   rtol=1e-2,
                                   atol=1e-2)
        if max_abs > 1e-2 or mean_abs > 1e-3 or relative_l2 > 2e-3:
            raise RuntimeError(
                f"{output_label} {name} error exceeds screening limits: "
                f"max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}, "
                f"relative_l2={relative_l2:.6g}")


def precast_inference_weights(diffusion_model: nn.Module) -> None:
    """Pre-cast autocast-eligible denoiser weights once before compilation."""
    eligible_types = (nn.Linear, nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d,
                      nn.Conv3d)
    seen = set()
    converted_tensors = 0
    converted_bytes = 0
    start = time.perf_counter()
    with torch.no_grad():
        for module in diffusion_model.modules():
            if not isinstance(module, eligible_types):
                continue
            for name in ("weight", "bias"):
                parameter = getattr(module, name, None)
                if (parameter is None or id(parameter) in seen or
                        parameter.dtype != torch.float32):
                    continue
                seen.add(id(parameter))
                converted_bytes += parameter.numel() * parameter.element_size()
                if name == "weight" and isinstance(module, nn.Conv2d):
                    converted = parameter.detach().to(
                        dtype=torch.float16,
                        memory_format=torch.channels_last)
                elif (name == "weight" and isinstance(module, nn.Conv3d)
                      and _CONV3D_CHANNELS_LAST):
                    converted = parameter.detach().to(
                        dtype=torch.float16,
                        memory_format=torch.channels_last_3d)
                else:
                    converted = parameter.detach().to(dtype=torch.float16)
                parameter.data = converted
                converted_tensors += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    _COMPILE_STATS["precast_seconds"] = elapsed
    _COMPILE_STATS["precast_tensors"] = converted_tensors
    _COMPILE_STATS["precast_source_gib"] = converted_bytes / (1024**3)
    print(
        ">>> Pre-cast denoiser Conv/Linear weights: "
        f"tensors={converted_tensors}, source_gib={converted_bytes / (1024**3):.3f}, "
        f"seconds={elapsed:.3f}")


def enable_conv3d_channels_last(diffusion_model: nn.Module) -> None:
    """Keep temporal Conv3d weights and inputs in channels-last-3d layout."""
    converted_weights = 0
    temporal_blocks = 0
    start = time.perf_counter()
    with torch.no_grad():
        for module in diffusion_model.modules():
            if isinstance(module, nn.Conv3d):
                module.weight.data = module.weight.detach().contiguous(
                    memory_format=torch.channels_last_3d)
                converted_weights += 1
            if hasattr(module, "use_channels_last_3d"):
                module.use_channels_last_3d = True
                temporal_blocks += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    _COMPILE_STATS["conv3d_channels_last_seconds"] = elapsed
    _COMPILE_STATS["conv3d_channels_last_weights"] = converted_weights
    _COMPILE_STATS["conv3d_channels_last_blocks"] = temporal_blocks
    print(
        ">>> Conv3d channels-last-3d enabled: "
        f"weights={converted_weights}, temporal_blocks={temporal_blocks}, "
        f"seconds={elapsed:.3f}")


def install_compiled_denoiser(model: nn.Module) -> None:
    """Compile the shared fixed-shape WMA denoiser and route inference to it."""
    if _AOT_INDUCTOR_BUILD and not _AOT_INDUCTOR:
        raise ValueError(
            "WMA_AOT_INDUCTOR_BUILD=1 requires WMA_AOT_INDUCTOR=1.")
    if _AOT_INDUCTOR:
        if not _TORCH_COMPILE:
            raise ValueError("WMA_AOT_INDUCTOR requires WMA_TORCH_COMPILE=1.")
        if _COMPILE_BACKEND != "inductor":
            raise ValueError(
                "WMA_AOT_INDUCTOR requires WMA_TORCH_COMPILE_BACKEND=inductor.")
        if not (_PHASE_HEAD_PRUNING and _GENERIC_PHASE_HEAD):
            raise ValueError(
                "WMA_AOT_INDUCTOR requires the single generic phase graph: "
                "set WMA_PHASE_HEAD_PRUNING=1 and WMA_GENERIC_PHASE_HEAD=1.")
    if not (_TORCH_COMPILE or _SDPA_ACTION_ATTENTION):
        return
    if model.model.conditioning_key != "hybrid":
        raise RuntimeError("Fixed WMA compilation requires hybrid conditioning.")

    diffusion_model = model.model.diffusion_model
    if _CONV3D_CHANNELS_LAST:
        enable_conv3d_channels_last(diffusion_model)
    if _INDUCTOR_PRECAST_WEIGHTS:
        if _MATH_MODE != "fp16":
            raise ValueError(
                "WMA_INDUCTOR_PRECAST_WEIGHTS requires WMA_MATH_MODE=fp16.")
        precast_inference_weights(diffusion_model)
    for module in diffusion_model.modules():
        if hasattr(module, "record_keypoints"):
            module.record_keypoints = not _TORCH_COMPILE
        if hasattr(module, "use_sdpa_action_attention"):
            module.use_sdpa_action_attention = _SDPA_ACTION_ATTENTION

    if _SDPA_ACTION_ATTENTION:
        print(">>> PyTorch SDPA enabled for masked agent-action attention")
    if not _TORCH_COMPILE:
        return

    import torch._dynamo

    torch._dynamo.reset()
    torch._dynamo.config.error_on_recompile = True

    if _COMPILE_BACKEND == "inductor":
        from torch._inductor import config as inductor_config
        inductor_config.fx_graph_cache = _COMPILE_FX_GRAPH_CACHE
        if _INDUCTOR_FUSION:
            if _INDUCTOR_MAX_FUSION_SIZE <= 0:
                raise ValueError(
                    "WMA_INDUCTOR_MAX_FUSION_SIZE must be positive.")
            inductor_config.aggressive_fusion = True
            inductor_config.max_fusion_size = _INDUCTOR_MAX_FUSION_SIZE
            inductor_config.epilogue_fusion = True
            inductor_config.epilogue_fusion_first = True
            inductor_config.permute_fusion = True
            print(
                ">>> Aggressive Inductor fusion enabled: "
                f"max_fusion_size={_INDUCTOR_MAX_FUSION_SIZE}, "
                "epilogue_fusion_first=True, permute_fusion=True")

    reference_denoiser = FixedHybridDenoiser(diffusion_model)
    adapter_cls = (CachedObservationHybridDenoiser
                   if _CACHE_OBSERVATION_ENCODER else FixedHybridDenoiser)
    if _CACHE_OBSERVATION_ENCODER:
        print(">>> Sampler-level action/state observation cache enabled")
    compile_options = {
        "backend": _COMPILE_BACKEND,
        "fullgraph": _COMPILE_FULLGRAPH,
        "dynamic": False,
    }
    if _COMPILE_BACKEND == "inductor":
        compile_options["mode"] = _COMPILE_MODE
    generic_adapter = None
    generic_runtime = None
    if _GENERIC_PHASE_HEAD:
        if not _PHASE_HEAD_PRUNING:
            raise ValueError(
                "WMA_GENERIC_PHASE_HEAD requires WMA_PHASE_HEAD_PRUNING=1.")
        validate_generic_head_compatibility(diffusion_model)
        generic_cls = (CachedGenericPhaseHybridDenoiser
                       if _CACHE_OBSERVATION_ENCODER else
                       GenericPhaseHybridDenoiser)
        generic_adapter = generic_cls(diffusion_model)
        eager_denoisers = {
            active_head: adapter_cls(diffusion_model, active_head=active_head)
            for active_head in ("action", "state")
        }
        compiled_denoisers = {}
        validation_pending = {
            active_head: _COMPILE_VALIDATE
            for active_head in eager_denoisers
        }
        print(
            ">>> Exported generic phase-head pruning enabled: "
            "one action-shaped graph, phase-selectable lifted weights")
    elif _PHASE_HEAD_PRUNING:
        eager_denoisers = {
            active_head: adapter_cls(diffusion_model, active_head=active_head)
            for active_head in ("action", "state")
        }
        compiled_denoisers = {
            active_head: torch.compile(adapter, **compile_options)
            for active_head, adapter in eager_denoisers.items()
        }
        validation_pending = {
            active_head: _COMPILE_VALIDATE
            for active_head in eager_denoisers
        }
        print(">>> Phase-specific action/state head pruning enabled")
    else:
        eager_denoisers = {"both": adapter_cls(diffusion_model)}
        compiled_denoisers = {
            "both": torch.compile(eager_denoisers["both"], **compile_options)
        }
        validation_pending = {"both": _COMPILE_VALIDATE}
    original_apply_model = model.apply_model

    def map_generic_outputs(outputs, active_head, x_action, x_state):
        video, head = outputs
        if active_head == "action":
            return video, head, x_state
        return video, x_action, head

    def compiled_apply_model(x_noisy, x_action_noisy, x_state_noisy,
                             timestep, condition, **kwargs):
        nonlocal generic_runtime
        if not isinstance(condition, dict):
            return original_apply_model(x_noisy, x_action_noisy,
                                        x_state_noisy, timestep, condition,
                                        **kwargs)
        try:
            active_head = (kwargs["active_head"]
                           if _PHASE_HEAD_PRUNING else "both")
            if active_head not in eager_denoisers:
                raise ValueError(
                    f"Missing compiled adapter for active head {active_head}.")
            # Dataset observations start as FP32, while autoregressive images
            # produced under FP16 autocast are FP16. Keep the raw observation
            # boundary stable so every rollout round reuses the same graph.
            observation_image = condition["c_crossattn_action"][0].float()
            observation_state = condition["c_crossattn_action"][1].float()
            reference_inputs = (
                x_noisy,
                x_action_noisy,
                x_state_noisy,
                timestep,
                condition["c_concat"][0],
                condition["c_crossattn"][0],
                observation_image,
                observation_state,
                kwargs["fs"],
            )
            if _CACHE_OBSERVATION_ENCODER:
                inputs = reference_inputs[:-1] + (
                    kwargs["action_obs_global_cond"],
                    kwargs["state_obs_global_cond"],
                    reference_inputs[-1],
                )
            else:
                inputs = reference_inputs

            if _GENERIC_PHASE_HEAD:
                x_head = (x_action_noisy if active_head == "action" else
                          x_state_noisy)
                generic_inputs = (
                    x_noisy,
                    x_head,
                    timestep,
                    condition["c_concat"][0],
                    condition["c_crossattn"][0],
                    observation_image,
                    observation_state,
                )
                if _CACHE_OBSERVATION_ENCODER:
                    selected_observation_cond = (
                        kwargs["action_obs_global_cond"]
                        if active_head == "action" else
                        kwargs["state_obs_global_cond"])
                    generic_inputs += (selected_observation_cond, )
                generic_inputs += (kwargs["fs"], )
        except (KeyError, IndexError, TypeError):
            return original_apply_model(x_noisy, x_action_noisy,
                                        x_state_noisy, timestep, condition,
                                        **kwargs)

        eager_denoiser = eager_denoisers[active_head]
        if _GENERIC_PHASE_HEAD:
            if generic_runtime is None:
                generic_runtime = ExportedGenericPhaseRuntime(
                    generic_adapter, generic_inputs, compile_options)
            graph_inputs = generic_runtime.inputs(active_head, generic_inputs)
            compiled_denoiser = generic_runtime.compiled
        else:
            compiled_denoiser = compiled_denoisers[active_head]
        if validation_pending[active_head]:
            torch.cuda.synchronize()
            eager_start = time.perf_counter()
            if _CACHE_OBSERVATION_ENCODER or _PHASE_HEAD_PRUNING:
                reference_outputs = reference_denoiser(*reference_inputs)
                eager_outputs = eager_denoiser(*inputs)
                output_indices = ((0, 1) if active_head == "action" else
                                  (0, 2) if active_head == "state" else
                                  (0, 1, 2))
                _validate_compiled_outputs(
                    reference_outputs,
                    eager_outputs,
                    stat_prefix=(f"semantic_{active_head}_"),
                    output_label=f"Optimized-{active_head}",
                    output_indices=output_indices)
            else:
                eager_outputs = eager_denoiser(*inputs)
            torch.cuda.synchronize()
            _COMPILE_STATS[f"eager_probe_{active_head}_seconds"] = (
                time.perf_counter() - eager_start)

            if _GENERIC_PHASE_HEAD:
                exported_start = time.perf_counter()
                exported_outputs = generic_runtime.graph_module(*graph_inputs)
                torch.cuda.synchronize()
                _COMPILE_STATS[f"exported_eager_{active_head}_seconds"] = (
                    time.perf_counter() - exported_start)
                exported_outputs = map_generic_outputs(
                    exported_outputs, active_head, x_action_noisy,
                    x_state_noisy)
                _validate_compiled_outputs(
                    eager_outputs,
                    exported_outputs,
                    stat_prefix=f"exported_eager_{active_head}_",
                    output_label=f"Exported-eager-{active_head}",
                    output_indices=output_indices)

            compile_start = time.perf_counter()
            compiled_outputs = compiled_denoiser(
                *(graph_inputs if _GENERIC_PHASE_HEAD else inputs))
            torch.cuda.synchronize()
            first_call_name = (
                f"generic_first_{active_head}_seconds"
                if _GENERIC_PHASE_HEAD else
                f"cold_compile_{active_head}_seconds")
            _COMPILE_STATS[first_call_name] = (
                time.perf_counter() - compile_start)
            if _GENERIC_PHASE_HEAD:
                compiled_outputs = map_generic_outputs(
                    compiled_outputs, active_head, x_action_noisy,
                    x_state_noisy)
            _validate_compiled_outputs(
                eager_outputs,
                compiled_outputs,
                stat_prefix=f"compiled_{active_head}_")
            validation_pending[active_head] = False
            return compiled_outputs

        if _GENERIC_PHASE_HEAD:
            return map_generic_outputs(
                compiled_denoiser(*graph_inputs), active_head,
                x_action_noisy, x_state_noisy)
        return compiled_denoiser(*inputs)

    model.apply_model = compiled_apply_model
    print(
        ">>> torch.compile fixed hybrid denoiser enabled: "
        f"backend={_COMPILE_BACKEND}, mode={_COMPILE_MODE}, "
        f"fullgraph={_COMPILE_FULLGRAPH}, dynamic=False, "
        f"fx_graph_cache={_COMPILE_FX_GRAPH_CACHE}")


def print_compile_profile() -> None:
    if not _TORCH_COMPILE:
        return
    print(">>> torch.compile summary")
    for name, value in sorted(_COMPILE_STATS.items()):
        print(f">>> COMPILE {name}: {value}")
    try:
        from torch._dynamo.utils import counters
        for category in ("frames", "stats", "graph_break", "inductor"):
            if counters.get(category):
                values = ", ".join(
                    f"{key}={value}"
                    for key, value in counters[category].items())
                print(f">>> COMPILE_COUNTER {category}: {values}")
    except Exception as exc:
        print(f">>> COMPILE_COUNTER unavailable: {exc}")


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Args:
        module (nn.Module): The model whose device is to be inferred.

    Returns:
        torch.device: The device of the model's parameters.
    """
    return next(iter(module.parameters())).device


def write_video(video_path: str, stacked_frames: list, fps: int) -> None:
    """Save a list of frames to a video file.

    Args:
        video_path (str): Output path for the video.
        stacked_frames (list): List of image frames.
        fps (int): Frames per second for the video.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                "pkg_resources is deprecated as an API",
                                category=DeprecationWarning)
        imageio.mimsave(video_path, stacked_frames, fps=fps)


def get_filelist(data_dir: str, postfixes: list[str]) -> list[str]:
    """Return sorted list of files in a directory matching specified postfixes.

    Args:
        data_dir (str): Directory path to search in.
        postfixes (list[str]): List of file extensions to match.

    Returns:
        list[str]: Sorted list of file paths.
    """
    patterns = [
        os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes
    ]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list


def _nested_meta_tensor_paths(value, prefix):
    """Yield meta tensors held outside normal module parameter/buffer slots."""
    if isinstance(value, torch.Tensor):
        if value.is_meta:
            yield prefix
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _nested_meta_tensor_paths(item, f"{prefix}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _nested_meta_tensor_paths(item, f"{prefix}[{index}]")


def _audit_materialized_model(model: nn.Module) -> None:
    """Fail before CUDA transfer if meta initialization left unusable tensors."""
    meta_paths = []
    meta_paths.extend(
        f"parameter:{name}" for name, value in model.named_parameters()
        if value.is_meta)
    meta_paths.extend(
        f"buffer:{name}" for name, value in model.named_buffers()
        if value.is_meta)

    # A few inference tensors are intentionally ordinary attributes rather
    # than parameters/buffers. Audit direct tensor containers without walking
    # arbitrary third-party Python object graphs.
    for module_name, module in model.named_modules():
        prefix = module_name or "<root>"
        for attribute, value in vars(module).items():
            if attribute in ("_parameters", "_buffers", "_modules"):
                continue
            if isinstance(value, (torch.Tensor, dict, list, tuple)):
                meta_paths.extend(
                    _nested_meta_tensor_paths(
                        value, f"attribute:{prefix}.{attribute}"))

    for scheduler_name in (
            "dp_noise_scheduler_action", "dp_noise_scheduler_state"):
        scheduler = getattr(model, scheduler_name, None)
        if scheduler is None:
            continue
        for attribute, value in vars(scheduler).items():
            if isinstance(value, (torch.Tensor, dict, list, tuple)):
                meta_paths.extend(
                    _nested_meta_tensor_paths(
                        value, f"scheduler:{scheduler_name}.{attribute}"))

    if meta_paths:
        preview = ", ".join(meta_paths[:12])
        raise RuntimeError(
            "WMA_META_INIT left tensors on the meta device: "
            f"count={len(meta_paths)}, first={preview}")


def materialize_meta_runtime_buffers(model: nn.Module) -> None:
    """Restore non-persistent buffers omitted by checkpoint assignment."""
    before = {
        name for name, value in model.named_buffers() if value.is_meta
    }
    calls = 0
    for module in model.modules():
        materialize = getattr(module, "materialize_runtime_buffers", None)
        if callable(materialize):
            materialize(device=torch.device("cpu"))
            calls += 1
    _audit_materialized_model(model)
    after = {
        name for name, value in model.named_buffers() if value.is_meta
    }
    restored = len(before - after)
    print(
        ">>> Meta runtime buffers materialized: "
        f"buffers={restored}, handlers={calls}")


def load_model_checkpoint(model: nn.Module, ckpt: str) -> nn.Module:
    """Load model weights from checkpoint file.

    Args:
        model (nn.Module): Model instance.
        ckpt (str): Path to the checkpoint file.

    Returns:
        nn.Module: Model with loaded weights.
    """
    load_options = {"map_location": "cpu"}
    if _CHECKPOINT_MMAP:
        # weights_only avoids arbitrary checkpoint object construction and is
        # required for the predictable tensor-only mmap inference path.
        load_options.update({"weights_only": True, "mmap": True})
    with profile_startup_stage("checkpoint_deserialize"):
        checkpoint = torch.load(ckpt, **load_options)

    with profile_startup_stage("checkpoint_prepare"):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = OrderedDict(
                (key[16:], value)
                for key, value in checkpoint["module"].items())

        if _SKIP_INFERENCE_EMA:
            before = len(state_dict)
            state_dict = OrderedDict(
                (key, value) for key, value in state_dict.items()
                if not key.startswith("dp_ema_model."))
            removed = before - len(state_dict)
            if removed == 0:
                raise RuntimeError(
                    "WMA_SKIP_INFERENCE_EMA was requested, but the checkpoint "
                    "contains no dp_ema_model.* weights.")
            print(f">>> Skipped {removed} unused inference EMA tensors")

        if any("framestride_embed" in key for key in state_dict):
            renamed = OrderedDict()
            for key, value in state_dict.items():
                renamed[key.replace("framestride_embed", "fps_embedding")] = value
            state_dict = renamed

    with profile_startup_stage("checkpoint_load_state_dict"):
        # strict=True remains intentional: optional filtering is limited to the
        # inference-only EMA prefix, so all active model weights are verified.
        model.load_state_dict(state_dict,
                              strict=True,
                              assign=_CHECKPOINT_ASSIGN)
    if _META_INIT:
        with profile_startup_stage("meta_runtime_materialize"):
            materialize_meta_runtime_buffers(model)
    print('>>> model checkpoint loaded.')
    return model


def is_inferenced(save_dir: str, filename: str) -> bool:
    """Check if a given filename has already been processed and saved.

    Args:
        save_dir (str): Directory where results are saved.
        filename (str): Name of the file to check.

    Returns:
        bool: True if processed file exists, False otherwise.
    """
    video_file = os.path.join(save_dir, "samples_separate",
                              f"{filename[:-4]}_sample0.mp4")
    return os.path.exists(video_file)


def save_results(video: Tensor, filename: str, fps: int = 8) -> None:
    """Save video tensor to file using torchvision.

    Args:
        video (Tensor): Tensor of shape (B, C, T, H, W).
        filename (str): Output file path.
        fps (int, optional): Frames per second. Defaults to 8.
    """
    video = video.detach().cpu()
    video = torch.clamp(video.float(), -1., 1.)
    n = video.shape[0]
    video = video.permute(2, 0, 1, 3, 4)

    frame_grids = [
        torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0)
        for framesheet in video
    ]
    grid = torch.stack(frame_grids, dim=0)
    grid = (grid + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
    torchvision.io.write_video(filename,
                               grid,
                               fps=fps,
                               video_codec='h264',
                               options={'crf': '10'})


def prepare_tensorboard_video(data: Tensor) -> Tensor:
    """Prepare the same video grid as log_to_tensorboard on the caller GPU."""
    if not isinstance(data, torch.Tensor) or data.dim() != 5:
        raise ValueError("TensorBoard video input must be a 5-D tensor.")
    video = data.detach()
    n = video.shape[0]
    video = video.permute(2, 0, 1, 3, 4)
    frame_grids = [
        torchvision.utils.make_grid(framesheet,
                                    nrow=int(n),
                                    padding=0) for framesheet in video
    ]
    grid = torch.stack(frame_grids, dim=0)
    grid = (grid + 1.0) / 2.0
    return grid.unsqueeze(dim=0).cpu()


def write_iteration_outputs(writer, dm_tensorboard, dm_tag, wm_tensorboard,
                            wm_tag, dm_video, dm_filename, wm_video,
                            wm_filename, fps):
    """Write one iteration's outputs in the original deterministic order."""
    start = time.perf_counter()
    writer.add_video(dm_tag, dm_tensorboard, fps=fps)
    writer.add_video(wm_tag, wm_tensorboard, fps=fps)
    save_results(dm_video, dm_filename, fps=fps)
    save_results(wm_video, wm_filename, fps=fps)
    return time.perf_counter() - start


def write_final_outputs(writer, full_video, sample_tag, filename, fps):
    """Write the final TensorBoard event and MP4."""
    start = time.perf_counter()
    log_to_tensorboard(writer, full_video, sample_tag, fps=fps)
    save_results(full_video, filename, fps=fps)
    return time.perf_counter() - start


class AsyncOutputPipeline:
    """Bounded single-worker output pipeline with exception propagation."""

    def __init__(self, writer, max_pending=2):
        if max_pending <= 0:
            raise ValueError("Async output max_pending must be positive.")
        self.writer = writer
        self.max_pending = max_pending
        self.executor = ThreadPoolExecutor(max_workers=1,
                                           thread_name_prefix="wma-output")
        self.pending = deque()
        self.worker_seconds = 0.0
        self.wait_seconds = 0.0
        self.completed = 0

    def _collect_one(self):
        future = self.pending.popleft()
        wait_start = time.perf_counter()
        self.worker_seconds += future.result()
        self.wait_seconds += time.perf_counter() - wait_start
        self.completed += 1

    def submit(self, function, *args):
        while len(self.pending) >= self.max_pending:
            self._collect_one()
        self.pending.append(self.executor.submit(function, *args))

    def close(self):
        while self.pending:
            self._collect_one()
        self.executor.shutdown(wait=True)
        self.writer.flush()
        self.writer.close()
        print(
            ">>> Async output summary: "
            f"tasks={self.completed}, worker_total={self.worker_seconds:.3f}s, "
            f"main_wait={self.wait_seconds:.3f}s")


def get_init_frame_path(data_dir: str, sample: dict) -> str:
    """Construct the init_frame path from directory and sample metadata.

    Args:
        data_dir (str): Base directory containing videos.
        sample (dict): Dictionary containing 'data_dir' and 'videoid'.

    Returns:
        str: Full path to the video file.
    """
    rel_video_fp = os.path.join(sample['data_dir'],
                                str(sample['videoid']) + '.png')
    full_image_fp = os.path.join(data_dir, 'images', rel_video_fp)
    return full_image_fp


def get_transition_path(data_dir: str, sample: dict) -> str:
    """Construct the full transition file path from directory and sample metadata.

    Args:
        data_dir (str): Base directory containing transition files.
        sample (dict): Dictionary containing 'data_dir' and 'videoid'.

    Returns:
        str: Full path to the HDF5 transition file.
    """
    rel_transition_fp = os.path.join(sample['data_dir'],
                                     str(sample['videoid']) + '.h5')
    full_transition_fp = os.path.join(data_dir, 'transitions',
                                      rel_transition_fp)
    return full_transition_fp


def prepare_init_input(start_idx: int,
                       init_frame_path: str,
                       transition_dict: dict[str, torch.Tensor],
                       frame_stride: int,
                       wma_data,
                       video_length: int = 16,
                       n_obs_steps: int = 2) -> dict[str, Tensor]:
    """
    Extracts a structured sample from a video sequence including frames, states, and actions,
    along with properly padded observations and pre-processed tensors for model input.

    Args:
        start_idx (int): Starting frame index for the current clip.
        video: decord video instance.
        transition_dict (Dict[str, Tensor]): Dictionary containing tensors for 'action', 
                                             'observation.state', 'action_type', 'state_type'.
        frame_stride (int): Temporal stride between sampled frames.
        wma_data: Object that holds configuration and utility functions like normalization, 
                transformation, and resolution info.
        video_length (int, optional): Number of frames to sample from the video. Default is 16.
        n_obs_steps (int, optional): Number of historical steps for observations. Default is 2.
    """

    indices = [start_idx + frame_stride * i for i in range(video_length)]
    init_frame = Image.open(init_frame_path).convert('RGB')
    init_frame = torch.tensor(np.array(init_frame)).unsqueeze(0).permute(
        3, 0, 1, 2).float()

    if start_idx < n_obs_steps - 1:
        state_indices = list(range(0, start_idx + 1))
        states = transition_dict['observation.state'][state_indices, :]
        num_padding = n_obs_steps - 1 - start_idx
        first_slice = states[0:1, :]  # (t, d)
        padding = first_slice.repeat(num_padding, 1)
        states = torch.cat((padding, states), dim=0)
    else:
        state_indices = list(range(start_idx - n_obs_steps + 1, start_idx + 1))
        states = transition_dict['observation.state'][state_indices, :]

    actions = transition_dict['action'][indices, :]

    ori_state_dim = states.shape[-1]
    ori_action_dim = actions.shape[-1]

    frames_action_state_dict = {
        'action': actions,
        'observation.state': states,
    }
    frames_action_state_dict = wma_data.normalizer(frames_action_state_dict)
    frames_action_state_dict = wma_data.get_uni_vec(
        frames_action_state_dict,
        transition_dict['action_type'],
        transition_dict['state_type'],
    )

    if wma_data.spatial_transform is not None:
        init_frame = wma_data.spatial_transform(init_frame)
    init_frame = (init_frame / 255 - 0.5) * 2

    data = {
        'observation.image': init_frame,
    }
    data.update(frames_action_state_dict)
    return data, ori_state_dim, ori_action_dim


def get_latent_z(model, videos: Tensor) -> Tensor:
    """
    Extracts latent features from a video batch using the model's first-stage encoder.

    Args:
        model: the world model.
        videos (Tensor): Input videos of shape [B, C, T, H, W].

    Returns:
        Tensor: Latent video tensor of shape [B, C, T, H, W].
    """
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def preprocess_observation(
        model, observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # Map to expected inputs for the policy
    return_observations = {}

    if isinstance(observations["pixels"], dict):
        imgs = {
            f"observation.images.{key}": img
            for key, img in observations["pixels"].items()
        }
    else:
        imgs = {"observation.images.top": observations["pixels"]}

    for imgkey, img in imgs.items():
        img = torch.from_numpy(img)

        # Sanity check that images are channel last
        _, h, w, c = img.shape
        assert c < h and c < w, f"expect channel first images, but instead {img.shape}"

        # Sanity check that images are uint8
        assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

        # Convert to channel first of type float32 in range [0,1]
        img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
        img = img.type(torch.float32)

        return_observations[imgkey] = img

    return_observations["observation.state"] = torch.from_numpy(
        observations["agent_pos"]).float()
    return_observations['observation.state'] = model.normalize_inputs({
        'observation.state':
        return_observations['observation.state'].to(model.device)
    })['observation.state']

    return return_observations


def image_guided_synthesis_sim_mode(
        model: torch.nn.Module,
        prompts: list[str],
        observation: dict,
        noise_shape: tuple[int, int, int, int, int],
        action_cond_step: int = 16,
        n_samples: int = 1,
        ddim_steps: int = 50,
        ddim_eta: float = 1.0,
        unconditional_guidance_scale: float = 1.0,
        fs: int | None = None,
        text_input: bool = True,
        timestep_spacing: str = 'uniform',
        guidance_rescale: float = 0.0,
        sim_mode: bool = True,
        decode_video: bool = True,
        **kwargs) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """
    Performs image-guided video generation in a simulation-style mode with optional multimodal guidance (image, state, action, text).

    Args:
        model (torch.nn.Module): The diffusion-based generative model with multimodal conditioning.
        prompts (list[str]): A list of textual prompts to guide the synthesis process.
        observation (dict): A dictionary containing observed inputs including:
            - 'observation.images.top': Tensor of shape [B, O, C, H, W] (top-down images)
            - 'observation.state': Tensor of shape [B, O, D] (state vector)
            - 'action': Tensor of shape [B, T, D] (action sequence)
        noise_shape (tuple[int, int, int, int, int]): Shape of the latent variable to generate, 
            typically (B, C, T, H, W).
        action_cond_step (int): Number of time steps where action conditioning is applied. Default is 16.
        n_samples (int): Number of samples to generate (unused here, always generates 1). Default is 1.
        ddim_steps (int): Number of DDIM sampling steps. Default is 50.
        ddim_eta (float): DDIM eta parameter controlling the stochasticity. Default is 1.0.
        unconditional_guidance_scale (float): Scale for classifier-free guidance. If 1.0, guidance is off.
        fs (int | None): Frame index to condition on, broadcasted across the batch if specified. Default is None.
        text_input (bool): Whether to use text prompt as conditioning. If False, uses empty strings. Default is True.
        timestep_spacing (str): Timestep sampling method in DDIM sampler. Typically "uniform" or "linspace".
        guidance_rescale (float): Guidance rescaling factor to mitigate overexposure from classifier-free guidance.
        sim_mode (bool): Whether to perform world-model interaction or decision-making using the world-model.
        decode_video (bool): Whether to decode the sampled video latent to pixel
            space. Action-only inference can disable this when its video output
            is not consumed.
        **kwargs: Additional arguments passed to the DDIM sampler.

    Returns:
        batch_variants (torch.Tensor | None): Predicted pixel-space video frames
            [B, C, T, H, W], or ``None`` when ``decode_video`` is disabled.
        actions (torch.Tensor): Predicted action sequences [B, T, D] from diffusion decoding.
        states (torch.Tensor): Predicted state sequences [B, T, D] from diffusion decoding.
    """
    b, _, t, _, _ = noise_shape
    ddim_sampler = DDIMSampler(model)
    batch_size = noise_shape[0]

    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    with profile_stage("conditioning"):
        img = observation['observation.images.top'].permute(0, 2, 1, 3, 4)
        cond_img = rearrange(img, 'b o c h w -> (b o) c h w')[-1:]
        cond_img_emb = model.embedder(cond_img)
        cond_img_emb = model.image_proj_model(cond_img_emb)

        if model.model.conditioning_key == 'hybrid':
            z = get_latent_z(model, img.permute(0, 2, 1, 3, 4))
            img_cat_cond = z[:, :, -1:, :, :]
            img_cat_cond = repeat(img_cat_cond,
                                  'b c t h w -> b c (repeat t) h w',
                                  repeat=noise_shape[2])
            cond = {"c_concat": [img_cat_cond]}

        if not text_input:
            prompts = [""] * batch_size
        cond_ins_emb = model.get_learned_conditioning(prompts)

        cond_state_emb = model.state_projector(observation['observation.state'])
        cond_state_emb = cond_state_emb + model.agent_state_pos_emb

        cond_action_emb = model.action_projector(observation['action'])
        cond_action_emb = cond_action_emb + model.agent_action_pos_emb

        if not sim_mode:
            cond_action_emb = torch.zeros_like(cond_action_emb)

        cond["c_crossattn"] = [
            torch.cat(
                [cond_state_emb, cond_action_emb, cond_ins_emb, cond_img_emb],
                dim=1)
        ]
        cond["c_crossattn_action"] = [
            observation['observation.images.top'][:, :,
                                                  -model.n_obs_steps_acting:],
            observation['observation.state'][:, -model.n_obs_steps_acting:],
            sim_mode,
            False,
        ]

    if _PHASE_HEAD_PRUNING:
        kwargs["active_head"] = "state" if sim_mode else "action"

    if _CACHE_OBSERVATION_ENCODER:
        with profile_stage("observation_encoding"), torch.no_grad():
            diffusion_model = model.model.diffusion_model
            observation_image = cond["c_crossattn_action"][0].float()
            observation_state = cond["c_crossattn_action"][1].float()
            observation_cond = (observation_image, observation_state)
            empty_global_cond = observation_image.new_empty((0, ))
            if not _PHASE_HEAD_PRUNING or kwargs["active_head"] == "action":
                kwargs["action_obs_global_cond"] = (
                    diffusion_model.action_unet.encode_observation(
                        observation_cond,
                        batch_size=batch_size,
                        horizon=diffusion_model.action_unet.horizon))
            else:
                kwargs["action_obs_global_cond"] = empty_global_cond
            if not _PHASE_HEAD_PRUNING or kwargs["active_head"] == "state":
                kwargs["state_obs_global_cond"] = (
                    diffusion_model.state_unet.encode_observation(
                        observation_cond,
                        batch_size=batch_size,
                        horizon=diffusion_model.state_unet.horizon))
            else:
                kwargs["state_obs_global_cond"] = empty_global_cond

    uc = None
    kwargs.update({"unconditional_conditioning_img_nonetext": None})
    cond_mask = None
    cond_z0 = None
    if ddim_sampler is not None:
        with profile_stage("ddim_sampling"):
            samples, actions, states, intermedia = ddim_sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=batch_size,
                shape=noise_shape[1:],
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                cfg_img=None,
                mask=cond_mask,
                x0=cond_z0,
                fs=fs,
                timestep_spacing=timestep_spacing,
                guidance_rescale=guidance_rescale,
                **kwargs)

        # Reconstruct from latent only when the caller consumes pixel-space video.
        batch_variants = None
        if decode_video:
            with profile_stage("vae_decode"):
                batch_variants = model.decode_first_stage(samples)

    return batch_variants, actions, states


def run_inference(args: argparse.Namespace, gpu_num: int, gpu_no: int) -> None:
    """
    Run inference pipeline on prompts and image inputs.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        gpu_num (int): Number of GPUs.
        gpu_no (int): Index of the current GPU.

    Returns:
        None
    """
    if _MATH_MODE == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print(">>> TF32 matmul/convolution enabled")
    elif _MATH_MODE in ("bf16", "fp16"):
        print(f">>> {_MATH_MODE.upper()} autocast enabled")
    elif _MATH_MODE != "fp32":
        raise ValueError(f"Unsupported WMA_MATH_MODE for this stage: {_MATH_MODE}")

    if _FP16_FULL_REDUCTION:
        if _MATH_MODE != "fp16":
            raise ValueError(
                "WMA_FP16_FULL_REDUCTION requires WMA_MATH_MODE=fp16.")
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        print(">>> FP16 matmul reduced-precision reduction disabled")

    if _META_INIT:
        required = {
            "WMA_CHECKPOINT_MMAP": _CHECKPOINT_MMAP,
            "WMA_CHECKPOINT_ASSIGN": _CHECKPOINT_ASSIGN,
            "WMA_SKIP_OPENCLIP_PRETRAINED": _SKIP_OPENCLIP_PRETRAINED,
            "WMA_SKIP_INFERENCE_EMA": _SKIP_INFERENCE_EMA,
        }
        missing = [name for name, enabled in required.items() if not enabled]
        if missing:
            raise ValueError(
                "WMA_META_INIT requires the checkpoint-only inference path: "
                f"missing={missing}")
        print(">>> Meta-device architecture initialization enabled")

    action_ddim_steps = (args.ddim_steps if args.action_ddim_steps is None else
                         args.action_ddim_steps)
    video_ddim_steps = (args.ddim_steps if args.video_ddim_steps is None else
                        args.video_ddim_steps)
    if action_ddim_steps <= 0 or video_ddim_steps <= 0:
        raise ValueError("Action and video DDIM steps must both be positive.")
    print(
        f">>> DDIM steps: action={action_ddim_steps}, video={video_ddim_steps}"
    )

    # The challenge consumes only the final concatenated MP4. Keep the richer
    # per-iteration diagnostics available behind the default-off switch.
    os.makedirs(args.savedir + '/inference', exist_ok=True)
    writer = None
    output_pipeline = None
    if _FINAL_ONLY_OUTPUT:
        if args.n_iter <= 0:
            raise ValueError("WMA_FINAL_ONLY_OUTPUT requires --n_iter > 0.")
        if not 0 < args.exe_steps <= args.video_length:
            raise ValueError(
                "WMA_FINAL_ONLY_OUTPUT requires 0 < --exe_steps <= "
                "--video_length.")
        print(
            ">>> Final-only output enabled: skipping TensorBoard and "
            "per-iteration MP4 files")
        if _ASYNC_OUTPUT:
            print(
                ">>> Async per-iteration output ignored by final-only mode")
    else:
        log_dir = args.savedir + f"/tensorboard"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    if _ASYNC_OUTPUT and not _FINAL_ONLY_OUTPUT:
        output_pipeline = AsyncOutputPipeline(
            writer, max_pending=_ASYNC_OUTPUT_MAX_PENDING)
        print(
            ">>> Async output enabled: workers=1, "
            f"max_pending={_ASYNC_OUTPUT_MAX_PENDING}")

    # Load prompt
    csv_path = os.path.join(args.prompt_dir, f"{args.dataset}.csv")
    df = pd.read_csv(csv_path)

    # Load config
    config = OmegaConf.load(args.config)
    config['model']['params']['wma_config']['params'][
        'use_checkpoint'] = False
    if _SKIP_OPENCLIP_PRETRAINED:
        # The unified checkpoint contains both complete OpenCLIP towers. Avoid
        # loading the same external pretrained file twice only to overwrite it.
        config.model.params.cond_stage_config.params.version = None
        config.model.params.img_cond_stage_config.params.version = None
        print(">>> Skipping redundant OpenCLIP pretrained loads")
    if _SKIP_INFERENCE_EMA:
        # dp_ema_model is maintained by the training hook and is never read by
        # this inference path. The matching checkpoint prefix is filtered by
        # load_model_checkpoint while every active key stays strict-checked.
        config.model.params.dp_use_ema = False
        print(">>> Skipping unused inference action EMA model")
    if _CHECKPOINT_MMAP:
        print(">>> Checkpoint weights_only mmap enabled")
    if _CHECKPOINT_ASSIGN:
        print(">>> Checkpoint assign loading enabled")
    with profile_stage("model_init_and_checkpoint"):
        with profile_startup_stage("model_architecture_init"):
            init_context = (torch.device("meta") if _META_INIT else
                            contextlib.nullcontext())
            with init_context:
                model = instantiate_from_config(config.model)
        model.perframe_ae = args.perframe_ae and not _BATCH_VAE
        if _BATCH_VAE:
            print(">>> Batched VAE encode/decode enabled")
        assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, args.ckpt_path)
        model.eval()
    print(f'>>> Load pre-trained model ...')

    # Build unnomalizer
    logging.info("***** Configing Data *****")
    with profile_stage("data_setup"):
        data = instantiate_from_config(config.data)
        data.setup()
    print(">>> Dataset is successfully loaded ...")

    with profile_stage("model_to_cuda"):
        with profile_startup_stage("model_to_cuda", synchronize_cuda=True):
            model = model.cuda(gpu_no)
    print_startup_profile()
    install_compiled_denoiser(model)
    device = get_device_from_parameters(model)

    # Run over data
    assert (args.height % 16 == 0) and (
        args.width % 16
        == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"

    # Get latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'>>> Generate {n_frames} frames under each generation ...')
    noise_shape = [args.bs, channels, n_frames, h, w]

    # Start inference
    for idx in range(0, len(df)):
        sample = df.iloc[idx]

        # Got initial frame path
        init_frame_path = get_init_frame_path(args.prompt_dir, sample)
        ori_fps = float(sample['fps'])

        video_save_dir = args.savedir + f"/inference/sample_{sample['videoid']}"
        if not _FINAL_ONLY_OUTPUT:
            os.makedirs(video_save_dir, exist_ok=True)
            os.makedirs(video_save_dir + '/dm', exist_ok=True)
            os.makedirs(video_save_dir + '/wm', exist_ok=True)

        # Load transitions to get the initial state later
        transition_path = get_transition_path(args.prompt_dir, sample)
        with h5py.File(transition_path, 'r') as h5f:
            transition_dict = {}
            for key in h5f.keys():
                transition_dict[key] = torch.tensor(h5f[key][()])
            for key in h5f.attrs.keys():
                transition_dict[key] = h5f.attrs[key]

        # If many, test various frequence control and world-model generation
        for fs in args.frame_stride:

            if not _FINAL_ONLY_OUTPUT:
                # For saving imagens in policy
                sample_save_dir = f'{video_save_dir}/dm/{fs}'
                os.makedirs(sample_save_dir, exist_ok=True)
                # For saving environmental changes in world-model
                sample_save_dir = f'{video_save_dir}/wm/{fs}'
                os.makedirs(sample_save_dir, exist_ok=True)
            # Final-only stays on GPU and performs a single host transfer.
            wm_video = [] if not _FINAL_ONLY_OUTPUT else None
            full_video_gpu = None
            # Initialize observation queues
            cond_obs_queues = {
                "observation.images.top":
                deque(maxlen=model.n_obs_steps_imagen),
                "observation.state": deque(maxlen=model.n_obs_steps_imagen),
                "action": deque(maxlen=args.video_length),
            }
            # Obtain initial frame and state
            start_idx = 0
            model_input_fs = ori_fps // fs
            batch, ori_state_dim, ori_action_dim = prepare_init_input(
                start_idx,
                init_frame_path,
                transition_dict,
                fs,
                data.test_datasets[args.dataset],
                n_obs_steps=model.n_obs_steps_imagen)
            observation = {
                'observation.images.top':
                batch['observation.image'].permute(1, 0, 2,
                                                   3)[-1].unsqueeze(0),
                'observation.state':
                batch['observation.state'][-1].unsqueeze(0),
                'action':
                torch.zeros_like(batch['action'][-1]).unsqueeze(0)
            }
            observation = {
                key: observation[key].to(device, non_blocking=True)
                for key in observation
            }
            # Update observation queues
            cond_obs_queues = populate_queues(cond_obs_queues, observation)

            # Multi-round interaction with the world-model
            for itr in tqdm(range(args.n_iter)):

                # Get observation
                observation = {
                    'observation.images.top':
                    torch.stack(list(
                        cond_obs_queues['observation.images.top']),
                                dim=1).permute(0, 2, 1, 3, 4),
                    'observation.state':
                    torch.stack(list(cond_obs_queues['observation.state']),
                                dim=1),
                    'action':
                    torch.stack(list(cond_obs_queues['action']), dim=1),
                }
                observation = {
                    key: observation[key].to(device, non_blocking=True)
                    for key in observation
                }

                # Use world-model in policy to generate action
                print(f'>>> Step {itr}: generating actions ...')
                with profile_stage("action_generation"), precision_context():
                    pred_videos_0, pred_actions, _ = image_guided_synthesis_sim_mode(
                        model,
                        sample['instruction'],
                        observation,
                        noise_shape,
                        action_cond_step=args.exe_steps,
                        ddim_steps=action_ddim_steps,
                        ddim_eta=args.ddim_eta,
                        unconditional_guidance_scale=args.
                        unconditional_guidance_scale,
                        fs=model_input_fs,
                        timestep_spacing=args.timestep_spacing,
                        guidance_rescale=args.guidance_rescale,
                        sim_mode=False,
                        decode_video=not _FINAL_ONLY_OUTPUT)

                # Update future actions in the observation queues
                for idx in range(len(pred_actions[0])):
                    observation = {'action': pred_actions[0][idx:idx + 1]}
                    observation['action'][:, ori_action_dim:] = 0.0
                    cond_obs_queues = populate_queues(cond_obs_queues,
                                                      observation)

                # Collect data for interacting the world-model using the predicted actions
                observation = {
                    'observation.images.top':
                    torch.stack(list(
                        cond_obs_queues['observation.images.top']),
                                dim=1).permute(0, 2, 1, 3, 4),
                    'observation.state':
                    torch.stack(list(cond_obs_queues['observation.state']),
                                dim=1),
                    'action':
                    torch.stack(list(cond_obs_queues['action']), dim=1),
                }
                observation = {
                    key: observation[key].to(device, non_blocking=True)
                    for key in observation
                }

                # Interaction with the world-model
                print(f'>>> Step {itr}: interacting with world model ...')
                with profile_stage("world_generation"), precision_context():
                    pred_videos_1, _, pred_states = image_guided_synthesis_sim_mode(
                        model,
                        "",
                        observation,
                        noise_shape,
                        action_cond_step=args.exe_steps,
                        ddim_steps=video_ddim_steps,
                        ddim_eta=args.ddim_eta,
                        unconditional_guidance_scale=args.
                        unconditional_guidance_scale,
                        fs=model_input_fs,
                        text_input=False,
                        timestep_spacing=args.timestep_spacing,
                        guidance_rescale=args.guidance_rescale)

                for idx in range(args.exe_steps):
                    observation = {
                        'observation.images.top':
                        pred_videos_1[0][:, idx:idx + 1].permute(1, 0, 2, 3),
                        'observation.state':
                        torch.zeros_like(pred_states[0][idx:idx + 1]) if
                        args.zero_pred_state else pred_states[0][idx:idx + 1],
                        'action':
                        torch.zeros_like(pred_actions[0][-1:])
                    }
                    observation['observation.state'][:, ori_state_dim:] = 0.0
                    cond_obs_queues = populate_queues(cond_obs_queues,
                                                      observation)

                with profile_stage("iteration_output"):
                    if _FINAL_ONLY_OUTPUT:
                        if pred_videos_1.ndim != 5:
                            raise RuntimeError(
                                "World-model video must be 5-D in final-only "
                                f"mode, got shape={tuple(pred_videos_1.shape)}")
                        executed_video = pred_videos_1[:, :, :args.exe_steps]
                        if executed_video.shape[2] != args.exe_steps:
                            raise RuntimeError(
                                "World-model video is shorter than --exe_steps: "
                                f"shape={tuple(pred_videos_1.shape)}, "
                                f"exe_steps={args.exe_steps}")
                        if full_video_gpu is None:
                            full_shape = list(executed_video.shape)
                            full_shape[2] = args.n_iter * args.exe_steps
                            full_video_gpu = executed_video.new_empty(full_shape)
                        frame_start = itr * args.exe_steps
                        frame_end = frame_start + args.exe_steps
                        full_video_gpu[:, :, frame_start:frame_end].copy_(
                            executed_video)
                    else:
                        dm_tag = f"{args.dataset}-vid{sample['videoid']}-dm-fs-{fs}/itr-{itr}"
                        wm_tag = f"{args.dataset}-vid{sample['videoid']}-wd-fs-{fs}/itr-{itr}"
                        dm_file = f'{video_save_dir}/dm/{fs}/itr-{itr}.mp4'
                        wm_file = f'{video_save_dir}/wm/{fs}/itr-{itr}.mp4'
                        if output_pipeline is not None:
                            dm_tensorboard = prepare_tensorboard_video(
                                pred_videos_0)
                            wm_tensorboard = prepare_tensorboard_video(
                                pred_videos_1)
                            dm_video_cpu = pred_videos_0.detach().cpu()
                            wm_video_cpu = pred_videos_1.detach().cpu()
                            output_pipeline.submit(
                                write_iteration_outputs, writer,
                                dm_tensorboard, dm_tag, wm_tensorboard, wm_tag,
                                dm_video_cpu, dm_file, wm_video_cpu, wm_file,
                                args.save_fps)
                        else:
                            log_to_tensorboard(writer,
                                               pred_videos_0,
                                               dm_tag,
                                               fps=args.save_fps)
                            log_to_tensorboard(writer,
                                               pred_videos_1,
                                               wm_tag,
                                               fps=args.save_fps)
                            save_results(pred_videos_0.cpu(),
                                         dm_file,
                                         fps=args.save_fps)
                            save_results(pred_videos_1.cpu(),
                                         wm_file,
                                         fps=args.save_fps)

                print('>' * 24)
                # Collect the result of world-model interactions
                if not _FINAL_ONLY_OUTPUT:
                    if output_pipeline is not None:
                        wm_video.append(
                            wm_video_cpu[:, :, :args.exe_steps].clone())
                    else:
                        wm_video.append(
                            pred_videos_1[:, :, :args.exe_steps].cpu())

            with profile_stage("final_output"):
                if _FINAL_ONLY_OUTPUT:
                    if full_video_gpu is None:
                        raise RuntimeError(
                            "Final-only output buffer was not initialized.")
                    sample_full_video_file = os.path.join(
                        args.savedir, "inference",
                        f"{sample['videoid']}_full_fs{fs}.mp4")
                    save_results(full_video_gpu,
                                 sample_full_video_file,
                                 fps=args.save_fps)
                else:
                    full_video = torch.cat(wm_video, dim=2)
                    sample_tag = f"{args.dataset}-vid{sample['videoid']}-wd-fs-{fs}/full"
                    sample_full_video_file = f"{video_save_dir}/../{sample['videoid']}_full_fs{fs}.mp4"
                    if output_pipeline is not None:
                        output_pipeline.submit(write_final_outputs, writer,
                                               full_video, sample_tag,
                                               sample_full_video_file,
                                               args.save_fps)
                    else:
                        log_to_tensorboard(writer,
                                           full_video,
                                           sample_tag,
                                           fps=args.save_fps)
                        save_results(full_video,
                                     sample_full_video_file,
                                     fps=args.save_fps)

    with profile_stage("output_drain"):
        if output_pipeline is not None:
            output_pipeline.close()
        elif writer is not None:
            writer.flush()
            writer.close()
    print_stage_profile()
    print_compile_profile()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir",
                        type=str,
                        default=None,
                        help="Path to save the results.")
    parser.add_argument("--ckpt_path",
                        type=str,
                        default=None,
                        help="Path to the model checkpoint.")
    parser.add_argument("--config",
                        type=str,
                        help="Path to the model checkpoint.")
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=None,
        help="Directory containing videos and corresponding prompts.")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="the name of dataset to test")
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM steps. If non-positive, DDPM is used instead.")
    parser.add_argument(
        "--action_ddim_steps",
        type=int,
        default=None,
        help="Action-head DDIM steps. Defaults to --ddim_steps.")
    parser.add_argument(
        "--video_ddim_steps",
        type=int,
        default=None,
        help="Video/world-model DDIM steps. Defaults to --ddim_steps.")
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="Eta for DDIM sampling. Set to 0.0 for deterministic results.")
    parser.add_argument("--bs",
                        type=int,
                        default=1,
                        help="Batch size for inference. Must be 1.")
    parser.add_argument("--height",
                        type=int,
                        default=320,
                        help="Height of the generated images in pixels.")
    parser.add_argument("--width",
                        type=int,
                        default=512,
                        help="Width of the generated images in pixels.")
    parser.add_argument(
        "--frame_stride",
        type=int,
        nargs='+',
        required=True,
        help=
        "frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)"
    )
    parser.add_argument(
        "--unconditional_guidance_scale",
        type=float,
        default=1.0,
        help="Scale for classifier-free guidance during sampling.")
    parser.add_argument("--seed",
                        type=int,
                        default=123,
                        help="Random seed for reproducibility.")
    parser.add_argument("--video_length",
                        type=int,
                        default=16,
                        help="Number of frames in the generated video.")
    parser.add_argument("--num_generation",
                        type=int,
                        default=1,
                        help="seed for seed_everything")
    parser.add_argument(
        "--timestep_spacing",
        type=str,
        default="uniform",
        help=
        "Strategy for timestep scaling. See Table 2 in the paper: 'Common Diffusion Noise Schedules and Sample Steps are Flawed' (https://huggingface.co/papers/2305.08891)."
    )
    parser.add_argument(
        "--guidance_rescale",
        type=float,
        default=0.0,
        help=
        "Rescale factor for guidance as discussed in 'Common Diffusion Noise Schedules and Sample Steps are Flawed' (https://huggingface.co/papers/2305.08891)."
    )
    parser.add_argument(
        "--perframe_ae",
        action='store_true',
        default=False,
        help=
        "Use per-frame autoencoder decoding to reduce GPU memory usage. Recommended for models with resolutions like 576x1024."
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=16,
        help="num of samples per prompt",
    )
    parser.add_argument(
        "--exe_steps",
        type=int,
        default=16,
        help="num of samples to execute",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=40,
        help="num of iteration to interact with the world model",
    )
    parser.add_argument("--zero_pred_state",
                        action='store_true',
                        default=False,
                        help="not using the predicted states as comparison")
    parser.add_argument("--save_fps",
                        type=int,
                        default=8,
                        help="fps for the saving video")
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    seed = args.seed
    if seed < 0:
        seed = random.randint(0, 2**31)
    seed_everything(seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)
