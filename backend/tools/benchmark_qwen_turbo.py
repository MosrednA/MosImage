"""Isolated official Qwen 2.1 Turbo quantization evaluation; no app registration.

Run from the repository root with backend/.venv/Scripts/python.exe. Prepare,
encode and each render variant run separately to release GPU/RAM between stages.
The encoder is shared across transformer variants, so quantization is isolated.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image
import psutil
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'data/qwen-turbo-official-2026-10-10'
OUTPUT.mkdir(parents=True, exist_ok=True)
# No developer .env or live app database in the research process.
os.chdir(OUTPUT)
os.environ['MOSIMAGE_DATA_DIR'] = str(OUTPUT / 'app-data')
os.environ['MOSIMAGE_DB_PATH'] = str(OUTPUT / 'app-data/app.db')
os.environ['MOSIMAGE_IMAGE_DIR'] = str(OUTPUT / 'app-data/images')
sys.path[:0] = [str(ROOT / 'models/research-cache/qwen-benchmark-deps'), str(ROOT / 'backend')]
from app.config import settings
assert settings.data_dir==OUTPUT/'app-data'
# Historical fixture helpers require backend as cwd; settings are already isolated.
os.chdir(ROOT / 'backend')
from benchmark_model_comparison import (ARMS, Case, KleinRenderer, QWEN_REVISION,
                                       load_cases, qwen_condition_images, workspaces)
from benchmark_inpaint_quality import assert_api_unloaded
os.chdir(OUTPUT)

OFFICIAL = ROOT / 'models/research-cache/Qwen-Image-2.1-Turbo-official'
QUANT = ROOT / 'models/research-cache/Qwen-Image-2.1-Turbo-NVFP4'
GGUF = ROOT / 'models/research-cache/Qwen-Image-2.1-Turbo-GGUF'
SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
SEEDS = (7, 42)
FROZEN = ('duo-coaster', 'dj', 'sword-edit', 'garden', 'portrait', 'architecture', 'mug')
REVISIONS = {'official': 'd65dbc9a7e8f6b5479e33dee6030eaab2a906509',
             'nvfp4': 'cb5a4e68cb747ca8a753a9f15416c3aeea6c8b67',
             'gguf': '5d1f9f759ea877ae556cda88dbec958b62ad3809', 'shared_components': QWEN_REVISION}


def memory_budget(torch) -> tuple[int, int]:
    free,total = torch.cuda.mem_get_info()
    # WDDM's CUDA free figure excludes some other desktop allocations.
    # Use the lower physical/device observation, preserving desktop headroom.
    observed = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                                          text=True).strip().splitlines()[0])*1024**2
    return min(free,observed)-int(1.25*1024**3),total


class Measurement(BaseModel):
    model_config = ConfigDict(extra='forbid')
    variant: str
    case: str
    seed: int
    seconds: float | None = None
    peak_allocated_gib: float | None = None
    peak_reserved_gib: float | None = None
    peak_rss_gib: float | None = None
    min_available_ram_gib: float | None = None
    native_calls: int = 0
    protected_changes: int | None = None
    selected_changes: int | None = None
    alpha_min: int | None = None
    alpha_max: int | None = None
    encoding_seconds: float | None = None
    renderer_upload_seconds: float | None = None
    render_seconds: float | None = None
    output_size: tuple[int,int] | None = None
    error: str | None = None


class MemorySample:
    def __init__(self) -> None:
        self.peak_rss = 0
        self.min_available = psutil.virtual_memory().available
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        process = psutil.Process()
        while not self.stop.is_set():
            self.peak_rss = max(self.peak_rss, process.memory_info().rss)
            self.min_available = min(self.min_available, psutil.virtual_memory().available)
            self.stop.wait(0.1)

    def __enter__(self) -> MemorySample:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop.set()
        self.thread.join()


def prepare() -> None:
    original = ROOT / 'data/model-comparison-2026-10-09'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(original / 'assets', OUTPUT / 'assets', dirs_exist_ok=True)
    shutil.copyfile(original / 'assets.json', OUTPUT / 'assets.json')
    cases = [c for c in load_cases(original) if c.name in FROZEN]
    cases.extend([
        Case(name='typography', title='Exact Dutch lettering', category='Generation', mode='compose', size=(1024, 1024),
             klein_prompt='A premium photographic poster for a Dutch coffee shop. Large crisp heading at the top reads exactly "KOFFIE & KOEK". A small subtitle reads exactly "Elke dag vers". A ceramic cup of cappuccino and a croissant on a dark green cafe table, warm morning light. Elegant restrained composition, perfectly legible Dutch text.',
             qwen_prompt='A premium photographic poster for a Dutch coffee shop. Large crisp heading at the top reads exactly "KOFFIE & KOEK". A small subtitle reads exactly "Elke dag vers". A ceramic cup of cappuccino and a croissant on a dark green cafe table, warm morning light. Elegant restrained composition, perfectly legible Dutch text.',
             criteria=('Exact spelling of both lines', 'Good cup and croissant', 'Readable composition')),
        Case(name='transparent', title='Native transparent cutout', category='Generation', mode='compose', size=(1024, 1024),
             klein_prompt='A single small red ceramic espresso cup with a curved handle and a tiny saucer. Product photography, a three-quarter view, crisp edges and subtle ceramic texture. Isolated cutout on a transparent background, native RGBA PNG with transparent alpha outside the object. No text, no floor, no background.',
             qwen_prompt='A single small red ceramic espresso cup with a curved handle and a tiny saucer. Product photography, a three-quarter view, crisp edges and subtle ceramic texture. Isolated cutout on a transparent background, native RGBA PNG with transparent alpha outside the object. No text, no floor, no background.',
             criteria=('Actual alpha transparency', 'Single cup and saucer', 'Clean edges')),
    ])
    (OUTPUT / 'cases.json').write_text(json.dumps([c.model_dump() for c in cases], indent=2), encoding='utf-8')
    hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted((OUTPUT / 'assets').glob('*.png'))}
    (OUTPUT / 'asset-hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    print(f'Prepared {len(cases)} frozen cases, seeds {SEEDS}', flush=True)


def encode(encoder_variant: str, resolution: int) -> None:
    import torch
    from diffusers import QwenImage21Pipeline
    from diffusers.pipelines.qwenimage21 import pipeline_qwenimage21 as module
    from transformers import Qwen3VLModel

    assert_api_unloaded()
    cases = load_cases(OUTPUT)
    prepared = workspaces(OUTPUT, ARMS['qwen21-turbo'], cases)
    processor = make_processor()
    budget,total = memory_budget(torch)
    torch.cuda.set_per_process_memory_fraction(budget / total)
    started = time.perf_counter()
    # No language head: the pipeline only reads hidden states, never vocabulary logits.
    with MemorySample() as memory:
        if encoder_variant == 'nvfp4':
            encoder = load_nvfp4_encoder().to('cuda')
        else:
            encoder = Qwen3VLModel.from_pretrained(str(OFFICIAL / 'text_encoder'),
                         dtype=torch.bfloat16, device_map='auto', max_memory={0: int(budget - 2.5 * 1024**3), 'cpu': '12GiB'},
                         local_files_only=True)
    protocol = {'encoder': f'Qwen3-VL-8B {encoder_variant} backbone, no unused lm_head',
                'revision': REVISIONS['nvfp4' if encoder_variant=='nvfp4' else 'official'],
                'load_seconds': round(time.perf_counter()-started, 3), 'load_peak_rss_gib': round(memory.peak_rss/1024**3, 3),
                'device_map': {k: str(v) for k, v in getattr(encoder,'hf_device_map',{'': 'cuda'}).items()}, 'cases': []}
    class EncoderBackbone(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model
            self.config = model.config

        @property
        def dtype(self):
            return self.model.dtype

        @property
        def device(self):
            return self.model.device

        def forward(self, **kwargs):
            return self.model(**kwargs)

    pipe = QwenImage21Pipeline(scheduler=None, vae=None, text_encoder=EncoderBackbone(encoder), processor=processor, transformer=None)
    suffix = '' if resolution==1024 else f'-cond{resolution}'
    folder = OUTPUT / f'embeddings-{encoder_variant}{suffix}'
    folder.mkdir(exist_ok=True)
    with torch.inference_mode():
        for case in cases:
            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            images = qwen_condition_images(module, pipe.image_processor, prepared[case.name].images(), resolution)
            with MemorySample() as memory:
                values = pipe.encode_prompt(prompt=case.qwen_prompt, image=images or None, device=torch.device('cuda'))
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            torch.save(tuple(None if t is None else t.cpu() for t in values), folder / f'{case.name}.pt')
            protocol['cases'].append({'case': case.name, 'seconds': round(elapsed, 3),
                   'peak_allocated_gib': round(torch.cuda.max_memory_allocated()/1024**3, 3),
                   'peak_rss_gib': round(memory.peak_rss/1024**3, 3), 'min_available_ram_gib': round(memory.min_available/1024**3, 3)})
            protocol['conditioning_resolution'] = resolution
            (OUTPUT / f'encoding-{encoder_variant}{suffix}.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
            print(f'ENCODE {case.name}: {elapsed:.3f}s', flush=True)


def make_processor():
    from transformers import AutoImageProcessor, AutoVideoProcessor, Qwen2TokenizerFast, Qwen3VLProcessor
    folder = str(OFFICIAL / 'processor')
    parts = dict(local_files_only=True)
    tokenizer = Qwen2TokenizerFast.from_pretrained(folder, **parts)
    return Qwen3VLProcessor(image_processor=AutoImageProcessor.from_pretrained(folder, **parts), tokenizer=tokenizer,
                           video_processor=AutoVideoProcessor.from_pretrained(folder, **parts), chat_template=tokenizer.chat_template)


def nvfp4_linear_class():
    import torch
    from app.models.nvfp4_runtime import Nvfp4Linear, kernel_api

    kernel = kernel_api()

    class DynamicNvfp4Linear(Nvfp4Linear):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.device.type != 'cuda' or self.packed.device != x.device:
                raise RuntimeError('Native NVFP4 requires CUDA inputs and weights')
            flat = x.reshape(-1, self.in_features).contiguous()
            # This checkpoint has calibrated weights, but no activation scales.
            # Cover the FP4 * FP8 representable range using a dynamic global scale.
            scale = (flat.float().abs().amax() / (6.0 * 448.0)).clamp_min(1e-12).reshape(1)
            qx, bx = kernel.quantize_nvfp4(flat, scale, pad_16x=True)
            y = kernel.scaled_mm_nvfp4(qx, self.packed, tensor_scale_a=scale, tensor_scale_b=self.scale,
                                      block_scale_a=bx, block_scale_b=self.blocks, out_dtype=x.dtype)
            self.calls += 1
            return y[:flat.shape[0], :self.out_features].reshape(*x.shape[:-1], self.out_features)

    return DynamicNvfp4Linear


def load_nvfp4():
    import torch
    from torch import nn
    from accelerate import init_empty_weights
    from diffusers import QwenImage21Transformer2DModel
    from safetensors import safe_open
    DynamicNvfp4Linear = nvfp4_linear_class()
    from app.models.nvfp4_runtime import kernel_api
    kernel = kernel_api()
    config = json.loads((OFFICIAL / 'transformer/config.json').read_text())
    with init_empty_weights():
        model = QwenImage21Transformer2DModel.from_config(config).to(dtype=torch.bfloat16)
    path = QUANT / 'diffusion_models/qwen_image_2.1_turbo_nvfp4.safetensors'
    with safe_open(str(path), framework='pt') as saved:
        consumed: set[str] = set()
        for key in saved.keys():
            if not key.endswith('.comfy_quant'):
                continue
            description = json.loads(bytes(saved.get_tensor(key).tolist()))
            if description != {'format': 'nvfp4'}:
                raise ValueError(f'Unsupported checkpoint recipe {key}: {description}')
            source = key.removesuffix('.comfy_quant')
            names = [source+s for s in ('.weight', '.weight_scale', '.weight_scale_2')]
            packed, blocks, scale = (saved.get_tensor(n) for n in names)
            consumed.update([key, *names])
            if source.endswith('img_mlp.gate_up'):
                targets = (source.replace('gate_up', 'gate_layer'), source.replace('gate_up', 'proj'))
                weights, scales = packed.chunk(2, dim=0), blocks.chunk(2, dim=0)
            else:
                targets, weights, scales = (source,), (packed,), (blocks,)
            for target, weight, block in zip(targets, weights, scales, strict=True):
                previous = model.get_submodule(target)
                if not isinstance(previous, nn.Linear) or previous.bias is not None:
                    raise ValueError(f'Unexpected native quantization target {target}')
                replacement = DynamicNvfp4Linear(weight, block, scale, torch.ones(1, dtype=torch.float32), kernel)
                if (replacement.in_features, replacement.out_features) != (previous.in_features, previous.out_features):
                    raise ValueError(f'Quantized dimensions mismatch: {target}')
                parent, leaf = target.rsplit('.', 1)
                setattr(model.get_submodule(parent), leaf, replacement)
        state = {k: saved.get_tensor(k) for k in saved.keys() if k not in consumed}
    for key in tuple(state):
        if key.endswith('img_mlp.gate_up.weight'):
            gate, up = state.pop(key).chunk(2, dim=0)
            state[key.replace('gate_up', 'gate_layer')] = gate
            state[key.replace('gate_up', 'proj')] = up
    result = model.load_state_dict(state, strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys or any(p.is_meta for p in model.parameters()):
        raise ValueError('Incomplete Qwen transformer load')
    return model.eval().requires_grad_(False)


def load_nvfp4_encoder():
    import torch
    from torch import nn
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import Qwen3VLConfig, Qwen3VLModel
    from app.models.nvfp4_runtime import kernel_api
    config = Qwen3VLConfig.from_pretrained(str(OFFICIAL/'text_encoder'),local_files_only=True)
    with init_empty_weights():
        model = Qwen3VLModel(config).to(dtype=torch.bfloat16)
    linear = nvfp4_linear_class()
    kernel = kernel_api()
    def mapped(name: str) -> str:
        if name.startswith('model.visual.'):
            return name.removeprefix('model.')
        if name.startswith('model.'):
            return 'language_model.'+name.removeprefix('model.')
        raise ValueError(f'Unexpected encoder tensor {name}')
    with safe_open(str(QUANT/'text_encoders/qwen3vl_8b_turbo_nvfp4.safetensors'),framework='pt') as saved:
        consumed = {'lm_head.weight'}
        for key in saved.keys():
            if not key.endswith('.comfy_quant'):
                continue
            if json.loads(bytes(saved.get_tensor(key).tolist())) != {'format':'nvfp4'}:
                raise ValueError(f'Unsupported encoder quantization {key}')
            source = key.removesuffix('.comfy_quant')
            names = [source+s for s in ('.weight','.weight_scale','.weight_scale_2')]
            target = mapped(source)
            previous = model.get_submodule(target)
            if not isinstance(previous,nn.Linear) or previous.bias is not None:
                raise ValueError(f'Unexpected encoder target {target}')
            packed,blocks,scale = (saved.get_tensor(n) for n in names)
            replacement = linear(packed,blocks,scale,torch.ones(1,dtype=torch.float32),kernel)
            if (replacement.in_features,replacement.out_features)!=(previous.in_features,previous.out_features):
                raise ValueError(f'Encoder dimensions mismatch {target}')
            parent,leaf = target.rsplit('.',1)
            setattr(model.get_submodule(parent),leaf,replacement)
            consumed.update([key,*names])
        state = {mapped(k):saved.get_tensor(k) for k in saved.keys() if k not in consumed}
    model.load_state_dict(state,strict=True,assign=True)
    if any(p.is_meta for p in model.parameters()):
        raise ValueError('Uninitialized Qwen encoder parameters')
    return model.eval().requires_grad_(False)


def restore_gguf_bf16(model) -> int:
    """Diffusers 0.41 treats BF16 as packed quantization, including in RMSNorm.

    Reinterpret the four native BF16 arrays from the GGUF bytes. Quantized
    weights still use Diffusers' standard GGUFLinear and dequantization code.
    """
    import torch
    from gguf import GGUFReader,GGMLQuantizationType
    fixed = 0
    reader = GGUFReader(str(GGUF/'qwen-image-2.1-turbo-Q4_K_M.gguf'))
    for tensor in reader.tensors:
        if tensor.tensor_type!=GGMLQuantizationType.BF16:
            continue
        target = tensor.name.removeprefix('model.diffusion_model.')
        parent,leaf = target.rsplit('.',1)
        value = torch.from_numpy(tensor.data.copy()).view(torch.bfloat16)
        expected = tuple(reversed([int(n) for n in tensor.shape]))
        if value.shape!=expected:
            raise ValueError(f'BF16 unpacking mismatch {target}')
        module = model.get_submodule(parent)
        if isinstance(module,torch.nn.Linear) and expected!=(module.out_features,module.in_features):
            raise ValueError(f'BF16 projection dimensions mismatch {target}')
        setattr(module,leaf,torch.nn.Parameter(value,requires_grad=False))
        fixed += 1
    if fixed!=4:
        raise ValueError(f'Expected four preserved BF16 GGUF tensors; found {fixed}')
    return fixed


def run(variant: str, kv_cache: bool, only: set[str] | None, encoder_variant: str, resolution: int) -> None:
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline, QwenImage21Transformer2DModel, GGUFQuantizationConfig
    from app.models.nvfp4_runtime import kernel_calls, transformer_storage_bytes

    assert_api_unloaded()
    cases = [c for c in load_cases(OUTPUT) if not only or c.name in only]
    prepared = workspaces(OUTPUT, ARMS['qwen21-turbo'], cases)
    budget,total = memory_budget(torch)
    torch.cuda.set_per_process_memory_fraction(budget/total)
    torch.backends.cuda.matmul.allow_tf32 = True
    protocol = {'variant': variant, 'revisions': REVISIONS, 'seeds': SEEDS, 'sigmas': SIGMAS, 'steps': 8,
                'kv_cache': kv_cache, 'gpu': torch.cuda.get_device_name(), 'vram_budget_gib': round(budget/1024**3, 3),
                'packages': {p:version(p) for p in ('torch', 'diffusers', 'transformers', 'comfy-kitchen')},
                'encoding': encoder_variant,'conditioning_resolution':resolution,
                'embedding_directory': f'embeddings-{encoder_variant}'}
    condition_suffix = '' if resolution==1024 else f'-cond{resolution}'
    label = variant+('-encnvfp4' if encoder_variant=='nvfp4' else '')+condition_suffix+('-kv' if kv_cache else '')
    path = OUTPUT / f'results-{label}.json'
    results: list[Measurement] = []
    def save() -> None:
        path.write_text(json.dumps({'protocol':protocol, 'results':[r.model_dump() for r in results]},indent=2),encoding='utf-8')
    started = time.perf_counter()
    try:
        with MemorySample() as memory:
            if variant == 'nvfp4':
                transformer = load_nvfp4()
            else:
                transformer = QwenImage21Transformer2DModel.from_single_file(str(GGUF / 'qwen-image-2.1-turbo-Q4_K_M.gguf'),
                    config=str(OFFICIAL / 'transformer'), quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                    torch_dtype=torch.bfloat16, local_files_only=True)
                protocol['restored_bf16_tensors'] = restore_gguf_bf16(transformer)
            protocol['transformer_storage_gib'] = round(transformer_storage_bytes(transformer)/1024**3, 3)
            pipe = QwenImage21Pipeline.from_pretrained(str(OFFICIAL),transformer=transformer,
                            text_encoder=None, processor=make_processor(),torch_dtype=torch.bfloat16,local_files_only=True)
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(json.loads((OFFICIAL/'scheduler/scheduler_config.json').read_text()))
            pipe.to('cuda')
            pipe.vae.enable_tiling()
            pipe.set_progress_bar_config(disable=True)
            torch.cuda.synchronize()
        protocol.update(load_seconds=round(time.perf_counter()-started,3),load_peak_rss_gib=round(memory.peak_rss/1024**3,3))
    except Exception as error:
        protocol['load_error'] = f'{type(error).__name__}: {error}'
        save()
        raise
    print(f'LOADED {variant}: {protocol["load_seconds"]}s, storage {protocol["transformer_storage_gib"]}GiB',flush=True)
    save()
    for case in cases:
        item = prepared[case.name]
        embeds = tuple(None if v is None else v.to('cuda') for v in torch.load(OUTPUT/f'embeddings-{encoder_variant}{condition_suffix}'/f'{case.name}.pt',weights_only=True))
        pipe.encode_prompt = lambda **_: embeds
        for seed in SEEDS:
            result = Measurement(variant=variant,case=case.name,seed=seed)
            results.append(result)
            stem = OUTPUT / 'renders' / f'{label}--{case.name}--{seed}'
            stem.parent.mkdir(exist_ok=True)
            torch.cuda.reset_peak_memory_stats()
            before = kernel_calls(transformer)
            start = time.perf_counter()
            def guard(_pipe, _step, _timestep, values):
                if time.perf_counter()-start > 120:
                    raise TimeoutError('Render exceeded 120s')
                return values
            try:
                with MemorySample() as memory, torch.inference_mode():
                    generated = pipe(prompt=case.qwen_prompt,image=item.images() or None,height=item.size[1],width=item.size[0],
                      output_resolution=resolution,num_inference_steps=8,sigmas=SIGMAS,true_cfg_scale=1.0,use_kv_cache=kv_cache,
                      generator=torch.Generator('cuda').manual_seed(seed),callback_on_step_end=guard).images[0]
                    generated.save(stem.with_name(stem.name+'-raw.png'))
                    if item.workspace:
                        generated.save(item.workspace.model_output_path)
                        item.workspace.composite_to(stem.with_suffix('.png'))
                        original = np.asarray(item.workspace.source.convert('RGBA'))
                        with Image.open(stem.with_suffix('.png')) as final:
                            changed = np.any(np.asarray(final.convert('RGBA')) != original,axis=2)
                        alpha = np.asarray(item.workspace.composite_mask)
                        result.protected_changes = int(np.count_nonzero(changed & (alpha==0)))
                        result.selected_changes = int(np.count_nonzero(changed & (alpha>0)))
                    else:
                        generated.save(stem.with_suffix('.png'))
                    alpha = np.asarray(generated.convert('RGBA').getchannel('A'))
                    result.output_size = generated.size
                    result.alpha_min,result.alpha_max = int(alpha.min()),int(alpha.max())
                    torch.cuda.synchronize()
                result.seconds = round(time.perf_counter()-start,3)
                result.peak_allocated_gib = round(torch.cuda.max_memory_allocated()/1024**3,3)
                result.peak_reserved_gib = round(torch.cuda.max_memory_reserved()/1024**3,3)
                result.peak_rss_gib = round(memory.peak_rss/1024**3,3)
                result.min_available_ram_gib = round(memory.min_available/1024**3,3)
                result.native_calls = kernel_calls(transformer)-before
                print(f'DONE {variant} {case.name} {seed}: {result.seconds}s / {result.peak_allocated_gib}GiB / {result.native_calls} native calls',flush=True)
            except Exception as error:
                result.error = f'{type(error).__name__}: {str(error).split("Process ")[0]}'
                print(f'FAIL {variant} {case.name} {seed}: {result.error}',flush=True)
                gc.collect()
                torch.cuda.empty_cache()
            save()


def cycle() -> None:
    """Measure real sequential CPU/GPU streaming, fresh prompts and full VAE decode."""
    import torch
    from torch.func import functional_call
    from diffusers import QwenImage21Pipeline,FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.qwenimage21 import pipeline_qwenimage21 as module
    from app.models.nvfp4_runtime import kernel_calls
    assert_api_unloaded()
    budget,total = memory_budget(torch)
    torch.cuda.set_per_process_memory_fraction(budget/total)
    torch.backends.cuda.matmul.allow_tf32 = True

    class Streamed(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model
            self.config = model.config
            self.gpu_state: dict[str,torch.Tensor] = {}

        @property
        def dtype(self) -> torch.dtype:
            return torch.bfloat16

        @property
        def device(self) -> torch.device:
            return torch.device('cuda')

        @property
        def transformer_blocks(self):
            return self.model.transformer_blocks

        def cache_context(self, name: str):
            return self.model.cache_context(name)

        def upload(self) -> None:
            if self.gpu_state:
                raise RuntimeError('Component is already resident')
            self.gpu_state = {n:t.to('cuda') for n,t in (*self.model.named_parameters(),*self.model.named_buffers())}

        def release(self) -> None:
            self.gpu_state.clear()
            torch.cuda.empty_cache()

        def forward(self, **kwargs):
            if not self.gpu_state:
                raise RuntimeError('Component is not resident')
            return functional_call(self.model,self.gpu_state,args=(),kwargs=kwargs)

    cases = [c for c in load_cases(OUTPUT) if c.name in ('dj','duo-coaster','typography','transparent')]
    prepared = workspaces(OUTPUT,ARMS['qwen21-turbo'],cases)
    started = time.perf_counter()
    with MemorySample() as memory:
        encoder = Streamed(load_nvfp4_encoder())
        transformer = load_nvfp4()
        processor = make_processor()
        pipe = QwenImage21Pipeline.from_pretrained(str(OFFICIAL),transformer=transformer,text_encoder=None,
                         processor=processor,torch_dtype=torch.bfloat16,local_files_only=True)
        pipe.vae.to('cuda')
        pipe.vae.disable_tiling()
        streamed = Streamed(transformer)
        pipe.transformer = streamed
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(json.loads((OFFICIAL/'scheduler/scheduler_config.json').read_text()))
        pipe.set_progress_bar_config(disable=True)
        text_pipe = QwenImage21Pipeline(scheduler=None,vae=None,text_encoder=encoder,processor=processor,transformer=None)
    protocol = {'variant':'cycle-cond512','encoding':'native NVFP4, CPU-mapped weights streamed to GPU per prompt',
                'kv_cache':True,'conditioning_resolution':512,'vae_tiling':False,'revisions':REVISIONS,
                'load_seconds':round(time.perf_counter()-started,3),'load_peak_rss_gib':round(memory.peak_rss/1024**3,3),
                'vram_budget_gib':round(budget/1024**3,3),'timing':'complete encode/upload/render/decode/save; model initial load separate'}
    results: list[Measurement] = []
    path = OUTPUT/'results-cycle-cond512.json'
    def save() -> None:
        path.write_text(json.dumps({'protocol':protocol,'results':[r.model_dump() for r in results]},indent=2),encoding='utf-8')
    save()
    for case in cases:
        item = prepared[case.name]
        for seed in SEEDS:
            result = Measurement(variant='cycle-cond512',case=case.name,seed=seed)
            results.append(result)
            stem = OUTPUT/'renders'/f'cycle-cond512--{case.name}--{seed}'
            start = time.perf_counter()
            before = kernel_calls(transformer)
            torch.cuda.reset_peak_memory_stats()
            try:
                with MemorySample() as memory,torch.inference_mode():
                    encoder.upload()
                    images = qwen_condition_images(module,text_pipe.image_processor,item.images(),512)
                    values = text_pipe.encode_prompt(prompt=case.qwen_prompt,image=images or None,device=torch.device('cuda'))
                    encoder.release()
                    result.encoding_seconds = round(time.perf_counter()-start,3)
                    upload = time.perf_counter()
                    streamed.upload()
                    result.renderer_upload_seconds = round(time.perf_counter()-upload,3)
                    pipe.encode_prompt = lambda **_: values
                    def last_step(_pipe, step, _timestep, tensors):
                        if step==7:
                            streamed.release()
                        return tensors
                    render_start = time.perf_counter()
                    generated = pipe(prompt=case.qwen_prompt,image=item.images() or None,height=item.size[1],width=item.size[0],
                         output_resolution=512,num_inference_steps=8,sigmas=SIGMAS,true_cfg_scale=1,use_kv_cache=True,
                         generator=torch.Generator('cuda').manual_seed(seed),callback_on_step_end=last_step).images[0]
                    generated.save(stem.with_name(stem.name+'-raw.png'))
                    generated.save(stem.with_suffix('.png'))
                    torch.cuda.synchronize()
                    result.render_seconds = round(time.perf_counter()-render_start,3)
                    result.seconds = round(time.perf_counter()-start,3)
                    result.output_size = generated.size
                    alpha = np.asarray(generated.convert('RGBA').getchannel('A'))
                    result.alpha_min,result.alpha_max = int(alpha.min()),int(alpha.max())
                result.peak_allocated_gib = round(torch.cuda.max_memory_allocated()/1024**3,3)
                result.peak_reserved_gib = round(torch.cuda.max_memory_reserved()/1024**3,3)
                result.peak_rss_gib = round(memory.peak_rss/1024**3,3)
                result.min_available_ram_gib = round(memory.min_available/1024**3,3)
                result.native_calls = kernel_calls(transformer)-before
                print(f'CYCLE {case.name} {seed}: {result.seconds}s total / {result.encoding_seconds}s encode / {result.peak_allocated_gib}GiB',flush=True)
            except Exception as error:
                result.error = f'{type(error).__name__}: {str(error).split("Process ")[0]}'
                print(f'FAIL CYCLE {case.name} {seed}: {result.error}',flush=True)
                encoder.release()
                streamed.release()
            save()


def run_klein() -> None:
    import torch
    import benchmark_model_comparison as comparison
    from app.models.nvfp4_runtime import kernel_calls
    assert_api_unloaded()
    cases = load_cases(OUTPUT)
    budget,total = memory_budget(torch)
    torch.cuda.set_per_process_memory_fraction(budget/total)
    # The retained helper predates the WDDM physical-memory guard.
    comparison._gpu_budget_gib = lambda _torch: budget/1024**3
    prepared = workspaces(OUTPUT,ARMS['klein9b'],cases)
    for item in prepared.values():
        item.size = tuple(n//32*32 for n in item.size)
    protocol = {'variant':'klein9b','gpu':torch.cuda.get_device_name(),'steps':4,'seeds':SEEDS,
                'encoding':'BF16 cached prompts, frozen comparison protocol', 'kv_cache':False,
                'output_alignment':32}
    results: list[Measurement] = []
    path = OUTPUT/'results-klein9b.json'
    def save() -> None:
        path.write_text(json.dumps({'protocol':protocol,'results':[r.model_dump() for r in results]},indent=2),encoding='utf-8')
    started = time.perf_counter()
    renderer = KleinRenderer(ARMS['klein9b'],cases,prepared,protocol)
    protocol['load_seconds'] = round(time.perf_counter()-started,3)
    save()
    for case in cases:
        item = prepared[case.name]
        for seed in SEEDS:
            result = Measurement(variant='klein9b',case=case.name,seed=seed)
            results.append(result)
            stem = OUTPUT/'renders'/f'klein9b--{case.name}--{seed}'
            stem.parent.mkdir(exist_ok=True)
            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            before = kernel_calls(renderer.pipe.transformer)
            try:
                with MemorySample() as memory, torch.inference_mode():
                    if case.mode=='compose' and not item.images():
                        generated = renderer.edit_pipe(prompt_embeds=renderer.embeddings[case.name].to('cuda'),
                              height=item.size[1],width=item.size[0],guidance_scale=1.0,num_inference_steps=4,
                              generator=torch.Generator('cuda').manual_seed(seed)).images[0]
                    else:
                        generated = renderer.render(case,seed,None)
                    generated.save(stem.with_name(stem.name+'-raw.png'))
                    if item.workspace:
                        generated.save(item.workspace.model_output_path)
                        item.workspace.composite_to(stem.with_suffix('.png'))
                        original = np.asarray(item.workspace.source.convert('RGBA'))
                        with Image.open(stem.with_suffix('.png')) as final:
                            changed = np.any(np.asarray(final.convert('RGBA'))!=original,axis=2)
                        mask = np.asarray(item.workspace.composite_mask)
                        result.protected_changes = int(np.count_nonzero(changed & (mask==0)))
                        result.selected_changes = int(np.count_nonzero(changed & (mask>0)))
                    else:
                        generated.save(stem.with_suffix('.png'))
                    alpha = np.asarray(generated.convert('RGBA').getchannel('A'))
                    result.output_size = generated.size
                    result.alpha_min,result.alpha_max = int(alpha.min()),int(alpha.max())
                    torch.cuda.synchronize()
                result.seconds = round(time.perf_counter()-start,3)
                result.peak_allocated_gib = round(torch.cuda.max_memory_allocated()/1024**3,3)
                result.peak_reserved_gib = round(torch.cuda.max_memory_reserved()/1024**3,3)
                result.peak_rss_gib = round(memory.peak_rss/1024**3,3)
                result.native_calls = kernel_calls(renderer.pipe.transformer)-before
                print(f'DONE klein9b {case.name} {seed}: {result.seconds}s / {result.peak_allocated_gib}GiB',flush=True)
            except Exception as error:
                result.error = f'{type(error).__name__}: {error}'
                print(f'FAIL klein9b {case.name} {seed}: {result.error}',flush=True)
            save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','encode','run','klein','cycle'),required=True)
    parser.add_argument('--variant',choices=('nvfp4','gguf'),default='nvfp4')
    parser.add_argument('--kv-cache',action='store_true')
    parser.add_argument('--encoder',choices=('bf16','nvfp4'),default='bf16')
    parser.add_argument('--conditioning-resolution',choices=(512,1024),type=int,default=1024)
    parser.add_argument('--only',nargs='*')
    args = parser.parse_args()
    if args.stage=='prepare':
        prepare()
    elif args.stage=='encode':
        encode(args.encoder,args.conditioning_resolution)
    elif args.stage=='klein':
        run_klein()
    elif args.stage=='cycle':
        cycle()
    else:
        run(args.variant,args.kv_cache,set(args.only) if args.only else None,args.encoder,args.conditioning_resolution)


if __name__=='__main__':
    main()
