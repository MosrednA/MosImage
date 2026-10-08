# Mask, context and reference quality investigation — 8 October 2026

Status: completed local experiment and verified prompt-cache fix. Quality presets and user documents are preserved. Broad editing-quality acceptance remains open.

User direction during this experiment: prefer ordinary reference photographs and masks, without generated pose images. Pose/layout results are retained as controls, not the recommended workflow. The subsequent five-seed confirmation uses one ordinary identity reference and mask/crop/resolution choices only.

## Protocol

The first cohort declares 12 recipes and seeds 7, 42, 20261008 before inspecting results. It compares the original curved park mask with a rectangular mask of the same bounding box; 0/8/16/32 px inward edges; four/eight distilled steps; 32/96/192 px context; 384/512/768 px references; an additional face close-up; and an identity/layout reference pair. The scene and Anna reference are the existing local synthetic fixtures. The rectangular mask deliberately permits changes in more pixels than the original curved selection.

All recipes, exact prompts, source/mask/reference SHA-256 hashes, model revisions and package versions are saved in `data/inpaint-quality-2026-10-08/screen-v2/protocol.json`. Results include every seed, raw generated crops, final composites and per-recipe comparison sheets. No best-seed-only reporting.

Review each output for: complete subject/head/feet; plausible anatomy/pose; reference likeness, hair, clothes and colours; plausible scale/perspective; contact with the ground; seams, lighting and texture. A complete face alone is not an overall quality pass. Classify **good** (usable at normal viewing size), **mixed** (recognizable subject but visible repair needed), or **failed** (missing/clipped/incorrect subject or obvious construction artifact). Identity is a visual assessment, not biometric verification. This small synthetic suite cannot establish a general success rate.

The renderer is the app's native NVFP4 implementation, BF16 VAE, Diffusers Klein inpaint pipeline, CUDA, TF32 and guidance 1.0. Text embeddings are prepared separately and the encoder released before loading the renderer. A 65% PyTorch allocation budget protects headroom; it does not cap total desktop/driver use. Per-render times include diffusion, VAE, PNG output and protected compositing, but exclude prompt encoding, model loading and initial warmup. Those setup costs are recorded separately. This is not a claim about normal interactive app latency.

## Primary-source research

- [BFL Klein 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B): official distilled recipe is four steps, guidance 1.0, BF16 and model CPU offload; supports multiple references. It does not guarantee likeness or prompt compliance.
- [BFL inference repository](https://github.com/black-forest-labs/flux2): distinguishes step-distilled Klein from the longer-running Base variants. More steps on distilled weights are an experiment, not automatically a quality upgrade.
- [Diffusers inpainting guide](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint#padding-mask-crop): crop-and-overlay focuses generation on the selected region with surrounding context. Its illustrative defaults are not a Klein-specific guarantee; local crop/context tradeoffs need measurement.
- [BFL image editing](https://docs.bfl.ai/flux_2/flux2_image_editing): describes text-directed, multi-reference edits. Hosted model examples do not establish local Klein 4B quality or timing.
- Installed Diffusers 0.41.0 source, `pipeline_flux2_klein_inpaint.py`: source latents are blended with generated latents using the mask after scheduler steps; reference images are encoded separately. This explains why protected compositing can be exact while subject placement is imperfect. Multiple references are preprocessed at the first reference's dimensions; the tests use equal-size canvases.
- [LanPaint author repository](https://github.com/scraed/LanPaint): offers additional conditional-sampling iterations and Klein examples. This is a plausible separate sampler experiment, but its quality/cost is not measured here and ComfyUI workflows are not automatically compatible with the app. No dependency or runtime replacement was made.

## Reproduction

From `backend/`, with the app model unloaded (the API may also be stopped):

```powershell
.\.venv\Scripts\python.exe tools/benchmark_inpaint_quality.py --output ../data/inpaint-quality-new/screen --variants curved4 box4 box8 hard4 edge8 edge32 context96 context192 ref384 ref768 face-detail layout4
```

Use a fresh output directory. The tool refuses to overwrite earlier experiments. All local inputs under `data/park-scale-2026-10-07/` must exist; the checked-in four-scene suite alone is insufficient for the Anna experiment.

## Results

**122 completed offline renders**, plus four successful post-fix app-provider edits. All 126 preserve every pixel outside their declared soft masks exactly. The larger masks authorize a larger edit footprint; this is not preservation of the earlier, tighter mask. Offline renderer allocation peaked at **4.687 GiB**. Text encoding has a separate, larger peak; these figures exclude other applications and driver allocations.

Local evidence root: `data/inpaint-quality-2026-10-08/`. Open `index.html` for the overview and links to all four galleries; `no-pose-examples.jpg` shows selected examples alongside their actual masks and ordinary references. `summary.json` contains per-recipe metrics. Full-size outputs and raw crops are retained, not just thumbnails.

| Cohort | Renders | Purpose |
|---|---:|---|
| `screen-v2` | 36 | Mask shape/edge, steps, context, reference size, additional face/layout reference |
| `broad` | 57 | Crop resolution, historical pose-init control, four existing tasks and an object reference |
| `no-pose-confirm` | 20 | Four no-pose recipes on five new seeds: 11, 123, 2048, 90817, 24680 |
| `mask-repair` | 9 | Corrected object prompt, wider object mask, complete shadow/removal mask |

### Ordinary identity references

| Recipe | Median warmed render | Visual findings |
|---|---:|---|
| Original curved mask, four steps | 1.146 s | All three heads visibly intersect the protected background; unacceptable. |
| Box, 16 px edge, context 32, reference 512, four steps | 1.166 s | Complete faces in all three screening seeds and all five new seeds. Pose, perceived age/likeness and shoe contact still vary. |
| Same box, eight steps | 2.017 s | No convincing consistent improvement in seated pose/contact to justify the extra time. |
| Hard / 8 / 32 px edge | 1.190 / 1.189 / 1.206 s | Similar overall composition. A hard edge risks seams; a wide transition makes shoes/edges translucent. Sixteen pixels is a useful tested compromise for this mask, not a universal radius. |
| Context 96 / 192 | 1.838 / 3.301 s | Subject grows relative to the mask; clipping reappears. More context is not automatically better. |
| Reference 384 / 768 | 1.081 / 1.601 s | Larger reference costs more with no consistent likeness win here. 512 is a reasonable middle ground. |
| Full reference plus facial close-up | 1.591 s | No reliable improvement over the single reference. This recipe also changes padding and role wording, so it does not isolate reference count alone. |
| Same box, crop upscaled to 768 / 1024 | about 1.9 / 3.6 s | More facial/detail resolution, but no universal pose/identity improvement. Additional five-seed 768 test retains complete faces. |
| Lower, wider target mask at native resolution / 768 | about 1.1 / 2.3 s | Smaller overall placement through the mask alone, but some heads still touch its upper edge. The 768 version gives useful examples, not a new universal preset. |

Recommended starting recipe for this person insertion: a roomy rectangular selection with full-opacity space above the head and around shoes/bench, 16 px inward edge, 32 px surrounding context, one 512 px ordinary reference, four steps, guidance 1, strength 1, native crop size. Use 768 px crop inference as an optional detail experiment; the app does not yet expose this upscaling operation. The broad box was more reliable than reducing the selection to force smaller placement. Exact real-world scale still cannot be inferred from the mask or reference alone.

### Object replacement, removal and detail

- **Mug reference:** the original prompt mistakenly described a pale, fluted mug, while the actual reference is golden yellow and smooth. Those six flawed outputs remain visible, labelled as an invalid reference-fidelity comparison. A corrected prompt alone restores material/colour but clips the handle on the old sphere-sized mask. With the same corrected prompt and a wider mask, all three results have one complete handle, recognizable mug geometry, flower occlusion and a grounded base; median **1.429 s**, peak **3.279 GiB**. This is the clearest mask/reference improvement. It is recognizable product transfer, not exact commercial product reproduction.
- **Mug removal:** the historical mask omits the long reflection/shadow tail, leaving a hard patch despite a removed cup. Extending the mask through the shadow/reflection removes that hard discontinuity in all three follow-ups; median **2.517 s**, peak **3.979 GiB**. Some tonal variation remains. The edit footprint is substantially larger.
- **Small daisy embroidery:** context 96 is fast (~0.44 s) but texture can resemble a flat flower sticker. Upscaling that crop to 768 (~2.13 s) makes thread detail more convincing. The face stays protected.
- **Framed botanical print:** context 96 gives plausible perspective and protected frame/bricks in all three outputs at ~0.96 s; context 320 costs ~3.55 s with no necessary quality benefit. Reflection and colour can vary within the edited paper.
- **Glowing mushroom:** all nine outputs contain the requested subject. Context 320 and the 768 detail crop produce more convincing cap/stem texture than the small native crop, at ~2.7 and ~2.2 s versus ~0.8 s. The glow remains stylized and is confined by the editable mask.

These observations are qualitative inspection of every displayed seed, plus full-scene review of the selected examples. They are not blinded human ratings or a representative benchmark. Good local examples coexist with failures; no single all-purpose success percentage is claimed.

### Retained pose controls, not the preferred workflow

The layout pair improved relative position and size but not reliable foot contact. Initializing the masked source with the existing dummy/bench at strength 0.6 sometimes left a bald or mannequin-like head; 0.8/1.0 restored people but still changed pose. These 12 outputs are retained because they were already part of the experiment. No new pose asset was generated in this run, and no pose-dependent product feature was added.

## Actual app performance and the verified fix

Testing the real `Flux2KleinNvfp4Provider`, `JobManager`, prompt cache and compositor uncovered a separate memory bug. The cache called `pipe.encode_prompt` outside the pipeline's no-grad `__call__`, retaining an autograd graph and encoder activations for each new prompt. A real CPU-tensor regression test failed before the fix: both cached tensors had `requires_grad=True`. The encoding call now runs under `torch.inference_mode()`, so only the embeddings are retained. This affects both BF16 and NVFP4 providers through their shared implementation.

Balanced mode, one 512 px reference, context 32, four steps, seeds 7/42/20261008; the fourth request changes the prompt and uses seed 1234:

| Request | Before | After | Peak PyTorch allocation after |
|---|---:|---:|---:|
| First edit, including model loading | 33.441 s | 32.521 s | 8.087 GiB |
| Same prompt, next seed | 1.917 s | 2.182 s | 2.957 GiB |
| Same prompt, another seed | 1.898 s | 2.077 s | 2.957 GiB |
| Changed prompt | Exceeded 90 s; cancelled | 4.198 s | 8.125 GiB |

Before the fix, repeated prompts still peaked at **10.396 GiB**; after, **2.957 GiB**. The three successfully comparable before/after output PNGs are **pixel-identical**. The fix improves memory retention and changed-prompt latency, not the underlying image-generation quality. The small repeated-prompt timing differences are not a speedup claim. Cold model loading remains a significant first-use cost. Measurements exclude browser and HTTP upload time. The post-fix cold-start overlapped the short CPU-only unit test run, so its exact time is indicative rather than a controlled loading benchmark.

App evidence: `app-runtime-default/results.json` and `app-runtime-fixed/results.json`; timeout is in `data/inpaint-quality-app-default.log`. A first diagnostic run using an artificial 68% allocation cap hit that cap, even with physical memory free; it is retained in `app-runtime/` and is **not** evidence that the unmodified app always OOMs. The initial offline `screen/` attempt stopped after warmup because the benchmark passed no model to `kernel_calls`; that harness error was fixed before the 36-image `screen-v2` run. Neither aborted attempt counts as a successful render.

### Validation

- Regression test reproduced the autograd-cache bug, then passed after the fix.
- All **97 backend tests pass**; app/tool syntax compilation passes.
- `npm run build` passes.
- Exact protected-pixel comparison passes for all 122 offline and four fixed-provider images.
- The earlier three app outputs and their post-fix equivalents are pixel-identical.
- Initial compositor test failures were sandbox temporary-directory permissions; rerunning with project-local `TEMP`/`TMP` passed. No application behavior was changed to bypass them.

### Additional reproduction

```powershell
# New seeds, ordinary reference only; no pose images passed to inference.
.\.venv\Scripts\python.exe tools/benchmark_inpaint_quality.py --output ../data/inpaint-quality-new/no-pose --variants box4 upscale768 target4 target768 --seeds 11 123 2048 90817 24680
.\.venv\Scripts\python.exe tools/benchmark_inpaint_quality.py --output ../data/inpaint-quality-new/masks --variants mug-corrected mug-wide kitchen-full-shadow
# Sequentially, after the offline process has exited:
.\.venv\Scripts\python.exe tools/benchmark_inpaint_app_latency.py --assets ../data/inpaint-quality-new/no-pose/assets --output ../data/inpaint-quality-new/app
```

This completes the investigation cohort. Remaining work is broader real-photo validation, automatic mask-placement assistance and a separately measured optional crop-upscaling UI. Current generation defaults stay unchanged; restart the backend to use the prompt-cache fix.
