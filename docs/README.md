# MosImage documentation

Status: Current reference
Last verified: 2026-10-17

The guide and architecture describe the current implementation. The roadmap owns remaining scope and next steps. Dated notes and benchmark reports preserve historical decisions and measurements; their timings and acceptance results apply to the recorded fixtures and runtime.

## Using and running MosImage

- [Guide](guide.md): installation, runtime modes, hot loading, storage limits, the desktop shell and a detailed reference for every tool.
- [Architecture](architecture.md): how the engine, job queue, model providers and UI fit together.
- [Roadmap](roadmap.md): the v2 implementation status by phase, plus what comes next.

## Benchmarks

- [FLUX 3 Image Klein versus current Klein 9B, 15 October 2026](../benchmarks/FLUX3_KLEIN_2026-10-15.md): 78 successful local images/requests, matched captions plus current-prompt controls, native Windows NVFP4 and a working sequential 16 GB lifecycle. The user rejected integration: selective visual gains at about twice the cached render time do not meet the required speed or quality improvement. Quality remains Klein 9B.

- [SAM 3 prompt refinement and real-photo evaluation, 10 October 2026](../benchmarks/SAM3_REFINEMENT_2026-10-10.md): completed threshold calibration, equal-input correction clicks/boxes, official pedestrian masks and exploratory fixture recovery; 593 verified masks, 276 identical evaluation repetitions, similar refined point quality, GPU speed and absent-concept rejection gains, with annotation limits.

- [SAM 3 selection comparison, 10 October 2026](../benchmarks/SAM3_SELECTION_2026-10-10.md): seven frozen point/text cases, CPU/GPU timing and memory, repeatability and all paired masks; GPU speed improves, while quality remains mixed and the current CPU default is unchanged.

- [Official Qwen 2.1 Turbo NVFP4 and GGUF, 10 October 2026](../benchmarks/QWEN_TURBO_QUANTS_2026-10-10.md): native Blackwell kernels, controlled eight-step renders, memory/loader failures and prompt-to-file lifecycle; the user prefers FLUX overall and Qwen is not selected for integration.

- [SeedVR2 3B detail, speed and face changes, 10 October 2026](../benchmarks/SEEDVR2_3B_2026-10-10.md): 27 retained outputs, native/reconstruction cases, two seeds, exact repeatability, colour ablation and standalone VRAM; more inferred texture with identity/text/fabric changes, not an accepted default or implemented feature.
- [Natural texture and classical 2x upscaling, 10 October 2026](../benchmarks/UPSCALING_FIDELITY_2026-10-10.md): six native/reconstruction cases, SwinIR Light/M and Swin2SR, BF16 and tile-size comparisons; Light preserves more source texture, with visual acceptance and shared-runtime timing recorded separately.
- [Upscaling quality, speed and Klein residency, 10 October 2026](../benchmarks/UPSCALING_2026-10-10.md): x2plus/x4v3, timings and shared-GPU evidence; the user rejected all displayed faces as too smooth, so photo integration awaits an accepted recipe.
- [LoRA, specialized outpaint and detail crops, 9 October 2026](../benchmarks/QUALITY_CANDIDATES_2026-10-09.md): native NVFP4 compatibility, optional editor switches and GPU acceptance, Consistency V2, fast FAL outpaint, the Base latency rejection and selected texture findings.
- [Benchmark suite](../benchmarks/README.md): fixed fixtures and the protocol behind every measurement.
- [Separate edit masks and resolution-aware expansion, 9 October 2026](../benchmarks/EDIT_MASK_RESULTS_2026-10-09.md): implementation, 30 paired Klein 9B renders, protected pixels, runtime and remaining visual limits.
- [Model comparison and the Klein 9B default, 9 October 2026](../benchmarks/MODEL_COMPARISON_2026-10-09.md): Klein 4B and 9B NVFP4 against Qwen-Image 2.1 turbo on creative and retouching cases, and how 9B fits in 16 GB.
- [Mask, context and reference quality, 8 October 2026](../benchmarks/INPAINT_QUALITY_2026-10-08.md): controlled GPU comparisons with full output galleries and app latency measurements.
- [Klein BF16 against NVFP4, 7 October 2026](../benchmarks/NVFP4_RESULTS_2026-10-07.md)

## Research and implementation notes

These notes are dated snapshots. They record what was tried, measured and decided at the time. Where they disagree with the guide, the guide describes the current behaviour.

- [Quality, speed and UX sweep, 10 October 2026](notes/2026-10-10-quality-speed-sweep.md): hidden token growth, tolerant ring seam matching, grain matching, area-based model size, automatic edges, unload policy and faster polling, with before/after timings and a ComfyUI community check.
- [Image quality across generation, inpaint and outpaint, 9 October 2026](notes/2026-10-09-image-quality-research.md): pre-change pipeline findings, CPU overlap/resolution probe, LoRA candidates and proposed evaluation stages, with a link to the implemented first slice.
- [Focus implementation and validation](notes/2026-10-07-focus-implementation.md): historical delivery log and regression evidence through 10 October; use the guide for current behaviour.
- [Inpainting, context crops and blending](notes/2026-10-07-inpainting.md): the inpainting route, with controlled experiments on:
  - [scale-aware placement](notes/2026-10-07-inpainting.md#scale-aware-placement-research)
  - [Anna/park settings](notes/2026-10-07-inpainting.md#controlled-annapark-experiments)
  - [mask shapes](notes/2026-10-07-inpainting.md#paired-mask-shape-comparison)
  - [pose and layout guides](notes/2026-10-07-inpainting.md#visual-poselayout-guide-comparison)
  - [ComfyUI community leads](notes/2026-10-07-inpainting.md#comfyui-community-leads-checked-7-october-2026)
- [Model research and product direction](notes/2026-10-07-direction.md) (Dutch): the original research snapshot and longer-term proposals.
