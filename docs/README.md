# MosImage documentation

## Using and running MosImage

- [Guide](guide.md): installation, runtime modes, hot loading, storage limits, the desktop shell and a detailed reference for every tool.
- [Architecture](architecture.md): how the engine, job queue, model providers and UI fit together.
- [Roadmap](roadmap.md): the v2 implementation status by phase, plus what comes next.

## Benchmarks

- [Benchmark suite](../benchmarks/README.md): fixed fixtures and the protocol behind every measurement.
- [Model comparison and the Klein 9B default, 9 October 2026](../benchmarks/MODEL_COMPARISON_2026-10-09.md): Klein 4B and 9B NVFP4 against Qwen-Image 2.1 turbo on creative and retouching cases, and how 9B fits in 16 GB.
- [Mask, context and reference quality, 8 October 2026](../benchmarks/INPAINT_QUALITY_2026-10-08.md): controlled GPU comparisons with full output galleries and app latency measurements.
- [Klein BF16 against NVFP4, 7 October 2026](../benchmarks/NVFP4_RESULTS_2026-10-07.md)

## Research and implementation notes

These notes are dated snapshots. They record what was tried, measured and decided at the time. Where they disagree with the guide, the guide describes the current behaviour.

- [Focus implementation and validation](notes/2026-10-07-focus-implementation.md): delivery log and regression evidence, updated through 9 October.
- [Inpainting, context crops and blending](notes/2026-10-07-inpainting.md): the inpainting route, with controlled experiments on:
  - [scale-aware placement](notes/2026-10-07-inpainting.md#scale-aware-placement-research)
  - [Anna/park settings](notes/2026-10-07-inpainting.md#controlled-annapark-experiments)
  - [mask shapes](notes/2026-10-07-inpainting.md#paired-mask-shape-comparison)
  - [pose and layout guides](notes/2026-10-07-inpainting.md#visual-poselayout-guide-comparison)
  - [ComfyUI community leads](notes/2026-10-07-inpainting.md#comfyui-community-leads-checked-7-october-2026)
- [Model research and product direction](notes/2026-10-07-direction.md) (Dutch): the original research snapshot and longer-term proposals.
