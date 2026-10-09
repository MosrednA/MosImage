# Klein BF16 versus native NVFP4 — local results

**Status:** Experiment completed locally on 7 October 2026. The experiment originally ran in isolated tooling. Subsequent app integration uses the same native loader: NVFP4 is now the Quality default on the RTX 5070 Ti, with BF16 available in Advanced controls. The measurements below remain the original paired benchmark, not a new app timing cohort.

The [fixed fixture suite](README.md) contains four original AI-generated scenes and deterministic RGBA/binary masks. Three predeclared seeds per scene produced 12 BF16 and 12 NVFP4 outputs; every output was retained and reviewed. Both precisions used the same native Klein inpaint pipeline, BF16 text encoder/VAE, four steps, guidance 1.0, strength 1.0, 320 px context, 16 px inward soft/composite mask and component CPU offload. Each precision ran in a separate process with the API's generation models unloaded. Seed 99999 was a separate warmup and was excluded.

## Measured performance

| Measurement | BF16 | Native NVFP4 |
| --- | ---: | ---: |
| Median warmed end-to-end edit | 9.82 s | 6.26 s |
| Observed latency range | 7.54–11.19 s | 4.66–8.13 s |
| Transformer tensor storage | 7.219 GiB | 2.291 GiB |
| Maximum PyTorch CUDA allocated memory | 9.054 GiB | 9.085 GiB |
| Maximum PyTorch CUDA reserved memory | 9.080 GiB | 9.100 GiB |
| Maximum sampled process RSS during edits | 16.95 GiB | 12.58 GiB |
| Changed protected pixels across all outputs | 0 | 0 |
| Successful native NVFP4 GEMM calls | 0 | 4,800 |

The median NVFP4 edit took **36.25% less time**, a **1.57× ratio** calculated from the cohort medians. Per-scene medians were:

| Scene | BF16 | NVFP4 |
| --- | ---: | ---: |
| Garden | 9.59 s | 6.26 s |
| Kitchen | 10.78 s | 7.57 s |
| Portrait | 7.68 s | 4.74 s |
| Architecture | 9.85 s | 6.26 s |

Transformer storage fell approximately 68%. The whole-pipeline VRAM peak did **not** fall: quantizing only the transformer leaves the much larger BF16 text-encoding phase in place. A separate garden residency probe, excluded from latency/visual counts, observed 7.76 / 7.79 GiB allocated at text-encoder return and 7.42 / 2.53 GiB at transformer return (BF16 / NVFP4). The VAE encoder also runs while text-encoder allocations remain resident in this pipeline's offload order. These component-return samples are not per-component peaks and include live pipeline tensors.

Latency includes prompt encoding, model transfers, sampling, VAE, PNG output and compositing. PyTorch memory counters exclude unrelated desktop apps and some driver/library allocations. RSS is a 100 ms sample of this Python process, not total system RAM or model-loading peak. One sequential cohort per precision cannot establish a universal speedup, confidence interval or absence of order/cache effects. The measured reduction is consistent across all four scenes in this run.

## Visual review of every seed

Review links show BF16 on the top row and NVFP4 on the bottom, with seeds 7, 42 and 20261007 left to right. The [interactive local viewer](../data/nvfp4-benchmark/index.html) supports all fixtures/seeds, source comparison, mask overlays, detail/full-image views and source/mask download. Open it as a local HTML file or serve the results directory.

| Scene and all three paired seeds | Observed behavior | Remaining limitation |
| --- | --- | --- |
| [Garden](../data/nvfp4-benchmark/garden-comparison.jpg) | All six contain a complete mushroom grounded among the flowers. Seed 7 is close across precisions; seed 42 has a rounder cap under NVFP4; seed 20261007 is again close | Strong yellow illumination and altered vegetation around the base remain visible. Complete placement does not prove invisible integration |
| [Kitchen](../data/nvfp4-benchmark/kitchen-comparison.jpg) | All six remove the mug/handle and reconstruct veined marble. Paired results are very similar | A soft gray shadow-shaped patch remains. The mask covers the mug/contact-shadow region but its far soft penumbra reaches protected pixels near the right edge; this depends on mask coverage as well as sampling, not solely precision |
| [Portrait](../data/nvfp4-benchmark/portrait-comparison.jpg) | All six replace the coral brooch with a white daisy and preserve the face exactly. BF16 seed 42 adds a tiny green stem; NVFP4 produces a more compact flower | At this small scale petals can read as a flat patch rather than clearly resolved stitches; textile integration needs stricter detail review |
| [Architecture](../data/nvfp4-benchmark/architecture-comparison.jpg) | All six produce blue botanical art on cream paper with the existing perspective/reflections. The protected frame and bricks are identical | Seed 7 changes the botanical silhouette/scale somewhat; no clear precision-specific degradation is evident in this fixture |

No visual output was excluded or replaced by a better seed. These are subjective observations on four synthetic scenes, without ground-truth edited images or blinded human ratings. Neither precision automatically solves residual shadow/glow/texture issues. NVFP4 is a promising speed/RAM option, not a new editing model or general quality fix.

## Runtime and reproducibility

- GPU: RTX 5070 Ti, compute capability 12.0, 16 GB profile.
- Runtime: Torch 2.14.1+cu130, Diffusers 0.41.0, Transformers 5.19.0, Accelerate 1.15.0, safetensors 0.8.0, Pillow 12.3.0.
- [BF16 base](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B): revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`.
- [Official NVFP4 transformer](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4): revision `1db2b2f776c24b76f1122e5f69ab1949fc620068`, safetensors file 2,460,413,488 bytes. The file excludes text encoder/VAE and working memory.
- [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen) 0.2.37 is the checked latest release, originally installed in `backend/.nvfp4-deps`, and now pinned/installed in the normal app environment. The Windows wheel is SHA-256 pinned in optional requirements. OSV returned no known advisory IDs; this is not a full security audit.
- The adapter uses checkpoint-calibrated input/global/block scales and native CUDA quantization/GEMM, with eager/Triton fallback disabled. All 100 mapped quantized target linears ran four times per edit. Unquantized transformer layers retain BF16; no ComfyUI source was copied.

CUDA adapter checks passed complete QKV scale-tile splitting, nonaligned 17-row padding/reshape, finite output, native GEMM versus a dequantized-operand reference (0.002211 relative mean absolute error), and rejection of CPU inputs without BF16 fallback. The backend suite passes 41 tests, including source/mask hashes, binary/RGBA equivalence, native dimensions and prepared soft-mask coverage. The viewer passed all 12 fixture/seed combinations, original/full-size mask toggles, desktop/mobile overflow checks and had no page errors.

Raw evidence: [summary](../data/nvfp4-benchmark/summary.json), [BF16 outcomes](../data/nvfp4-benchmark/bf16/results.json), [NVFP4 outcomes](../data/nvfp4-benchmark/nvfp4/results.json), [BF16 residency](../data/nvfp4-benchmark/profile-bf16/results.json), [NVFP4 residency](../data/nvfp4-benchmark/profile-nvfp4/results.json), [dependency check](../data/nvfp4-benchmark/dependency-check.json). Source-generation provenance is in [generation-prompts.json](fixtures/v1/generation-prompts.json).

Application adoption now includes pinned checkpoint loading, explicit unsupported-runtime errors, one-pipeline residency, cancellation/unload and generation as well as inpainting. App integration regression coverage includes soft-mask routing for both precisions, failed/cancelled load publication and precision-switch cleanup. Further total VRAM reduction needs text-encoder precision/placement measurement. Quality work should continue with shadow/glow mask coverage and controlled integration comparisons using the fixed suite.

Subsequent API integration checks repeated all four scenes at seed 7: outputs were pixel-identical to the corresponding NVFP4 benchmark results, including exact protected pixels. Main-app generation, Advanced precision selection, mobile settings, expansion routing, precision switches and loading/inference cancellation passed. The updated backend suite passes 49 tests; frontend build and 17 tests pass. See the [current app validation](../docs/notes/2026-10-07-focus-implementation.md#nvfp4-application-validation) for scope and local evidence.
