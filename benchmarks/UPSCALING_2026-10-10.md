# Upscaling comparison — 10 October 2026

Status: bounded local experiment completed after midnight, following a 9 October start. Native inference works on the current Windows runtime. This adds comparison tools and evidence; **AI upscaling is not yet an editor or Export feature**. HAT and SeedVR2 were not tested.

## Decision and flow

Both architectures meet the bounded speed requirement, but **the user rejected all three displayed face variants on 10 October because skin and beard are too smooth**. This includes x2plus as well as x4v3 denoise 0.5 and 1. The earlier recommendation to start photo export with x2plus is withdrawn. Its relative advantage on denim/beard does not meet the user's naturalness requirement. Do not integrate these recipes as approved photo or face refinements. Stronger compact-model denoise did not solve preservation problems.

The [classical 2x follow-up](UPSCALING_FIDELITY_2026-10-10.md) retains six native enlargements and reconstruction cases for SwinIR Light/M and Swin2SR, including a real camera photograph. SwinIR Light is a more faithful next candidate; the original rejection stays in force. Its FP32, precision, larger-tile and separate shared-runtime results must be judged on their own, rather than inheriting the timings or residency conclusion below.

For a future candidate that passes visual acceptance, Edit would upscale raw generated RGB before its existing Lanczos enlargement to source dimensions. Replace would upscale the contextual generated crop only when the native destination needs more pixels, before drift correction and protected-mask compositing. Fine detail may already generate above destination resolution. Export would flatten visible layers and upscale once into a separate output. A manual action needs a comparison proposal, Apply and Undo; document resizing also needs consistent layer/mask/coordinate scaling. These integrations remain proposed and blocked on acceptable visual quality, not on the measured speed.

## Checkpoints and runtime

Sources: the author's [model zoo](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md) and [inference recipe](https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py). The compact model always computes native 4×; requested 2× output uses an additional Lanczos resize. No face restoration model is used.

| Official checkpoint / release | Bytes | Locally recorded SHA-256 |
|---|---:|---|
| RealESRGAN_x2plus.pth / v0.2.1 | 67,061,725 | `49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb` |
| realesr-general-x4v3.pth / v0.2.5.0 | 4,885,111 | `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292` |
| realesr-general-wdn-x4v3.pth / v0.2.5.0 | 4,885,111 | `1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca` |

Downloaded from the author's GitHub releases to ignored `models/research-cache/Real-ESRGAN/`. The runner checks hashes before `torch.load(weights_only=True)` and passes named tensors to [Spandrel](https://github.com/chaiNNer-org/spandrel). Hashes pin the downloaded bytes; they are not an independent author signature. Denoise 0.5 interpolates general/weak tensors as in the official recipe; denoise 1 uses the general checkpoint alone.

Hardware: Ryzen 7 9700X, 32 GB RAM, RTX 5070 Ti 16 GB; driver 617.42. Python 3.13.13, PyTorch 2.14.1+cu130, CUDA 13.0, Pillow 12.3.0. Spandrel 0.4.2/einops 0.8.2 are installed into ignored `data/upscale-benchmark-2026-10-09/dependencies`; app requirements/environment were not changed. Upscaling uses FP16, 512-pixel input tiles, 32-pixel context and CPU stitching unless specified otherwise. Klein retains BF16/NVFP4. Eight CPU threads, TF32 enabled, no torch.compile or face enhancement.

## Speed

Three repetitions per size/scale after a tiny warm-up. **SR** includes input conversion, CPU/GPU transfers, tiled inference, CPU stitching and final resize, with CUDA synchronization. It excludes model loading, source-file decoding, PNG encoding and the API/browser. PNG is one separate local save measurement; SR + PNG estimates processing/saving, not measured full UI latency. Speed inputs are resized versions of the frozen synthetic man fixture.

| Input | Variant / output | Warm SR median | PNG save | SR + PNG |
|---|---|---:|---:|---:|
| 512 × 512 | x2plus → 1024 × 1024 | 0.072 s | — | — |
| 512 × 512 | x4v3, dn 0.5 → 2048 × 2048 | 0.039 s | — | — |
| 1024 × 1024 | x2plus → 2048 × 2048 | 0.339 s | 0.341 s | 0.679 s |
| 1024 × 1024 | x2plus, halo 128 → 2048 × 2048 | 0.484 s | 0.333 s | 0.817 s |
| 1024 × 1024 | x4v3, dn 0.5 → 2048 × 2048 | 0.289 s | 0.352 s | 0.641 s |
| 1024 × 1024 | x4v3, dn 0.5 → 4096 × 4096 | 0.177 s | 1.500 s | 1.677 s |
| 2048 × 1536 | x2plus → 4096 × 3072 | 1.285 s | 1.039 s | 2.324 s |
| 2048 × 1536 | x2plus, halo 128 → 4096 × 3072 | 2.034 s | 1.023 s | 3.057 s |
| 2048 × 1536 | x4v3, dn 0.5 → 4096 × 3072 | 0.879 s | 1.081 s | 1.960 s |
| 2048 × 1536 | x4v3, dn 0.5 → 8192 × 6144 | 0.535 s | 4.452 s | 4.987 s |

The compact 4× path can be faster than 2× because both compute 4× and 2× then resizes a large image on the CPU. No arbitrary 4K/8K input timings are established. At the largest tested input, PyTorch peak allocated memory was 0.668 GiB for x2plus and 0.163 GiB for x4v3; reserved peaks were 1.662/0.369 GiB. These counters exclude desktop/driver/library allocations and CPU output memory.

The larger x2plus halo reduces tile-boundary differences at a measurable cost: about 43% more SR time at 1024² and 58% at 2048 × 1536 in the separate run. The largest halo-128 input peaked at 1.163 GiB allocated / 2.465 GiB reserved. Those timings do not include Klein residency.

Fresh isolated-process model load plus first 128-pixel inference took **8.4–9.9 seconds**, excluding Python startup/downloads. This differs materially from an already running Klein backend.

### Real Klein residency

The [residency runner](../backend/tools/benchmark_upscale_residency.py) performed three real four-step Quality 9B edits with the same prompt/input/seed and upscaled between them. It uses the existing provider, streamed encoder and serialized JobManager, with cached weights/offline Hub access.

| Stage | Time | Peak allocated |
|---|---:|---:|
| First Klein edit, including cold load/preparation | 62.076 s | 10.927 GiB |
| Load x4v3 beside Klein | 0.198 s | 5.624 GiB total |
| x4v3 2×, 1024² input, median of three | 0.347 s | 5.765 GiB total |
| Next Klein edit, compact model still loaded | 9.513 s | 7.609 GiB |
| Load x2plus beside Klein | 0.202 s | 5.682 GiB total |
| x2plus 2×, 1024² input, median of three | 0.386 s | 6.220 GiB total |
| Next Klein edit, x2plus still loaded | 8.355 s | 7.638 GiB |

Klein transformer identity remained unchanged. Both later Klein PNGs decode **pixel-identically** to the first edit. No reload or OOM occurred. Upscalers run sequentially, and are released before testing the next one. The cold edit is retained as a runtime observation, not attributed to upscaling. Two warm edit samples cannot establish upscaler-induced slowdown/speedup. Residency was measured with a 32-pixel SR halo.

## Quality and contracts

Fixed cohort: garden/flowers, face/beard/jacket, denim pocket, typography/line art. Photo fixtures are **AI-generated synthetic images**, not real-camera photos or a representative dataset. Native references were downsampled 2× with Lanczos, then reconstructed with Lanczos and the three neural variants. All four cases/variants survive: 57 timed neural calls across the baseline speed/quality cohorts, plus a separate larger-halo follow-up.

- **Face:** x2plus sharpens glasses/beard but simplifies detail; compact variants smooth skin/beard more heavily. Neither reconstructs the true source texture exactly.
- **Denim:** x2plus retains more weave/seams here; x4v3 erases fine weave and changes stitching into harder, uneven outlines.
- **Garden:** added contrast/sharpness comes with simplified flower/leaf shapes, especially on x4v3.
- **Text:** x4v3 denoise 0.5 improves some large edges, but small letters change shape in both architectures. x2plus produces broken/wavy glyph edges. Exact text/logo fidelity needs a bypass or protected region.

Lanczos had higher reference PSNR on all three photo/texture cases. On typography, x4v3 denoise 0.5 scored 28.625 dB versus Lanczos 26.382 dB while visually corrupting small labels: a global score does not prove text correctness. PSNR also does not fully assess naturalness. Full outputs and fixed 100% crops were inspected.

**User visual acceptance, 10 October:** all three shown faces are unacceptable because they look too smooth. Preserve this negative result alongside the speed evidence. Follow-up comparisons must retain skin pores, small wrinkles, individual beard hairs and irregular natural texture, without stronger beauty/denoise processing or merely adding grain. Compare against the untouched source and Lanczos at matched display dimensions. Include native-source 2× export as well as the downsample/reconstruct diagnostic; the current diagnostic does not by itself establish naturalness on native photo enlargement. No new model is accepted by this feedback.

The production compositor was exercised once per baseline variant and again for x2plus/halo128 using a 512-pixel architecture workspace, upscaled RGB and the original native soft mask: **zero changes outside the declared selection** in all four outputs. This proves the compositing contract, not newly generated edit quality. Alpha uses geometric resampling and matches expected resized alpha exactly in all four probes. Alpha-edge RGB behaviour still needs wider validation.

### Tile overlap

The 384² portrait compares a full pass with four 192-pixel tiles at different input-context widths. Max/MAE are absolute output-channel levels out of 255, not perceptual scores.

| Variant | Halo | Max delta | MAE |
|---|---:|---:|---:|
| x2plus | 32 | 24 | 0.27334 |
| x2plus | 64 | 7 | 0.10160 |
| x2plus | 128 | 1 | 0.05482 |
| x4v3, dn 0.5 | 32 | 1 | 0.00002 |
| x4v3, dn 0.5 | 64 / 128 | 0 | 0 |
| x4v3, dn 1 | 32 | 1 | 0.00003 |
| x4v3, dn 1 | 64 / 128 | 0 | 0 |

No straight join was apparent in the bounded 100% portrait inspection. x2plus is not tile-equivalent at halo 32: prefer evaluating 128 for production. Larger texture joins remain an acceptance cohort. Larger-halo full outputs and a separate x2plus/128 timing run are retained beside the baseline results.

## Reproduce and review

Run from `backend`, with API models unloaded and no competing inference. Download the three weights from the release URLs in [the author's script](https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py) into the named cache folder. Use project-local TEMP/TMP on Windows.

```powershell
cd backend
$upscaleTemp = Join-Path (Split-Path $PWD) 'data/upscale-benchmark-2026-10-09/temp'
New-Item -ItemType Directory -Force -Path $upscaleTemp | Out-Null
$env:TEMP = $upscaleTemp
$env:TMP = $upscaleTemp
.venv/Scripts/python.exe -m pip install --only-binary=:all: --no-deps --target ../data/upscale-benchmark-2026-10-09/dependencies spandrel==0.4.2 einops==0.8.2
foreach ($upscaleVariant in @('x2plus', 'x4v3-dn05', 'x4v3-dn10')) {
    .venv/Scripts/python.exe tools/benchmark_upscalers.py --variant $upscaleVariant --dependency-path ../data/upscale-benchmark-2026-10-09/dependencies --output "../data/upscale-benchmark-2026-10-09/$upscaleVariant"
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $upscaleVariant" }
}
.venv/Scripts/python.exe tools/benchmark_upscalers.py --variant x2plus --halo 128 --dependency-path ../data/upscale-benchmark-2026-10-09/dependencies --output ../data/upscale-benchmark-2026-10-09/x2plus-halo128
.venv/Scripts/python.exe tools/benchmark_upscale_residency.py --dependency-path ../data/upscale-benchmark-2026-10-09/dependencies --output ../data/upscale-benchmark-2026-10-09/resident
```

Implementation: [benchmark_upscalers.py](../backend/tools/benchmark_upscalers.py). Ignored artifacts: `data/upscale-benchmark-2026-10-09/index.html`, review PNGs, full reconstruction/export PNGs, each variant's `results.json`, `asset-manifest.json`, residency `results.json` and `pixel-equivalence.json`. Weights/libraries/outputs are not committed. The cache manifest also records model URLs/hashes.

## Remaining acceptance work

Find a recipe that passes the user's face/texture acceptance before implementing photo upscaling. Validate native-source enlargement, real camera images, alpha edge colour, text/logo bypass, larger-image tile joins, cancellation between tiles, export size limits, source/proposal/export consistency and UI review. Retain one heavy Klein pipeline; the small SR model can stay loaded as demonstrated. HAT and optional generative restoration remain untested comparisons, with no accepted quality result or measured substitute timings.

## Validation

All three baseline variants and the x2plus/halo128 follow-up completed, including finite output, native dimensions, compositing and alpha checks. The three real Klein residency edits completed with one transformer identity and pixel-identical outputs. Python compilation and frontend production build passed. Documentation links and whitespace were checked. No app runtime/UI changes, release rebuild, commit or push were performed for this experiment.
