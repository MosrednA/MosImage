# Natural texture and classical 2x upscaling — 10 October 2026

Status: follow-up GPU/visual experiment after the user rejected all three [Real-ESRGAN face examples](UPSCALING_2026-10-10.md) as too smooth. Three new author checkpoints and a BF16 ablation are retained. This is research tooling, not an implemented editor or Export capability. User visual acceptance remains open.

## Decision

**SwinIR Light 2x in FP32 is the next review candidate.** On this cohort it keeps individual beard hairs, small wrinkles, hair and cloth detail closer to the source than the rejected x2plus reconstruction. It performs a native 2x enlargement without diffusion, denoise, face restoration or added grain. Larger SwinIR-M and Swin2SR improve reconstruction PSNR slightly on most photographs, but their visible improvement is small and their matched-recipe processing costs roughly twice as much.

This does not establish fully natural skin. Strong native enlargement exposes a fine stippled/raster-like texture in the portrait and cloth; its origin has not been isolated between source texture, learned reconstruction and interpolation. Hair and fabric edges can look more emphasized. The garden retains source softness rather than creating reliable new botanical detail. The synthetic man already has somewhat smooth skin in the original. An upscaler that preserves that source cannot produce missing natural skin texture with certainty. No model here is approved as an automatic face enhancement or quality default.

The subsequent [SeedVR2 3B experiment](SEEDVR2_3B_2026-10-10.md) adds more generative skin/beard detail, with changed eyes, fabric and text. SwinIR Light remains the preservation candidate; SeedVR2 is a separate extra-detail candidate. Its standalone GPU test does not establish coexistence with Klein or approval for automatic inpaint/edit postprocessing.

For a candidate that passes review, start with optional **2x Export** or a manual comparison with Apply/Undo. Post-render Edit/Replace integration would run once on raw generated RGB only when the destination requires more pixels, before the existing resizing, colour correction and protected-mask compositor. A contextual Fine detail crop that already generates above destination resolution should not automatically receive another upscale. Enlargement of the original input and enlargement of a generated edit are distinct operations; this experiment primarily measures the former plus a reconstruction diagnostic.

## Models and runtime

These are the authors' classical/lightweight SR checkpoints, rather than their separately published GAN real-world restoration variants:

| Candidate | Exact checkpoint | Local downloaded SHA-256 |
|---|---|---|
| [SwinIR Light](https://github.com/JingyunLiang/SwinIR) | [002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth) | `193b229909ca89cd8b55de9c9e7fce146ae759d59dfcd78d8feb9dd1d6fa0fd7` |
| [SwinIR M](https://github.com/JingyunLiang/SwinIR) | [001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth) | `2032ebf8f401dd3ce2fae5f3852117cb72101ec6ed8358faa64c2a3fa09ed4ac` |
| [Swin2SR](https://github.com/mv-lab/swin2sr) | [Swin2SR_ClassicalSR_X2_64.pth](https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X2_64.pth) | `04834dbf863f00168ece32f9fc34ad1e3ad48ff1bf8098812573ed5810b59fa0` |

Weights are in ignored `models/research-cache/fidelity-upscalers/`. Hashes pin observed downloaded bytes, not independently signed author provenance. Loading checks the hash and uses `torch.load(weights_only=True)` through Spandrel. Author repositories carry Apache-2.0 licenses. No checkpoints or libraries are added to app requirements.

RTX 5070 Ti 16 GB, Ryzen 7 9700X, 32 GB RAM, Windows, Python 3.13.13, PyTorch 2.14.1+cu130, Spandrel 0.4.2/einops 0.8.2. Isolated dependencies remain in `data/upscale-benchmark-2026-10-09/dependencies`. FP32 with TF32 allowed, eight CPU threads, native scale 2, 256-pixel tiles plus 64 input pixels of context for the matched cohort, CPU stitching. No compilation or face repair. The later tile sweep changes only tile size.

## Frozen cases and comparison method

Six sources are used for **both** native 2x enlargement and half-size downsample/reconstruction:

| Case | Source / crop | Native input |
|---|---|---|
| Man / beard | `fixtures/v2/man.png`, entire image | 832 x 1216 |
| Portrait / skin | `fixtures/v1/portrait.png`, box `(350,80,862,592)` | 512 x 512 |
| Real camera photo | scikit-image astronaut fixture, entire image | 512 x 512 |
| Denim | `fixtures/v1/portrait.png`, box `(610,630,1122,1142)` | 512 x 512 |
| Garden | `fixtures/v1/garden.png`, box `(115,115,1139,1139)` | 1024 x 1024 |
| Text / lines | Frozen runner recipe using Windows Segoe UI | 1024 x 1024 |

The [scikit-image fixture documentation](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut) identifies the astronaut image as a public-domain NASA photograph. Downloaded from the [versioned source](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/astronaut.png), original-file SHA-256 `88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5`. It is a small real-photo sanity check, not a representative camera-image evaluation. Other photographs are synthetic local fixtures; the typography case is deterministic drawing.

Each candidate records reference PNG hashes and sizes. All candidates use the same six reference hashes. Every native/reconstruction output and baseline is retained. Review sheets use identical source-coordinate crops. Native sheets show the source with nearest-neighbour 2x to expose its actual pixels, Lanczos 2x and all three candidates at the same dimensions. Reconstruction sheets show the untouched source, Lanczos and candidates recovered from the same Lanczos half-size input. The previous rejected x2plus output is compared only on the matching reconstruction task.

Native 2x outputs have **no high-resolution ground truth**. Their JSON `reference_psnr` is consistency after reducing the result back to source size, not a measure of newly recovered detail or naturalness. PSNR below is the separate downsample/reconstruct diagnostic. None of these metrics proves good faces.

## Speed

Warm timings include RGB conversion, GPU inference, tile handling and CPU stitching. PNG saving is measured separately. Speed fixtures resize the same man to the stated square dimensions; they are throughput probes rather than additional quality cases. Three repeats per speed case, median shown.

| Matched cohort, 256px tile / 64px context | 256² input -> 512² | 1024² input -> 2048² | PNG save at 2048² | Peak allocated / reserved at 1024² |
|---|---:|---:|---:|---:|
| SwinIR Light FP32 | 0.137 s | 3.944 s | 0.211 s | 0.836 / 1.398 GiB |
| SwinIR Light BF16 + autocast | 0.136 s | 3.990 s | 0.303 s | 0.745 / 1.438 GiB |
| SwinIR M FP32 | 0.245 s | 7.900 s | 0.236 s | 1.542 / 2.873 GiB |
| Swin2SR FP32 | 0.283 s | 9.000 s | 0.237 s | 1.530 / 2.920 GiB |

Native enlargement of the actual 832 x 1216 man takes 4.257 / 8.537 / 9.494 seconds for Light / M / Swin2SR with this matched recipe. Light takes about 0.832–0.835 seconds for the native 512² real-photo/portrait inputs. These are single quality-case measurements.

### Larger tiles on SwinIR Light

The original 256px tiles pay for repeated overlapping context. A separate FP32 sweep on the same 1024² input reduces that work:

| Tile / 64px context | Median processing | PNG save | Processing + save | Peak allocated / reserved |
|---|---:|---:|---:|---:|
| 256 | 3.992 s | 0.209 s | 4.200 s | 0.836 / 1.398 GiB |
| 512 | 2.698 s | 0.207 s | **2.905 s** | **1.822 / 2.318 GiB** |
| 1024, full frame | 2.228 s | 0.202 s | 2.430 s | 5.653 / 7.178 GiB |

Use 512px as the next integration candidate because it cuts processing by about one-third with much lower memory than full frame. Its result differs from the 256-tile output by at most one 8-bit channel value, mean absolute delta 0.000041. Full frame also has max delta one, mean 0.003731. This verifies one 1024² throughput image; it is not a proof for every source size or tile join. Full-frame residency beside Klein has not been tested.

Two additional standalone size probes use the same FP32 model and 512px tile / 64px context, with three timed repeats:

| Input -> output | Median processing | PNG save | Total | Peak allocated / reserved |
|---|---:|---:|---:|---:|
| 512 x 512 -> 1024 x 1024 | 0.539 s | 0.054 s | **0.592 s** | 1.449 / 1.848 GiB |
| 2048 x 1536 -> 4096 x 3072 | 9.863 s | 0.519 s | **10.382 s** | 2.237 / 4.545 GiB |

These resize the source for throughput, not new quality evidence. The larger-image result is a roughly 3.1MP input producing a 12.6MP PNG. It has not been measured beside Klein or checked against full-frame inference. Timings scale with input size and image content; a larger export cannot inherit the one-megapixel latency.

Fresh-process model loading is 1.416–1.599 seconds; the first 64² invocation then costs **14.538–15.264 seconds**. Those cold initialization costs are excluded from warm timing and must not be advertised as a two-second first click. Shared-runtime measurements are recorded separately below.

### Actual Klein 9B residency and unused cache

Two independent runs use the real Quality 9B provider, serialized JobManager, the same four-step jacket recolouring request and seed 7. Each run makes one initial edit, enlarges its completed RGB PNG resized to 1024², and performs the same real edit again while SwinIR Light stays loaded. These are tests of shared residency and unchanged Klein output, not a visual evaluation of raw-edit postprocessing before the provider's own native resizing.

| Stage, 512px tile / 64px context | Existing unused CUDA cache retained | Unused cache explicitly released before SR load |
|---|---:|---:|
| Initial cold Klein edit, including model load | 59.833 s | 48.794 s |
| Cache release | — | 0.064 s |
| Light model load beside Klein | 0.299 s | 0.248 s |
| First 1024² SR invocation | **38.289 s** | **4.743 s** |
| Later SR invocations | **24.105 / 25.775 s** | **3.283 / 2.847 s** |
| Total SR peak allocated / reserved | 7.440 / **13.904 GiB** | 7.442 / **7.969 GiB** |
| Subsequent real Klein edit, Light still loaded | 16.393 s | 7.530 s |

In both runs the Klein transformer retains its exact object identity and the second PNG decodes **pixel-identically** to the first: zero changed pixels. There is no model reload or OOM. Releasing unused allocator cache does not unload live Klein weights. The matched second run changes this cache-release step and reduces reserved memory and SR latency markedly. Windows GPU paging under memory pressure is a plausible explanation, not a directly measured driver-level diagnosis. Cold edit differences and two warm samples do not establish a general edit-speed improvement.

The integration candidate must release unused CUDA cache at the serialized transition into SR, and account for the roughly **3–5 seconds** observed beside Klein. Keeping the model loaded by itself is insufficient to guarantee the standalone 2.7-second timing. Full-frame 1024px SR beside Klein remains untested; its higher memory is unnecessary for the current candidate. Cache policy is tested in the research runner only; no production runtime cleanup or upscale feature was added.

### BF16 negative result

Spandrel marks these architectures as not supporting FP16 and as supporting BF16. A direct BF16 cast passed the tiny warmup but failed at a 256² input because a dynamic shifted-window mask remains FP32, producing a Float/BF16 matmul mismatch. Standard `torch.autocast` around inference resolves that path without changing vendor source. The failure is retained in `bf16-direct-cast-failure.json`.

The complete BF16 Light cohort then passes the same compositor/alpha probes but gives no speed advantage and slightly lower reconstruction PSNR on every case. Tile differences rise from max one to max two channel values. Native FP32/BF16 output comparisons are retained in `bf16-vs-float32.json`; values are pixel differences, not naturalness scores. Prefer FP32 for this candidate.

## Quality and contracts

| Downsample/reconstruct PSNR, dB | Lanczos | Light FP32 | SwinIR M | Swin2SR |
|---|---:|---:|---:|---:|
| Garden | 29.141 | 31.399 | 31.913 | 31.880 |
| Man / beard | 36.036 | 39.687 | 39.921 | 39.923 |
| Denim | 31.557 | 32.687 | 32.820 | 32.805 |
| Text / lines | 26.382 | 34.804 | 34.474 | 34.531 |
| Real camera photo | 31.030 | 34.267 | 34.620 | 34.608 |
| Portrait / skin | 33.325 | 35.549 | 35.864 | 35.838 |

All six native and six reconstruction review crops were inspected for each FP32 candidate. The man comparison retains more beard separation and wrinkles than the rejected x2plus reconstruction. The real photo keeps fine hair and facial shading; all three larger outputs remain close to one another. Portrait pores/freckles remain visible, with the stippled texture caveat above. Denim remains textured but may emphasize repeating source patterns. Small reconstructed text is clearer than Lanczos; native text is already sharp and gains little. This supports a text/logo bypass, not automatic processing of every layer. Garden enlargement keeps source structure and blur; it does not establish recovered missing flower detail.

The production compositor probe uses the existing 512px architecture workspace and native soft mask. **Protected-pixel changes: zero** for all three FP32 candidates and Light BF16. Geometrically resized alpha matches exactly in all four probes. These prove boundary contracts for these probes, not newly generated edit quality or correct RGB under all transparent edges.

On the 256² full/split probe, a 64px halo gives FP32 max difference one channel value and mean delta 0.00155 / 0.00144 / 0.00323 for Light / M / Swin2SR. BF16 Light gives max two and mean 0.06814. Wider image, alpha-edge and cancellation validation remain needed before product integration.

## Reproduction and retained artifacts

Download the three exact author weights above into their runner names `swinir-light2.pth`, `swinir-classical2.pth`, `swin2sr-classical2.pth` under `models/research-cache/fidelity-upscalers/`. Download the versioned astronaut fixture above into `data/upscale-fidelity-2026-10-10/assets/astronaut.png`. The runner rejects different weight/astronaut bytes. Use the isolated dependencies from the [first upscaling report](UPSCALING_2026-10-10.md#reproduce-and-review), not a global pip install.

Run sequentially with the API unloaded and no editor job running:

```powershell
cd backend
$fidelityTemp = Join-Path (Split-Path $PWD) 'data/upscale-fidelity-2026-10-10/temp'
New-Item -ItemType Directory -Force -Path $fidelityTemp | Out-Null
$env:TEMP = $fidelityTemp
$env:TMP = $fidelityTemp
foreach ($fidelityCandidate in @('swinir-light2', 'swinir-classical2', 'swin2sr-classical2')) {
    .venv/Scripts/python.exe tools/benchmark_upscale_fidelity.py --candidate $fidelityCandidate --dependency-path ../data/upscale-benchmark-2026-10-09/dependencies --output "../data/upscale-fidelity-2026-10-10/$fidelityCandidate"
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $fidelityCandidate" }
}
.venv/Scripts/python.exe tools/benchmark_upscale_fidelity.py --candidate swinir-light2 --precision bfloat16 --dependency-path ../data/upscale-benchmark-2026-10-09/dependencies --output ../data/upscale-fidelity-2026-10-10/swinir-light2-bf16
.venv/Scripts/python.exe tools/benchmark_upscale_fidelity_tiles.py --dependency-path ../data/upscale-benchmark-2026-10-09/dependencies --output ../data/upscale-fidelity-2026-10-10/tiles
.venv/Scripts/python.exe tools/benchmark_upscale_fidelity_tiles.py --width 512 --height 512 --tiles 512 --dependency-path ../data/upscale-benchmark-2026-10-09/dependencies --output ../data/upscale-fidelity-2026-10-10/crop-export
.venv/Scripts/python.exe tools/benchmark_upscale_fidelity_tiles.py --width 2048 --height 1536 --tiles 512 --dependency-path ../data/upscale-benchmark-2026-10-09/dependencies --output ../data/upscale-fidelity-2026-10-10/large-export
.venv/Scripts/python.exe tools/summarize_upscale_fidelity.py
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.venv/Scripts/python.exe tools/benchmark_upscale_residency.py --variant swinir-light2 --tile 512 --halo 64 --dependency-path ../data/upscale-benchmark-2026-10-09/dependencies --output ../data/upscale-fidelity-2026-10-10/resident
.venv/Scripts/python.exe tools/benchmark_upscale_residency.py --variant swinir-light2 --tile 512 --halo 64 --release-unused-cache --dependency-path ../data/upscale-benchmark-2026-10-09/dependencies --output ../data/upscale-fidelity-2026-10-10/resident-cache-released
```

Tools: [cohort runner](../backend/tools/benchmark_upscale_fidelity.py), [tile runner](../backend/tools/benchmark_upscale_fidelity_tiles.py), [residency runner](../backend/tools/benchmark_upscale_residency.py), [review generator](../backend/tools/summarize_upscale_fidelity.py). Artifacts are ignored under `data/upscale-fidelity-2026-10-10/`: four complete `results.json` cohorts, native/reconstruction/baseline/full PNGs, matched review crops, `index.html`, failure records, precision comparisons, tile sweep and both real-Klein residency runs with recipe/pixel-equivalence JSON. The page links every FP32 candidate's full PNG, not only selected crops. No model downloads, dependency target or image outputs are committed.

Remaining work is visual acceptance on additional real photos, then a concrete optional Export/manual feature with size limits and cancellation. Edit/Replace additionally needs visual tests on raw Klein results before native resizing and protected compositing. HAT and ClearReality remain untested; no speed or quality is claimed for them. [SeedVR2 3B](SEEDVR2_3B_2026-10-10.md) now has a separate generative-restoration test, with inferred detail and source changes. Existing smooth source faces require better generation/crop detail or an explicitly accepted generative proposal, not a promise of faithful recovery of missing skin.

## Validation

All four six-case cohorts completed; matching source hashes and 14 timing records per cohort were checked. The tile sweep, crop/large-export probes and both real-Klein residency runs completed. Protected pixels and resized alpha are unchanged in all four contract probes; both residency runs preserve transformer identity and decoded edit pixels. Review crops were inspected for every FP32 source/operation. `npm run build` and backend/app plus research-tool compilation pass. Local documentation/review links resolve and `git diff --check` passes. This remains local uncommitted research; the desktop binary and product runtime were not rebuilt or changed for upscaling.
