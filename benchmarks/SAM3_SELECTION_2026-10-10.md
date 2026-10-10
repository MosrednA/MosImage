# SAM 3 selection comparison — 10 October 2026

**Status: completed on the seven frozen images, CPU FP32 and GPU BF16.** Official model access was granted after the initial HTTP 403. All four SAM 3 routes loaded without missing, mismatched or unexpected weights. The editor continues to use SAM 2.1 Tiny/Grounding DINO on CPU; this experiment makes no provider or UI changes.

## Finding

**SAM 3 on the GPU is a substantial speed candidate, but this cohort does not show a general selection-quality improvement.** CPU SAM 3 takes roughly 13 times longer than current single-click selection on a new image. On the RTX 5070 Ti, warmed SAM 3 processes a new image in about 0.15 seconds for points and 0.20 seconds for text. Single-click ambiguity remains; the poster result regresses, and the standard text threshold misses two target objects. Retain the current CPU default while any optional GPU integration receives broader quality and loaded-Klein validation.

These conclusions apply to the fixed prompts and default thresholds below. Six sources are synthetic, only one is a real photograph, and there are no ground-truth object masks. This is a local paired evaluation, not a dataset-level accuracy benchmark.

The [prompt-refinement and real-photo follow-up](SAM3_REFINEMENT_2026-10-10.md) tests calibrated thresholds, equal-input correction clicks/boxes, visual concept prompts and official pedestrian ground truth. The default-settings results here remain unchanged; they do not establish SAM 3's best achievable workflow.

Open the [interactive source/current/SAM 3 CPU/SAM 3 GPU gallery](../data/sam3-benchmark-2026-10-10/review/index.html) and [raw summary](../data/sam3-benchmark-2026-10-10/review/summary.json). The gallery switches between overlays and binary masks; each image opens at native resolution. All assets remain local and ignored by Git.

## Fixed scope and methods

The current Object route uses CPU FP32 `facebook/sam2.1-hiera-tiny`, chooses the highest predicted-IoU mask for a single point, and retains one image embedding. Find uses CPU Grounding DINO tiny at production thresholds (0.3 box / 0.25 text), production `strongest()` filtering and SAM 2.1 box masks, joined across detections. Find caches the SAM image embedding but reruns DINO. ViTMatte cutout refinement is outside this experiment.

SAM 3 point selection uses the official tracker image model, the same single point, three mask candidates and the highest predicted-IoU candidate. Text uses the same English phrase, official instance postprocessing at 0.5 confidence / 0.5 mask threshold, and the union of returned instances. Image embeddings are cached; text is encoded on every call. Scores are not calibrated across different models, so matching raw thresholds would not mean matching operating points. No threshold tuning or prompt rewrite was performed after seeing the failures.

Every run uses eight PyTorch CPU threads and PyTorch SDPA. CPU compute is FP32; the separate GPU option uses BF16 weights and CUDA autocast, with TF32 allowed. The installed Transformers implementation performs the documented shared-backbone/neck mapping from the official combined checkpoint to the point and detector models. Both routes were loaded and measured in separate processes; they were never simultaneously resident in this experiment.

Seven sources and prompts were frozen before the SAM 3 run:

| Case | Text | Positive point (native pixels) | Criterion |
|---|---|---|---|
| Sphere | `terracotta sphere` | 375, 865 | Sphere with leaves/flowers excluded |
| Mug | `yellow mug` | 795, 780 | Body and handle, excluding the handle hole |
| Brooch | `brooch` | 795, 711 | Small coral brooch without the shirt |
| Poster | `poster` | 420, 488 | Paper/artwork without frame or brick |
| Man | `person` | 416, 610 | Complete person, including hands and shoes |
| Phone | `smartphone` | 631, 590 | Phone without the hand/street |
| Camera astronaut | `person` | 245, 330 | Person without background/helmet |

The six synthetic sources are unchanged v1/v2 fixtures. The real-camera astronaut is the unchanged [scikit-image/NASA source](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut) retained by the upscale-fidelity experiment, SHA-256 `88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5`. The [frozen manifest](../data/sam3-benchmark-2026-10-10/baseline-run2/manifest.json) records all image hashes, dimensions, points and criteria. Candidate runners refuse source-hash changes.

Existing inpaint masks include context and shadow, so they are not used as object-mask ground truth. The mask IoU values in the summary measure **agreement between outputs**, not accuracy. Empty/empty agreement is 1.0 but does not establish successful selection of a present target.

## Latency

Hardware: Ryzen 7 9700X, 32 GB RAM, RTX 5070 Ti 16 GB, Windows, Python 3.13.13, PyTorch 2.14.1+cu130, Transformers 5.19.0. Each case has one measured new-image prediction and three cached-image repetitions, after one excluded warm-up per route.

| Route and device | New-image median, seven cases | Cached-image median, 21 repeats |
|---|---:|---:|
| Current point, CPU FP32 | 0.807 s | 0.034 s |
| SAM 3 point, CPU FP32 | 10.735 s | 0.036 s |
| SAM 3 point, GPU BF16 | 0.152 s | 0.019 s |
| Current text, CPU FP32 | 3.223 s | 2.432 s |
| SAM 3 text, CPU FP32 | 11.673 s | 1.084 s |
| SAM 3 text, GPU BF16 | 0.198 s | 0.064 s |

The GPU rows are a separate deployment option, not an equal-device speed comparison. Relative to the current CPU routes, warmed new-image GPU processing is about 5.3 times faster for points and 16.3 times faster for text. CPU SAM 3 benefits on repeated text because it caches the image; first-image encoding dominates its latency.

Timing includes image preprocessing, inference and full-resolution mask postprocessing. It excludes image decoding, model loading, PNG saves, Python startup, the API and job queue. CUDA calls synchronize before and after timing. These are model-route timings, not end-to-end UI latency.

| SAM 3 run | Local checkpoint load | First excluded inference / warm-up |
|---|---:|---:|
| Point CPU | 1.11 s | 10.07 s |
| Text CPU | 1.79 s | 11.80 s |
| Point GPU | 1.14 s | 15.27 s |
| Text GPU | 2.35 s | 15.56 s |

The first GPU use therefore costs roughly **16–18 seconds for loading plus the measured first inference**, before Python startup. The small warmed timings must not be presented as first-launch timings. The one-time official download took about 36 seconds and is excluded. Current cached-checkpoint loading was measured at 3.69 seconds for SAM and 1.33 seconds for DINO in the initial baseline, separate from its warm-up.

## Memory

| Run | Peak process working set | Peak CUDA tensors during measured predictions | Peak CUDA reservation |
|---|---:|---:|---:|
| Current CPU baseline, SAM + DINO together | 3.02 GiB | 0 | 0 |
| SAM 3 point CPU | 2.85 GiB | 0 | 0 |
| SAM 3 text CPU | 4.64 GiB | 0 | 0 |
| SAM 3 point GPU | 2.52 GiB | 1.20 GiB | 1.42 GiB |
| SAM 3 text GPU | 3.40 GiB | 2.12 GiB | 2.38 GiB |

Process RAM includes Python/Torch, mapped weights and transient buffers. The current baseline holds both selection models; each SAM 3 row holds only its one route, so these are process footprints rather than a component-size comparison. CUDA measurements are allocator counters after warm-up, not total graphics/driver VRAM. Unrelated GPU memory was already in use; these runs do not prove Klein coexistence, concurrent generation, shared-backbone residency for both SAM 3 routes, or multi-model editor memory safety.

## Paired visual review

| Case | Current result | SAM 3 result, CPU and GPU |
|---|---|---|
| Sphere | Both routes select the sphere | Similar contour and coverage |
| Yellow mug | Both select body/handle and exclude its hole | Similar; no clear quality gain |
| Small brooch | Point works; text incorrectly selects two background flower regions | Point works; text returns no match. It avoids that false positive but does not select the brooch |
| Poster | Both routes select the paper/artwork cleanly | Point also selects disconnected brick regions; text returns no match at 0.5 |
| Full person | Point selects the zip; text selects the full person | Point selects the jacket, still not the full person. Text selects the person but adds small disconnected background specks near the head |
| Smartphone | Both select the phone | Similar contour; no clear gain |
| Camera astronaut | Point selects a small suit detail; text selects the person | Point still selects that detail. Text gives fuller collar/suit coverage, with hard mask edges and the helmet excluded |

Review all [point sheets](../data/sam3-benchmark-2026-10-10/review/point-comparison-1.jpg), [poster/person/phone point sheet](../data/sam3-benchmark-2026-10-10/review/point-comparison-2.jpg), [camera point sheet](../data/sam3-benchmark-2026-10-10/review/point-comparison-3.jpg), and [text sheet](../data/sam3-benchmark-2026-10-10/review/text-comparison-1.jpg), [poster/person/phone text sheet](../data/sam3-benchmark-2026-10-10/review/text-comparison-2.jpg), [camera text sheet](../data/sam3-benchmark-2026-10-10/review/text-comparison-3.jpg). Native crops preserve [poster spill](../data/sam3-benchmark-2026-10-10/review/poster-point-native-detail.png), [background specks near the person](../data/sam3-benchmark-2026-10-10/review/man-text-native-detail.png) and [mug handle holes](../data/sam3-benchmark-2026-10-10/review/mug-point-native-detail.png).

All **126 repeated predictions** (42 current, 84 SAM 3) are pixel-identical to their first result. CPU FP32 and GPU BF16 are not identical to each other: nonempty text-mask agreement ranges from 0.99929 to 0.99984, and point agreement from 0.98786 to 1.0. The largest difference is 2,440 poster pixels, without changing the overall failure. This is precision agreement, not quality accuracy.

## Checkpoint and reproduction

Official [facebook/sam3](https://huggingface.co/facebook/sam3), pinned revision **`3c879f39826c281e95690f02c7821c4de09afae7`**. Local `model.safetensors`: **3,439,938,512 bytes**, SHA-256 **`6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a`**. Only official safetensors and processor/config files were used. The runner refuses missing or mismatched weight loads. Processor/model interfaces follow the [Transformers SAM 3](https://huggingface.co/docs/transformers/model_doc/sam3) and [point selection](https://huggingface.co/docs/transformers/model_doc/sam3_tracker) APIs; no vendor or app dependency was patched.

Current revisions are SAM 2.1 Tiny `de431c4043854a71d8101e17995dfe596bf101a5` and Grounding DINO tiny `a2bb814dd30d776dcf7e30523b00659f4f141c71`. The baseline records production selection/detection source-code hashes and calls the actual prediction functions.

Run from `backend/`, with a fresh output directory for each candidate. The candidate uses the frozen existing baseline and the pinned checkpoint at `models/research-cache/sam3-REVISION/`:

```powershell
.\.venv\Scripts\python.exe tools/benchmark_selection_baseline.py --output ../data/sam3-benchmark-new/baseline-run2 --repeats 3
.\.venv\Scripts\python.exe tools/benchmark_sam3_selection.py --baseline ../data/sam3-benchmark-new/baseline-run2 --output ../data/sam3-benchmark-new/sam3-point-cpu --route point --device cpu --precision float32 --repeats 3
.\.venv\Scripts\python.exe tools/benchmark_sam3_selection.py --baseline ../data/sam3-benchmark-new/baseline-run2 --output ../data/sam3-benchmark-new/sam3-text-cpu --route text --device cpu --precision float32 --repeats 3
.\.venv\Scripts\python.exe tools/benchmark_sam3_selection.py --baseline ../data/sam3-benchmark-new/baseline-run2 --output ../data/sam3-benchmark-new/sam3-point-gpu --route point --device cuda --precision bfloat16 --repeats 3
.\.venv\Scripts\python.exe tools/benchmark_sam3_selection.py --baseline ../data/sam3-benchmark-new/baseline-run2 --output ../data/sam3-benchmark-new/sam3-text-gpu --route text --device cuda --precision bfloat16 --repeats 3
.\.venv\Scripts\python.exe tools/summarize_sam3_selection.py --input ../data/sam3-benchmark-new
```

The summarizer expects the retained layout (`baseline-run2`, `sam3-point-cpu`, `sam3-text-cpu`, `sam3-point-gpu`, `sam3-text-gpu`). For a new complete review, create its baseline as `baseline-run2`, point every candidate at that same baseline, and pass the new root to the summarizer. It validates every saved mask hash and size, including zero masks, before writing its summary.

The astronaut source currently depends on the ignored local file `data/upscale-fidelity-2026-10-10/assets/astronaut.png`; a fresh checkout needs that exact fixture. Runners do not initialize the editor database, modify live documents, restart/unload the backend, or register a provider. The baseline isolates Settings from unrelated `.env` fields; SAM 3 never imports app Settings. Initial setup failures (root `.env` validation and a removed private Transformers provenance attribute) occurred before baseline inference and are excluded. The first SAM 3 point loader's empty diagnostic sets were serialized as `set()`; subsequent loader reports serialize them as JSON arrays. All four loads were explicitly checked to contain no missing, unexpected or mismatched keys.

## Remaining acceptance and validation

The [follow-up](SAM3_REFINEMENT_2026-10-10.md) completes six real-photo calibration cases, twelve separate pedestrian evaluation cases, threshold sweeps, absent-object controls and equal positive/negative point refinement with mask feedback. Before editor integration, extend those bounded results to broader annotation-complete object cases, SAM/ViTMatte soft cutouts and simultaneous loaded-Klein memory/scheduling. There is no accepted selection-provider switch.

Validation executed **168 measured predictions**, six excluded warm-ups, and **42 full-size binary output masks**. Every repeat was checked against its first mask; every saved mask hash/dimension was verified. Contact sheets and native detail crops were inspected. All 224 gallery asset references exist and its controller passes JavaScript syntax validation. A live Playwright interaction check could not start because its shared browser profile was in use; no existing browser was stopped. Python compilation, the frontend build, local documentation references and `git diff --check` passed. No backend restart, `mosimage.exe` rebuild, commit or push was needed or performed for this research-only change.
