# SAM 3 prompt refinement and real-photo evaluation — 10 October 2026

**Status: completed.** Six real-photo calibration cases, twelve separate evaluation cases, and all seven original diagnostic sources are retained. All 593 binary masks passed size, value and hash checks; all 276 evaluation repetitions were pixel-identical. This follows the [seven-image default-settings comparison](SAM3_SELECTION_2026-10-10.md). The editor's selection providers are unchanged.

**Finding:** extra clicks and previous-mask feedback substantially improve SAM 3, reaching mean annotation IoU 0.929 against 0.925 for equally refined SAM 2.1 Tiny. That small paired mean difference does not establish a general quality advantage. SAM 3 is much faster on the GPU and rejects the absent `astronaut` concept on all twelve street photographs; the current default text route incorrectly selects ordinary people on all twelve. Text-only mask ranking is limited by incomplete dataset annotations. The poster/brooch failures can be recovered with targeted prompts or explicit boxes, so the original default-settings result does not describe SAM 3's practical ceiling.

Open the [complete interactive comparison](../data/sam3-refinement-2026-10-10/review/index.html) and [verified summary](../data/sam3-refinement-2026-10-10/review/summary.json). The gallery includes sources, official masks, every evaluation route and every exploratory variation; it switches between overlays and binary masks.

## Protocol frozen before evaluation

The official [Penn-Fudan pedestrian dataset](https://www.cis.upenn.edu/~jshi/ped_html/) provides real street photographs and instance masks; the [PyTorch tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html#writing-a-custom-dataset-for-pennfudan) describes the mask IDs. This experiment selects eighteen images without inspecting model predictions: rank each filename by SHA-256 of `mosimage-sam3-refinement-v1:filename`, then take three calibration and six evaluation images from each of the Fudan and Penn groups. The archive hash is `9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2`.

The [frozen source/annotation manifest](../data/sam3-refinement-2026-10-10/manifest.json), [protocol](../data/sam3-refinement-2026-10-10/protocol.json), and [cohort preview](../data/sam3-refinement-2026-10-10/frozen-cohort.jpg) retain every filename, hash, size and input coordinate. The twelve evaluation photographs are withheld from threshold selection. This is a local calibration/evaluation split; the public dataset may have appeared in model training.

- Point/box target: the largest annotated pedestrian. One torso click; three positive clicks plus a negative background click; the same refined clicks with the previous selected low-resolution mask logits; and the person's bounding box. All input points and boxes are derived from annotations to simulate equal ideal user input, not an automatic locator. Neither model chooses a candidate using ground-truth IoU.
- Text target: the union of all annotated pedestrians for `person`. Each model sweeps confidence thresholds `0.1, 0.2, 0.3, 0.4, 0.5, 0.6` on the six calibration images. Choose the highest mean mask IoU with zero nonempty selections for absent `zebra`; exact ties prefer the higher threshold. Freeze the selected values before evaluation.
- Evaluation negatives: `zebra` and the harder human-shaped `astronaut` query, neither present in the photographs. The latter was declared before evaluation and does not affect calibration. This limited control set does not establish broad absent-object precision.
- SAM 3 concept diagnostics: text plus the positive person box, text plus a background negative box, and reference-style mask-logit resizing. The boxes are explicit additional information; these rows are not equal-input comparisons against text alone.

The original seven sources receive an exploratory sweep of short noun phrases, thresholds and manual clicks/boxes, frozen in a separate [diagnostic manifest](../data/sam3-refinement-2026-10-10/diagnostic-manifest.json). Those images have no object ground truth, so their masks are visually reviewed without reporting accuracy scores or selecting per-image production defaults.

## Implementation and measurement

Current selection uses production SAM 2.1 Tiny and Grounding DINO tiny on CPU FP32, including production `strongest()` filtering and DINO's 0.25 token threshold. SAM 3 uses the same pinned official checkpoint as the earlier comparison through Transformers, on CUDA BF16 with TF32 and SDPA. Each process uses eight CPU threads. Model runs are sequential to avoid shared-device timing interference.

One excluded warm-up precedes each scored run. Each evaluation route has an initial prediction and a cached-image repetition. The first single-point request includes new-image encoding; refined clicks and boxes then reuse that embedding. Every text route starts with an empty image cache, and its repetition retains the image embedding. File loading, PNG saves, model loading, warm-up, API, queue and UI time are excluded from prediction timings. Exploratory diagnostic timings are not warmed latency claims.

IoU compares each binary mask to the official annotation target, unlike the previous report's model-output agreement. Precision/recall are pixel metrics. **Visual review found incomplete/ambiguous annotation:** some clearly visible people are unlabelled (for example FudanPed00008/00046), some carried bags are included, and overlapping people can have ambiguous instance boundaries (PennPed00031). A correct extra person detection can lower text IoU. These results establish agreement with the provided labels, not complete semantic accuracy or the ideal edit footprint. Per-image metrics, misses and negative controls remain in raw results. Binary masks do not assess hair matting or ViTMatte cutouts.

The [reference text postprocessor](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image_processor.py) resizes logits before sigmoid/threshold; the installed Transformers version resizes sigmoid probabilities. The additional row changes only that ordering, with the same network inputs and confidence rule. It is not proof of complete parity with Meta's native package. The installed tracker postprocessor also leaves hole/sprinkle processing unimplemented; no hidden custom cleanup is applied.

## Results

### Equal-input point and box selection

Mean IoU across twelve evaluation photographs; the target is the largest annotated person. The input clicks and boxes are the same for both models.

| Input | Current SAM 2.1 Tiny CPU | SAM 3 tracker GPU |
|---|---:|---:|
| One torso click, highest predicted-IoU candidate | 0.858 | 0.801 |
| Three positive clicks plus negative background click | 0.924 | 0.902 |
| Same refined clicks plus initial mask logits | 0.925 | 0.929 |
| Person box | 0.915 | 0.910 |

SAM 3 improves from 0.801 to 0.929 with explicit refinement and mask feedback. The current model also improves. For feedback, SAM 3 exceeds the current score by more than 0.02 on two images, the current model exceeds SAM 3 by more than 0.02 on three, and seven are within 0.02. The minimum feedback IoUs are 0.824 versus 0.728, but the lowest case has overlapping-person annotation ambiguity. Neither a 0.004 mean difference nor this minimum is a general accuracy claim.

The current app already uses multiple clicks with `multimask_output=False`; it does not feed previous mask logits back into later clicks. The feedback rows are experimental on both models. Current SAM 2.1 gains only 0.0007 mean IoU from feedback and regresses on some individual images, so this experiment does not justify automatically changing its refinement behavior.

### Calibration and concept prompts

| Confidence threshold | Current calibration mean IoU | SAM 3 calibration mean IoU |
|---|---:|---:|
| 0.1 | 0.8362 | 0.7928 |
| 0.2 | 0.8362 | 0.8006 |
| 0.3 | 0.8362 | 0.8040 |
| 0.4 | **0.8368** | 0.8059 |
| 0.5 | 0.8306 | 0.8061 |
| 0.6 | 0.8117 | **0.8115** |

All calibration `zebra` queries are empty. The frozen rule selects current 0.4 and SAM 3 0.6. These are fitted values for this small person cohort and declared grid, not global defaults. Calibration uses the same annotation convention and limitations as evaluation.

| Text input / processing | Evaluation mean annotation IoU | Nonempty positive selections |
|---|---:|---:|
| Current `person`, production 0.3 | 0.829 | 12/12 |
| Current `person`, fitted 0.4 | 0.820 | 12/12 |
| SAM 3 `person`, reference-example 0.5 | 0.785 | 12/12 |
| SAM 3 `person`, fitted 0.6 | 0.788 | 12/12 |
| SAM 3 fitted + positive person box | 0.792 | 12/12 |
| SAM 3 fitted + negative background patch box | 0.667 | 10/12 |
| SAM 3 fitted + reference-style logit resizing | 0.7885 | 12/12 |

The current fitted threshold loses evaluation recall and IoU, despite its calibration improvement. Retain the production threshold. SAM 3 tuning gives only a small gain; reference-style resizing changes boundaries slightly and does not account for the original misses. **Do not automatically add a negative box on an empty background patch:** this specific heuristic creates two complete positive misses. This is a result for the tested patch heuristic, not a conclusion about correctly supplied semantic negative exemplars in general.

Incomplete person labels penalize extra detections, so the text IoU table alone cannot establish which model recognizes all visible people better. The negative controls provide a separate, directly interpretable observation:

| Absent query | Current default false selections | Current fitted false selections | SAM 3 default / fitted false selections |
|---|---:|---:|---:|
| `zebra` | 0/12 | 0/12 | 0/12 / 0/12 |
| `astronaut` | **12/12** | **11/12** | **0/12 / 0/12** |

All street sources were visually checked and contain ordinary pedestrians, with no astronauts or zebras. On this human-shaped negative control, Grounding DINO often returns ordinary people; SAM 3 rejects the absent concept. One hard-negative category on twelve photos does not establish broad rejection accuracy.

### Original fixture recovery

- **Poster text:** `poster` is empty at 0.5, but returns a mask with score about 0.305 at 0.3. `framed poster` returns the artwork at 0.5 with score about 0.980. A positive visual box also recovers it. Frame coverage still requires review. Conversely, `picture` at 0.1 selects nearly the entire image: lowering thresholds is not a universal repair.
- **Brooch text:** `brooch` and `coral brooch` remain empty even at 0.1. `flower pin` finds a small candidate only at 0.1, around score 0.111. A positive box with the original `brooch` phrase finds the brooch around score 0.949, with some edge margin. Extra spatial information matters more than a blanket lower threshold.
- **Poster point:** correction clicks remove the original detached brick spill; small mask holes/frame-edge issues remain. The current model also receives and benefits from these corrections.
- **Full-body man / astronaut point:** SAM 3's additional body-region clicks recover much of the full man, with small holes and a detached background fragment remaining. The current model's simultaneous refined clicks can still select only disconnected body parts; mask feedback improves coverage but remains incomplete. On the astronaut, both models' refined clicks cover much more of the person, while feedback introduces additional holes in some variants. No per-image winning mode is promoted into a general recipe.
- **Sphere, mug and phone:** all original sources and every variation remain in the gallery, including already successful cases. These are exploratory visual findings without object ground-truth scores.

See the [original point corrections](../data/sam3-refinement-2026-10-10/review/diagnostic-point-1.jpg), [poster/man corrections](../data/sam3-refinement-2026-10-10/review/diagnostic-point-2.jpg), [brooch text variations](../data/sam3-refinement-2026-10-10/review/diagnostic-text-1.jpg), and [poster text variations](../data/sam3-refinement-2026-10-10/review/diagnostic-text-2.jpg). The full gallery retains the remaining pages and all variations.

### Latency and resources

Medians, after excluded warm-up. GPU rows are a deployment option versus today's CPU implementation, not an equal-device comparison.

| Route | Current CPU initial / repeat | SAM 3 GPU initial / repeat |
|---|---:|---:|
| Single point, new image | 0.780 / 0.025 s | 0.119 / 0.011 s |
| Refined clicks, image embedding already retained | 0.043 / 0.025 s | 0.011 / 0.011 s |
| Refined clicks + initial mask logits, embedding retained | 0.026 / 0.026 s | 0.012 / 0.011 s |
| Box after point encoding | 0.026 / 0.026 s | 0.011 / 0.010 s |
| Text at production/reference threshold, new image | 4.056 / 3.307 s | 0.159 / 0.049 s |
| Text at fitted threshold, new image | 3.995 / 3.251 s | 0.159 / 0.049 s |

SAM 3 is about 6.6 times faster for warmed new-image point selection and 25.6 times faster for new-image text in this cohort. Refinement rows reuse the first point's embedding, so their small times must not be presented as standalone new-image selection times. These timings do not include the editor/API workflow.

| Evaluation process | Peak RAM working set | CUDA tensor / reservation peak | Local load / first warm-up |
|---|---:|---:|---:|
| Current SAM + DINO | 3.14 GiB | 0 / 0 | 2.10 / 3.32 s |
| SAM 3 tracker only | 2.59 GiB | 1.20 / 1.42 GiB | 1.21 / 15.16 s |
| SAM 3 detector only | 3.53 GiB | 2.12 / 2.38 GiB | 2.39 / 15.49 s |

RAM includes Python/Torch/mapped weights. CUDA peaks measure the tensors/allocator during post-warm evaluation, not all driver VRAM or first-load peaks. Routes run in separate processes, and loaded-Klein coexistence is untested. The first SAM 3 load plus inference still costs about 16–18 seconds before Python startup.

Checkpoint provenance and hardware match the [original report](SAM3_SELECTION_2026-10-10.md#checkpoint-and-reproduction): `facebook/sam3` revision `3c879f39826c281e95690f02c7821c4de09afae7`, PyTorch 2.14.1+cu130 / Transformers 5.19.0, Ryzen 7 9700X / RTX 5070 Ti 16 GB. Current cached SAM/DINO revisions were rechecked and match the original run. All five new SAM 3 loaders have no missing, unexpected or mismatched weights.

## Practical decision

SAM 3 should be evaluated as a fast concept-aware workflow with explicit refinement and retained mask state. Default one-click behavior, global low confidence thresholds, and automatic empty-background negative boxes are poor substitutes for user intent. The current CPU selector remains the app default: it already delivers strong refined masks, while a SAM 3 integration needs broader, annotation-complete object cases, meaningful negative concepts, soft cutout testing and loaded-Klein/runtime evidence. This stage improves the benchmark and identifies useful recipes; it does not change the app or prove a globally optimal configuration.

## Reproduction

From the repository root, using the existing backend environment and downloaded official SAM 3 checkpoint:

```powershell
backend/.venv/Scripts/python.exe backend/tools/prepare_selection_refinement.py --output data/sam3-refinement-new
backend/.venv/Scripts/python.exe backend/tools/probe_selection_refinement.py freeze --root data/sam3-refinement-new --baseline data/sam3-benchmark-2026-10-10/baseline-run2
backend/.venv/Scripts/python.exe backend/tools/benchmark_selection_refinement.py run --root data/sam3-refinement-new --output data/sam3-refinement-new/current-calibration --model current --split calibration
backend/.venv/Scripts/python.exe backend/tools/benchmark_selection_refinement.py run --root data/sam3-refinement-new --output data/sam3-refinement-new/sam3-text-calibration --model sam3-text --split calibration
backend/.venv/Scripts/python.exe backend/tools/benchmark_selection_refinement.py select --root data/sam3-refinement-new
backend/.venv/Scripts/python.exe backend/tools/benchmark_selection_refinement.py run --root data/sam3-refinement-new --output data/sam3-refinement-new/current-holdout --model current --split holdout
backend/.venv/Scripts/python.exe backend/tools/benchmark_selection_refinement.py run --root data/sam3-refinement-new --output data/sam3-refinement-new/sam3-point-holdout --model sam3-point --split holdout
backend/.venv/Scripts/python.exe backend/tools/benchmark_selection_refinement.py run --root data/sam3-refinement-new --output data/sam3-refinement-new/sam3-text-holdout --model sam3-text --split holdout
backend/.venv/Scripts/python.exe backend/tools/probe_selection_refinement.py run --root data/sam3-refinement-new --output data/sam3-refinement-new/current-diagnostic --model current
backend/.venv/Scripts/python.exe backend/tools/probe_selection_refinement.py run --root data/sam3-refinement-new --output data/sam3-refinement-new/sam3-point-diagnostic --model sam3-point
backend/.venv/Scripts/python.exe backend/tools/probe_selection_refinement.py run --root data/sam3-refinement-new --output data/sam3-refinement-new/sam3-text-diagnostic --model sam3-text
backend/.venv/Scripts/python.exe backend/tools/summarize_selection_refinement.py --root data/sam3-refinement-new
```

Outputs create new directories and refuse to overwrite completed runs or a frozen threshold choice. The dataset archive, source photos, annotations and generated artifacts remain under ignored `data/`. App data, temporary files and settings are isolated, and no editor database is initialized.

## Acceptance boundaries

This evaluation covers pedestrian segmentation and exploratory fixture recovery. It does not establish general object accuracy, native-package equivalence, soft cutout quality, shared residency with loaded Klein, or end-to-end editor latency. A default provider change requires evidence for those user-facing conditions. No backend restart, web rebuild or `mosimage.exe` rebuild is needed to use the benchmark results.

## Validation

The summarizer verified all 593 mask dimensions, binary values and hashes, rechecked frozen source/annotation hashes, and recomputed calibration/evaluation IoUs. All 276 evaluation repetitions match their initial mask exactly. Backend/app and all benchmark scripts pass Python compilation; the frontend TypeScript/Vite build passes. The local reference audit checks five documents, 96 file links, 18 Markdown anchors and 2,544 gallery asset references with no failures. The gallery JavaScript passes syntax checking. Playwright loaded all 636 gallery images with no broken assets and verified both mask/overlay toggles and their native-image links; see the [browser audit](../data/sam3-refinement-2026-10-10/review/browser-audit.json). Paired contact sheets and the browser render were inspected visually. Generated artifacts remain ignored, and there are no editor/provider source changes, deployment or executable rebuilds.
