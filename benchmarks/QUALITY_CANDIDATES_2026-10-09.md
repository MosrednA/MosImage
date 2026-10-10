# LoRA, specialized outpaint and detail crops — 9 October 2026

Status: bounded evaluation completed: 78 cohort outputs and two additional LoRA equivalence renders. The 50-step Base kitchen arm was stopped by the user's latency constraint; one render was interrupted and five skipped. This follows the [mask/compositing implementation](EDIT_MASK_RESULTS_2026-10-09.md) and the [quality research snapshot](../docs/notes/2026-10-09-image-quality-research.md).

## Runtime implementation

`backend/app/models/klein_lora.py` applies a BF16 low-rank residual beside each original linear. It supports the two evaluated transformer-only ai-toolkit/PEFT exports, including fused QKV, input and modulation projections, through the pinned Diffusers 0.41 converter. It does not merge or dequantize NVFP4 weights, install PEFT, change text embeddings, or introduce an additional resident pipeline. Arbitrary adapters, DoRA, text-encoder adapters, adapter stacking and training are outside this implementation.

All targets and dimensions validate before attachment. Cancellation during preparation leaves the original modules intact. Scale zero returns the original linear result directly; detach restores the original objects. Finite weights, dtype overflow, shape mismatch, fused projections, cancellation, exact zero-scale equivalence and restoration have regression coverage. Original BFL final AdaLN adapters are rejected because their output-row order needs additional conversion; neither evaluated checkpoint contains that target.

The real 9B provider verified scale-zero and detach by rerendering seed 7: both match the original baseline pixel for pixel. Generation/edit and inpaint wrappers share the same adapted transformer. All 36 cohort renders executed 552 native NVFP4 linear calls each. The Consistency adapter adds 348,127,232 bytes (332 MiB) across 144 target linears.

The initial evaluation added a tested runtime component and a research runner. A subsequent editor integration exposes three optional switches, all initially off; see the integration section below. Existing cohort timings and visual findings remain the evidence from that initial evaluation.

## Fixed protocol

Run `backend/tools/benchmark_quality_candidates.py` from `backend/`, with the API model unloaded. Stages use separate processes. Original app documents, SQLite metadata and revisions are untouched. Assets, parameters, hashes, sampling settings, full-resolution PNGs, galleries and measurements remain under `data/quality-candidates-2026-10-09/`.

| Candidate | Pinned revision | Checkpoint |
|---|---|---|
| [Consistency V2](https://huggingface.co/dx8152/Flux2-Klein-9B-Consistency) | `8df0c7338cf68cfcd89ca7e461fe679905634607` | `Flux2-Klein-9B-consistency-V2.safetensors` |
| [FAL outpaint](https://huggingface.co/fal/flux-2-klein-4B-outpaint-lora) | `b11770ac6a3cf9325dcf81742c12b1c4e257880f` | `flux-outpaint-lora.safetensors` |
| [Klein Base 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) | `a3b4f4849157f664bdbc776fd7453c2783562f4d` | `transformer/diffusion_pytorch_model.safetensors` |

The 4B Base transformer uses the existing cached distilled 4B tokenizer, encoder, VAE and scheduler. Their configurations match; transformer configurations differ only in a training-path metadata field. Sampling explicitly sets `is_distilled=False`, 50 steps and guidance 4, following BFL's Base example. FAL's published SDK selects a Base edit endpoint despite the card's broader 4B label. Its exact instruction, green RGB `(0,255,0)`, and scale 1.1 are evaluated locally. Images are not sent to FAL.

Checkpoint SHA-256:

```text
Consistency V2 61db2017ce420b97bd5ef11984e5a894c90003a6bbf0dc9473f8d7b9ebb3ff93
FAL outpaint   9623927f2471613e0c9d7810da3c39f0b277a448193b223ea3cee67d5750974a
Base 4B        e109674697ffa1a3983126e32512f5428a9442bd8df59f9c95566ee90a473bb6
```

Seeds are 7, 42 and 20261009. 9B uses the actual app provider and RAM-streamed FP8 encoder, four steps and guidance 1. Consistency compares disabled, 0.6 and 1.0; no adapters are stacked. Disabled retains adapter buffers for the paired runs after the first unadapted baseline. Whole edits retain the app's approximately one-megapixel sampling and resize back to the source. Masked cases use current production workspaces and protected compositing.

Timing starts after initial pipeline loading and includes prompt encoding on a cache miss, inference, resizing and compositing. Workspace preparation, asset hashing, gallery construction and API/UI overhead are excluded. The first DJ baseline includes a cold prompt encode (23.77 seconds). Reported table values are medians across the three seeds, which suppress that single cold encode; they are not pure renderer times. PyTorch allocated/reserved memory excludes other Windows applications and driver allocations. The early Consistency protocol lacks package versions, and Consistency/Base protocols lack a runner source hash; subsequent fast/current/detail protocols record both. The cohort used the pinned installed dependencies and the same residual implementation throughout.

## Consistency V2: completed

36 images: DJ whole edit, phone-to-sword whole edit, selected shirt-to-sweater replacement and garden right expansion, with three weights and three seeds.

| Case | Disabled | Weight 0.6 | Weight 1.0 | Visual finding |
|---|---:|---:|---:|---|
| DJ | 7.74 s | 8.60 s | 8.59 s | Closer portrait framing and clothing preservation; stronger weight can suppress the requested hand/turntable interaction. |
| Phone → sword | 7.83 s | 8.68 s | 8.67 s | No consistent grip/anatomy improvement. Weight 1 introduces or retains fingers on the blade. |
| Shirt → knitted sweater | 7.49 s | 8.35 s | 8.34 s | More of the old shirt pockets/buttons/cuffs survive at stronger weights, weakening the requested material change. Blue edge residue remains. |
| Garden expansion | 4.80 s | 5.42 s | 5.42 s | Different vegetation, without a consistent seam-quality gain across all three seeds. |

Maximum allocated/reserved GPU memory was 11.294/11.445 GiB; protected pixels changed in none of the selected-region or expansion outputs. Whole edits have no protection guarantee. The lower-image diagnostic region in the sword edit has median RGB MAE 13.909 / 14.115 / 16.470 for disabled / 0.6 / 1.0. The LoRA therefore does not universally improve unchanged-region pixel fidelity. MAE is a drift diagnostic, not an identity or realism score.

Decision: keep disabled by default. V2 is a usable preservation experiment at moderate weight, with measurable edit-compliance tradeoffs. It is not a broad repair for masks, anatomy or outpaint seams. The author's V2 improvements are author claims; this cohort supplies narrower local evidence.

## Specialized outpaint

The Base4 recipe completed all six garden outputs at 157–180 seconds each. The adapter fills the green extension; disabled outputs retain green. Its latency exceeds the user's interactive requirement, so the kitchen arm was interrupted at step 21/50 and its remaining five calls skipped. All completed images and the cancellation record remain available. It is rejected as an interactive route on this hardware. Minimum available system RAM during the completed runs was 0.259 GiB; PyTorch allocated/reserved GPU maxima were 9.373/9.885 GiB.

The replacement evaluation completed the same green-border recipe on the app's distilled Fast4 NVFP4 model at four steps, disabled versus scale 1.1. This is a compatibility ablation, not the published Base recipe. Current 9B native masked expansion supplies a product-workflow control. Cases are garden right expansion and kitchen expansion on four sides, 256 native pixels per expanded side (about 20% of the 1254-pixel source). The working acceptance ceiling is 30 seconds for an ordinary edit after loading; model loading is recorded separately.

| Case | Current 9B | Fast4 green, disabled | Fast4 FAL 1.1 | Base4 green, disabled | Base4 FAL 1.1 |
|---|---:|---:|---:|---:|---:|
| Garden | 4.84 s | 6.24 s | 6.27 s | 163.12 s | 169.73 s |
| Kitchen | 8.56 s | 6.52 s | 6.63 s | Stopped | Skipped |

24 completed outputs, three seeds per populated cell. Fast4's first prompt took 26.111 seconds after 5.888 seconds of pipeline loading; Current9's first edit took 20.817 seconds after 19.580 seconds of loading. The shared green-border prompt stays cached for the kitchen. Fast4 ran 400 native NVFP4 calls per image; Current9 ran 552. The adapter remains a residual alongside the packed base weights.

Fast4 with FAL fills all six extensions without exact-green pixels. Disabled outputs retain substantial green or incomplete borders. At normal size the enabled garden and kitchen are plausible. Native seam crops still show texture/density transitions in the garden and varying cabinetry/material choices in the kitchen; the adapter does not consistently beat Current9 in every seed. This is a promising fast alternative to test on more real images, not a universal seam fix or an accepted app default.

Fast4's complete-run allocator maxima were 9.061/9.082 GiB, driven by the initial BF16 text encode; enabled warm images peaked at 4.481 GiB allocated. Current9 maxima were 10.947/11.002 GiB. Every completed outpaint result preserved all protected RGBA pixels, including the separately rechecked Base outputs.

The model and recipe both differ between Current9 and the green-border arms. Current9 uses a localized inpaint context; green-border editing uses the complete approximately one-megapixel canvas. Attribute a same-model disabled/enabled difference to the adapter; comparisons across model families are workflow comparisons.

## Targeted detail

18 completed outputs: native 1254×1254 architecture, marble countertop and blue denim-shirt fixtures. A selected contextual crop and a whole-image edit each sample about one megapixel with the same instruction and seed. Both are composited through the same selection and 16-pixel inward feather, so pixels outside it stay exact. The baseline's full whole-edit PNG also remains available to reveal unprotected changes. A 96-native-pixel context surrounds each rectangular selection; the portrait crop includes mirrored context below the canvas edge. The countertop selection includes the lower mug, so changes to its shape and reflection are also reviewed.

| Case | Whole edit | Context crop | Visual finding |
|---|---:|---:|---|
| Architecture | 8.27 s | 8.12 s | Crop retains more weathered brick structure in all three seeds; whole editing substitutes more regular bricks and changes the light. Crop also invents some pitting. |
| Countertop | 8.45 s | 8.09 s | Crop shows finer veins and surface texture, but two seeds add strong new gold/brown veins. Geometry, reflections and the mug remain candidates for drift. |
| Denim fabric | 7.92 s | 8.04 s | Crop resolves visible weave while retaining pocket/buttons/folds, but changes the mottled denim appearance and sometimes its colour. Whole edits remain closer in pixel values and softer. |

The crop concentrates roughly 1.8–2.0 times as many generated pixels per native linear dimension in these selections, while keeping the four-step baseline. Both routes have approximately the same compute cost. The first architecture whole edit took 24.054 seconds after 19.121 seconds of loading; the other 17 edits took 7.85–8.77 seconds. Allocated/reserved GPU maxima were 10.959/11.020 GiB; every render executed 552 native calls.

Selected-region median RGB MAE, whole/crop: architecture 19.744/10.233; countertop 8.793/7.966; denim 7.528/10.307. This only diagnoses change from the source: lower MAE does not prove more useful detail, and higher sharpness can mean invented texture. Native source/whole/crop sheets retain all three seeds for review.

Decision: selected crops are the stronger research direction for optional detail regeneration. They do not recover missing photographic truth and do not justify automatic enhancement, face restoration or increasing steps across all tools. Crop rendering through image-conditioned Edit is a research workflow here; it has not been added as a new editor action or substituted for native mask-conditioned Replace.

## Validation and retained evidence

142 backend tests, 77 frontend tests, the TypeScript/Vite build, backend compilation and benchmark syntax checks pass. The documentation audit resolved 101 local links and 19 heading links across nine changed documents; whitespace checks pass. The new adapter tests cover the independent residual calculation, packed-buffer preservation, QKV conversion, atomic invalid/cancelled attachment, finite/dtype validation, unsupported final AdaLN row order, disabling and restoring modules. No frontend behaviour changed in this candidate evaluation.

The final `data/quality-candidates-2026-10-09/audit.json` verifies all 78 cohort dimensions, three declared seeds per completed cell, 60 selected outputs with zero protected RGBA changes, and the two pixel-exact scale-zero/detach proofs. Other 18 outputs are unrestricted whole edits. The local Consistency/outpaint/detail `index.html` galleries link original assets and full-resolution images. Outpaint `*-seams-100percent.png` and detail `*-native-review.png` retain native-scale comparisons. `stopped-base4.json` records the explicit latency stop. Models and benchmark processes were unloaded at the end.

Fixtures are synthetic and the cohorts small. These results establish runtime compatibility and bounded visual findings, not a general quality success rate. Broader real-photo validation remains open. Editor integration and its separate acceptance checks are recorded below.

## Optional editor integration

Generation settings now exposes Consistency (V2, weight 0.6, distilled Quality 9B edits), Specialized outpaint (FAL at 1.1, distilled Fast4, four steps, guidance one) and Fine detail (selected Replace crop through Edit at full strength). All start off. Specialized outpaint temporarily selects Fast and restores the previous model/sampler when switched off. Fine detail retains the current Surrounding context, excludes Blend in and restores prior strength when disabled. Protected selection and expansion compositing remain active. The [guide](../docs/guide.md#runtime-modes) documents the switches and first-load costs. Production checkpoints use the pinned revisions and hashes above under ignored `models/adapters/`; normal jobs disable unused residuals, and model unload releases them.

Separate real editor acceptance uses an isolated database under `data/quality-toggles-ui-2026-10-09/`. Architecture detail with Consistency on and off, FAL garden expansion and ordinary Fast4 expansion after disabling FAL completed. All protected RGBA pixels stayed exact. Apply, Undo/Redo and Library reopening worked; both decoded exports exactly match their proposals. Export bytes were captured from the actual Export handler and uploaded into the test database, because the browser automation context closed on an ordinary download. The 1920 px desktop and 390 px Settings layouts were inspected with no horizontal overflow. Only one model remained loaded; regular Fast expansion derived the inpaint wrapper around the existing FAL-wrapped transformer and ran with its residual disabled.

The first jobs reached final compositing in about 36.1 s for cold 9B/Consistency/detail and 17.5 s for switching to FAL/Fast4. Warm detail without Consistency reached it in 8.0 s; ordinary Fast4 expansion after disabling FAL in 8.4 s. These measure submission-to-finishing-phase, excluding final composition. The original status timer kept counting after completed jobs; acceptance derives these values by subtracting final-phase elapsed time from total elapsed time. That timer is now corrected to freeze at completion, with regression coverage. Raw status snapshots and this timing limitation are retained in `proof.json`. These acceptance timings are separate from the renderer measurements above.

Two further warm FAL jobs reached final compositing in 7.4 and 7.6 s. The latter repeated the original right-256-pixel input and seed after ordinary inpaint had disabled the adapter; its output matches the initial FAL result pixel for pixel. All six editor generations preserve their protected RGBA pixels. The six-job proof, two captured exports and Settings screenshots remain in the ignored acceptance folder.

Integration validation: 153 backend tests, 80 frontend tests, backend compilation, TypeScript/Vite build and the Tauri release build pass. The documentation audit resolves 104 local links and 22 anchors across nine documents. `npm run tauri:build -- --no-bundle` updates the desktop shortcut's `frontend/src-tauri/target/release/mosimage.exe`; this does not create new installer bundles. Test servers and model processes are stopped after acceptance.

## Reproduction

The runner requires the ignored retained `data/model-comparison-2026-10-09/` inputs and masks used by the earlier comparison, plus the versioned v1 fixtures. Copying a clean checkout does not reproduce those ignored inputs automatically. Download the three checkpoints above into the matching `models/research-cache/<repository-name>/` folders, keeping the Base transformer's `transformer/` subdirectory and config. These checkpoints are optional and ignored by Git.

```powershell
cd backend
.venv/Scripts/python.exe tools/benchmark_quality_candidates.py consistency --output ../data/quality-candidates-2026-10-09/consistency
.venv/Scripts/python.exe tools/benchmark_quality_candidates.py outpaint --outpaint-arm fast4 --output ../data/quality-candidates-2026-10-09/outpaint
.venv/Scripts/python.exe tools/benchmark_quality_candidates.py outpaint --outpaint-arm current9 --output ../data/quality-candidates-2026-10-09/outpaint
.venv/Scripts/python.exe tools/benchmark_quality_candidates.py detail --output ../data/quality-candidates-2026-10-09/detail
```

Use a fresh output folder for a new comparison. Existing recorded results resume only when the arm's protocol matches; the runner rejects changed code or parameters in that folder. The historical Base arm can be selected with `--outpaint-arm base4`, but is unsuitable for interactive use and is not part of the accepted fast rerun. The galleries link full-resolution results; detail sheets crop to the declared region for direct inspection. Generated media and model weights are not committed.
