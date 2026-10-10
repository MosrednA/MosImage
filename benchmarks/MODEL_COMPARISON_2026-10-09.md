# Model comparison and Klein 9B default — 9 October 2026

MosImage's Quality default is now **FLUX.2 Klein 9B distilled NVFP4**. This page records the comparison behind that change and how the 9B model fits on the 16 GB RTX 5070 Ti.

The Qwen arm below is the **Viggle community six-step FP8 model**. The [10 October evaluation](QWEN_TURBO_QUANTS_2026-10-10.md) tests the distinct official eight-step Turbo release with NVFP4 and GGUF; its lifecycle timings should not be confused with the render-only timings here.

## Comparison

[`backend/tools/benchmark_model_comparison.py`](../backend/tools/benchmark_model_comparison.py) renders twelve cases with three seeds (7, 42, 20261009) per arm: 216 outputs, all retained, none failed. Each arm runs in its own process. Prompts are encoded first with the full BF16 encoder, which is then released before the renderer loads.

Creative cases come first:

- two people from two references composed into a roller coaster and a ski hut;
- "turn this person into a DJ" as a whole-image edit;
- a phone swapped for a sword, both as a masked edit and as a whole-image edit.

The four v1 retouch scenes plus a shirt recolour, an Anna reference insert and a mug addition follow as controls. The new man and phone sources are generated locally by [`create_creative_fixtures.py`](../backend/tools/create_creative_fixtures.py); [fixtures/v2/manifest.json](fixtures/v2/manifest.json) records their hashes.

| Arm | Steps | Median | Range | Peak allocated |
| --- | --- | --- | --- | --- |
| Klein 4B NVFP4 (previous default) | 4 | 2.4 s | 0.4–4.7 s | 4.9 GiB |
| Klein 9B NVFP4 | 4 | 3.8 s | 0.6–8.1 s | 8.0 GiB |
| Qwen-Image 2.1 turbo (FP8) | 6 | 17.8 s | 9.1–89 s | 9.2 GiB |

Times exclude prompt encoding and model loading. Klein 9B takes about 1.6× as long as 4B in every case: 2.3 s against 1.5 s for the masked sword, 5.8–6.2 s against 3.6 s for the two-person scenes. Three Qwen renders took 59–89 s. Qwen needed `use_kv_cache=False`, VAE tiling and 5 GiB of encoder headroom to fit at all.

Visual findings:

- **9B is better than 4B** on the cases that matter most for creative edits. It gives a cleaner seam on the Anna insert, a more believable grip on the sword, and more coherent two-person scenes.
- **Qwen-Image is strong at whole-image edits but weak at masked work.** It made an oversized mug, one broken seed and visible seams around Anna. It is also five to seven times slower than Klein 9B.
- **Encoder variants make no material difference.** The abliterated Qwen3-4B and Qwen3-8B encoders and Qwen3-4B-Instruct-2507 produce near-identical images and timings to the stock encoders.
- **All models show a blue fringe in the shirt case.** This points to compositing or feathering rather than the model, and remains open.

The gallery, renders and per-arm JSON are kept locally under `data/model-comparison-2026-10-09/` (not committed). Rebuild them from `backend/`:

```powershell
.\.venv\Scripts\python.exe tools/benchmark_model_comparison.py --output ../data/model-comparison-2026-10-09 --arms all
.\.venv\Scripts\python.exe tools/benchmark_model_comparison.py --output ../data/model-comparison-2026-10-09 --gallery
```

## Fitting 9B on 16 GB

The 9B transformer is small in NVFP4, but its text encoder is Qwen3-8B, about 16 GB in BF16. Model CPU offload would move the encoder onto the card for every prompt and the transformer back and forth for every edit. [`klein_text_encoder.py`](../backend/app/models/klein_text_encoder.py) avoids both:

- **Truncation.** Klein reads hidden states 9, 18 and 27, so only the first 28 of 36 decoder layers are loaded; `lm_head` is skipped. Against the full BF16 model this is exact (relative error 0.0000 on all three layers).
- **FP8 projections.** Linear weights are stored as per-row scaled `float8_e4m3fn` and upcast per matmul. Relative embedding error against the full BF16 encoder is 0.6%, 0.96% and 3.7% on layers 9, 18 and 27, with cosine similarity near 1.
- **Streaming.** The weights are pinned in RAM (about 6 GB). For each new prompt they are copied to the GPU, used through `torch.func.functional_call` and released. The 1.2 GB token embedding stays on the CPU. One encode takes about 0.4 s (1.2 s the first time), and the prompt cache skips repeats.

[`Flux2Klein9bNvfp4Provider`](../backend/app/models/flux2_klein_nvfp4.py) loads the pinned [9B base](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) (revision `92196c8e11f7b6cf2b7493e037d8c5345c559216`) without its encoder and attaches the streamed one. It also loads the [official NVFP4 transformer](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-nvfp4) (revision `e882f64f6aa086fcf8915a7763550e05af10ef13`). Transformer and VAE stay resident on CUDA in every memory mode. The NVFP4 loader now accepts up to `8 × double + 2 × single` quantized modules: 9B ships 110, keeping the first block's QKV in BF16. The strict state-dict load still proves every weight is present.

## App validation

Measured through the real provider, with other desktop applications already using 5.7 GB of the card:

| Step | Time | Peak allocated | Card in use |
| --- | --- | --- | --- |
| Load + inpaint warm-up | 25.4 s | 10.9 GiB | 15.4 GB |
| Full-frame 832 × 1216 sword inpaint, new prompt | 7.8 s | 10.9 GiB | 15.2 GB |
| Same prompt, other seed (cached embedding) | 7.3 s | 7.5 GiB | 15.2 GB |
| 1024 × 1024 generation, new prompt | 3.2 s | 11.0 GiB | 14.8–15.2 GB |

The resident model set is 5.6 GiB, and the backend process uses about 10.5 GiB RAM. After unloading, the card returned to 3.7 GB and the process to 2.2 GiB RAM. This full-frame inpaint bypasses the app's context crop; in the app, a selection is cropped first, as the benchmark's 2.3 s masked sword case did. The three app renders (sword, ski hut, fox) used the FP8 encoder and looked equivalent to the benchmark's BF16-encoder renders. That is a spot check, not a paired evaluation.

## Licence

The Klein 9B weights are released for **non-commercial use only**; the app says so under Advanced controls. Klein 4B NVFP4 and 4B BF16 remain selectable for commercial work and for GPUs other than Blackwell.
