"""Generate the synthetic v2 creative-edit sources with local Klein 4B NVFP4. Run from backend/.

    python tools/create_creative_fixtures.py --candidates   # writes several seeds per source for review
    python tools/create_creative_fixtures.py                # writes the chosen seeds and the manifest

The chosen seeds are fixed below after review, so the fixtures are reproducible.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models.flux2_klein_nvfp4 import BASE_REPO, BASE_REVISION, QUANT_REPO, QUANT_REVISION

OUTPUT = Path(__file__).resolve().parents[2] / "benchmarks/fixtures/v2"
SOURCES = {
    "man": dict(width=832, height=1216, seed=23, prompt=(
        "Full-body studio photograph of a cheerful bald man in his fifties with a short grey beard and black rectangular glasses, "
        "wearing a red puffer jacket, blue jeans and brown hiking boots, standing relaxed and facing the camera. "
        "Plain light grey seamless background, soft even studio lighting, sharp focus, 50mm lens.")),
    "phone": dict(width=832, height=1216, seed=11, prompt=(
        "Photograph of a young man with short black hair wearing a dark green hoodie, standing on a sunny city street. "
        "He holds a black smartphone upright in his raised right hand at chest height and looks at its screen. "
        "His hand and the phone are clearly visible with empty space above them. Shallow depth of field, natural daylight, 35mm lens.")),
}
CANDIDATE_SEEDS = (11, 23, 37, 51)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", action="store_true")
    args = parser.parse_args()
    import torch
    from diffusers import Flux2KleinPipeline
    from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
    from nvfp4_transformer import load_transformer

    jobs = [(name, seed) for name, spec in SOURCES.items() for seed in (CANDIDATE_SEEDS if args.candidates else (spec["seed"],))]
    if any(seed is None for _, seed in jobs):
        raise ValueError("Choose seeds with --candidates first")
    encoder = Qwen3ForCausalLM.from_pretrained(BASE_REPO, revision=BASE_REVISION, subfolder="text_encoder", torch_dtype=torch.bfloat16,
                                               local_files_only=True, device_map={"": "cuda"})
    tokenizer = Qwen2TokenizerFast.from_pretrained(BASE_REPO, revision=BASE_REVISION, subfolder="tokenizer", local_files_only=True)
    text = Flux2KleinPipeline(scheduler=None, vae=None, text_encoder=encoder, tokenizer=tokenizer, transformer=None, is_distilled=True)
    with torch.inference_mode():
        embeddings = {name: text.encode_prompt(spec["prompt"], device=torch.device("cuda"))[0].cpu() for name, spec in SOURCES.items()}
    del text, encoder
    torch.cuda.empty_cache()
    pipe = Flux2KleinPipeline.from_pretrained(BASE_REPO, revision=BASE_REVISION, transformer=load_transformer(), text_encoder=None,
                                              tokenizer=None, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    target = OUTPUT / "candidates" if args.candidates else OUTPUT
    target.mkdir(parents=True, exist_ok=True)
    manifest = dict(model=QUANT_REPO, model_revision=QUANT_REVISION, base=BASE_REPO, base_revision=BASE_REVISION, steps=4, guidance=1.0, sources={})
    for name, seed in jobs:
        spec = SOURCES[name]
        with torch.inference_mode():
            image = pipe(prompt_embeds=embeddings[name].to("cuda"), width=spec["width"], height=spec["height"], num_inference_steps=4,
                         guidance_scale=1.0, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        path = target / (f"{name}-{seed}.png" if args.candidates else f"{name}.png")
        image.save(path)
        manifest["sources"][name] = dict(spec, seed=seed, file=path.name, sha256=sha256(path.read_bytes()).hexdigest())
        print(f"Wrote {path}", flush=True)
    if not args.candidates:
        (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
