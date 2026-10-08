"""Paired prompt ablation for Anna/park; run from backend with API unloaded.

Only prompt wording changes. The frozen source/mask/reference, seeds, 512 px
reference, context 32, eight steps and NVFP4 renderer remain fixed.
Choose explicit variants for a separately declared follow-up cohort.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

from benchmark_park_base import BaseProtocol, BaseVariant, run_cohort
from app.models.flux2_klein import _inpaint_prompt
from app.models.flux2_klein_nvfp4 import QUANT_REVISION
from app.schemas import InpaintForm


POSE = (
    "Anna, the adult woman in the reference photograph, is sitting comfortably on a wooden garden bench beside the gravel path. "
    "Her hips rest on the seat, her thighs extend forward along the seat, and her knees bend naturally so her lower legs descend toward the path. "
    "Both shoes rest flat on the gravel directly below her knees. Show her complete seated body, including both feet. "
    "Keep her face, curly auburn hair, mustard sweater and dark green trousers recognizable from the reference. "
    "The bench supports her weight. Match the garden photograph's camera angle, daylight and depth of field."
)
LAYOUT = (
    "Place Anna from the reference photograph seated on a wooden bench inside the editable part of this garden photograph. "
    "Use normal adult anatomy and a natural seated pose. Her seated figure occupies about two thirds of the crop height: "
    "the top of her head is near the upper fifth, her hips and the bench seat are near the middle, and the soles of her shoes are near nine tenths of the crop height. "
    "Her thighs lie on the seat, both knees bend downward, and both feet are visible resting on the gravel. "
    "Keep her entire head, seated body and shoes inside the editable region. Preserve her face, auburn curls, mustard sweater and green trousers. "
    "Use the source photograph's fixed viewpoint and lighting."
)
CONTACT = (
    "A photograph of Anna, the woman in the reference, sitting on a realistically sized wooden garden bench beside the path. "
    "Anna and the bench share the gravel path's ground plane and perspective. She sits firmly on the seat with her weight supported at her hips, "
    "thighs resting on the seat and knees bent comfortably. Her lower legs end in two visible shoes with their soles touching the gravel. "
    "The bench legs also meet the ground. Small soft contact shadows appear beneath the shoes and bench legs, matching the existing daylight. "
    "The bench has ordinary adult seating proportions relative to her body. Preserve her identity, curly auburn hair, mustard sweater and green trousers. "
    "Existing flowers naturally overlap the bench where appropriate while her shoes remain readable."
)


PARALLEL = (
    "Anna sits upright facing the camera on a wooden garden bench beside the gravel path. "
    "Both thighs rest parallel on the bench seat. Her knees are side by side and bent at right angles. "
    "Her shins point downward, with both white shoes planted side by side on the same patch of gravel below her knees. "
    "The bench seat is at her knee height. Show her entire seated figure from the top of her head to both shoe soles, with space above her head. "
    "Use natural adult body proportions and the garden's perspective. Keep her face, curly auburn hair, mustard sweater and dark green trousers from the reference. "
    "Small contact shadows connect both shoes and the bench legs to the gravel."
)
SHORT = (
    "The reference woman sits facing the camera on a wooden garden bench, feet together on the gravel path. "
    "Her knees bend at right angles; both thighs rest on the seat. Her complete seated body fits comfortably in the editable area. "
    "Preserve her face, auburn curls, yellow sweater and green trousers. Natural adult proportions, matching perspective, soft daylight and contact shadows beneath both shoes."
)
PROMPT_VARIANTS = {
    "prompt-baseline": "plain", "prompt-pose": "pose", "prompt-layout": "layout", "prompt-contact": "contact",
    "prompt-parallel": "parallel", "prompt-short": "short"
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=tuple(PROMPT_VARIANTS), default=list(PROMPT_VARIANTS)[:4])
    args = parser.parse_args()
    baseline = _inpaint_prompt(InpaintForm(prompt="sitting on a bench", blend_feather=0, context_padding=32), ["Anna"])
    available_prompts = {"plain": baseline, "pose": POSE, "layout": LAYOUT, "contact": CONTACT, "parallel": PARALLEL, "short": SHORT}
    prompts = {PROMPT_VARIANTS[name]: available_prompts[PROMPT_VARIANTS[name]] for name in args.variants}
    protocol = BaseProtocol(
        model="Klein Distilled NVFP4 prompt-only ablation", revision=QUANT_REVISION, prompt=baseline, prompts=prompts,
        variants=tuple(BaseVariant(name=name, prompt_key=PROMPT_VARIANTS[name], steps=8) for name in args.variants),
        per_render_seconds_limit=15.0,
        packages={name: version(name) for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors", "comfy-kitchen")}
    )
    run_cohort(args.input, args.output, protocol)


if __name__ == "__main__":
    main()
