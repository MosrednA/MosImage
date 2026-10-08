"""Test Anna identity with no guide, neutral pose guide and scene-layout guide."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path

from benchmark_park_base import BaseProtocol, BaseVariant, run_cohort
from benchmark_park_prompts import PARALLEL
from app.models.flux2_klein_nvfp4 import QUANT_REVISION

GUIDED = (
    "Place Anna from the studio reference photograph on a wooden garden bench in the selected part of the garden photograph. "
    "The studio photograph supplies her identity: same face, curly auburn hair, mustard sweater and dark green trousers. "
    "The extra image with the grey mannequin supplies only the seated pose and bench geometry. "
    "Replace that mannequin with Anna as a real photographed woman, keeping the guide's complete seated silhouette, head-to-seat proportions, "
    "parallel thighs, bent knees, downward shins and two feet resting flat side by side. "
    "Use the guide's relative size and placement for the figure and bench, with a complete head and both shoes inside the selection. "
    "Match the garden's lighting and perspective, with contact shadows beneath both shoes and the bench legs."
)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=args.input.resolve()
    assets=json.loads((root/"guides/manifest.json").read_text(encoding="utf-8"))["assets"]
    protocol=BaseProtocol(
        model="Klein Distilled NVFP4 identity/structure reference ablation", revision=QUANT_REVISION,
        prompt=PARALLEL, prompts={"plain":PARALLEL,"guided":GUIDED}, asset_hashes=assets,
        variants=(
            BaseVariant(name="guide-baseline",steps=8,reference_files=("guides/anna.png",)),
            BaseVariant(name="guide-pose",steps=8,prompt_key="guided",reference_files=("guides/anna.png","guides/pose.png")),
            BaseVariant(name="guide-layout",steps=8,prompt_key="guided",reference_files=("guides/anna.png","guides/layout.png"))
        ), per_render_seconds_limit=15,
        packages={name:version(name) for name in ("torch","diffusers","transformers","accelerate","safetensors","comfy-kitchen")}
    )
    run_cohort(root,args.output,protocol)


if __name__ == "__main__":
    main()
