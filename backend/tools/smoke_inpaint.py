from __future__ import annotations

import argparse
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from app.compositing import prepare_inpaint_workspace
from app.config import settings
from app.models.flux2_klein_nvfp4 import Flux2Klein9bNvfp4Provider
from app.schemas import InpaintForm


class ConsoleJobControl:
    def raise_if_cancelled(self) -> None:
        return

    def set_progress(self, progress: float) -> None:
        print(f"progress={progress:.2f}", flush=True)

    def diffusion_callback(self, total_steps: int):
        def on_step_end(_pipeline: Any, step_index: int, _timestep: Any, callback_kwargs: dict[str, Any]):
            self.set_progress(0.15 + (0.75 * ((step_index + 1) / max(total_steps, 1))))
            return callback_kwargs

        return on_step_end


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an isolated one-image MosImage inpaint smoke test.")
    parser.add_argument("input", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()

    provider = Flux2Klein9bNvfp4Provider()
    control = ConsoleJobControl()
    started = time.perf_counter()
    try:
        with TemporaryDirectory(prefix="mosimage-smoke-") as temporary_dir:
            workspace = prepare_inpaint_workspace(
                args.input,
                args.mask,
                Path(temporary_dir),
                settings.inpaint_max_dimension,
                settings.inpaint_context_ratio,
                settings.inpaint_minimum_context,
            )
            print(
                f"native={workspace.source.size} crop={workspace.source.crop(workspace.crop_box).size} "
                f"model={_image_size(workspace.model_input_path)}",
                flush=True,
            )
            request = InpaintForm(
                prompt="Preserve the scene while making a subtle coherent edit",
                runtime_mode="balanced",
                steps=args.steps,
                guidance=1.0,
                strength=1.0,
            )
            provider.inpaint(
                request,
                workspace.model_input_path,
                workspace.model_mask_path,
                workspace.model_output_path,
                control,  # type: ignore[arg-type]
            )
            workspace.composite_to(args.output)
    finally:
        provider.unload()

    print(f"completed_seconds={time.perf_counter() - started:.2f} output={args.output}", flush=True)


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


if __name__ == "__main__":
    main()
