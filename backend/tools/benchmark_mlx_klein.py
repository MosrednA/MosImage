"""Klein on Apple silicon: the BF16 PyTorch port against MLX 8-bit and 4-bit.

Every case runs through the app's own services (the code behind the API), so
masked edits go through the same crop, mask and compositing as in the editor,
references through the reference store and LoRAs through the providers. Each
arm runs in its own process with its own database and image folder under the
output directory; the app's data is never touched.

    cd backend
    .venv/bin/python tools/benchmark_mlx_klein.py --output ../data/mlx-klein-2026-10-11 --arms bf16,q8,q4
    .venv/bin/python tools/benchmark_mlx_klein.py --output ../data/mlx-klein-2026-10-11 --compare

Memory is the process's physical footprint (Activity Monitor's Memory column),
which on Apple silicon includes every Metal allocation of PyTorch and MLX. As
in the app, the GPU caches are released after every job.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
V1 = ROOT / "benchmarks/fixtures/v1"
V2 = ROOT / "benchmarks/fixtures/v2"
SEEDS = (7, 42)
GIB = 1024**3

DUO = ("Photograph of the woman from image 1 and the man from image 2 sitting together at a wooden table inside a cosy alpine "
       "ski hut, toasting with mugs of hot chocolate and laughing. Ski jackets hang on the wall, snowy mountains show through "
       "the window, warm lamplight and a crackling fireplace.")
DJ = ("Turn the woman in image 1 into a club DJ: she stands behind a DJ booth with turntables and a mixer, large over-ear "
      "headphones around her neck and one hand on a turntable, in a dark nightclub with colourful stage lights, haze and a "
      "dancing crowd behind her. Keep her face, curly hair and identity exactly the same.")
FOX = "A red fox sitting in a snowy birch forest at dawn, soft golden light, shallow depth of field, detailed fur."
MUG = ("Replace the terracotta ball with the exact golden yellow ceramic coffee mug shown in the reference photograph. Keep its "
       "smooth glossy glaze, round rim and single rounded handle on the right. Show the entire mug including its handle and "
       "base. It stands upright among the garden flowers, with natural daylight and a soft contact shadow.")
SWORD = ("A medieval steel longsword held upright in his raised right hand, gripped by its leather-wrapped hilt below a cross "
         "guard, the polished blade pointing straight up")

# name, model role, service, request fields, assets
CASES: list[dict] = [
    {"name": "fox", "role": "quality", "task": "generate", "title": "Text to image, 1024²",
     "request": {"prompt": FOX, "width": 1024, "height": 1024}},
    {"name": "duo", "role": "quality", "task": "generate", "title": "Two people from two references",
     "request": {"prompt": DUO, "width": 1216, "height": 832}, "references": ["woman.png", "man.png"]},
    {"name": "dj", "role": "quality", "task": "edit", "title": "Whole-picture edit", "image": "portrait.png",
     "request": {"prompt": DJ}},
    {"name": "dj-consistency", "role": "quality", "task": "edit", "title": "Whole-picture edit + Consistency LoRA",
     "image": "portrait.png", "request": {"prompt": DJ, "consistency_lora": True}},
    {"name": "garden", "role": "quality", "task": "inpaint", "title": "Masked replace", "image": "garden.png",
     "mask": "garden-mask.png", "fixture": "garden"},
    {"name": "garden-consistency", "role": "quality", "task": "inpaint", "title": "Masked replace + Consistency LoRA",
     "image": "garden.png", "mask": "garden-mask.png", "fixture": "garden", "request": {"consistency_lora": True}},
    {"name": "kitchen", "role": "quality", "task": "inpaint", "title": "Masked remove", "image": "kitchen.png",
     "mask": "kitchen-mask.png", "fixture": "kitchen"},
    {"name": "mug", "role": "quality", "task": "inpaint", "title": "Masked insert from a reference", "image": "garden.png",
     "mask": "mug-mask.png", "references": ["mug.png"], "request": {"prompt": MUG}},
    {"name": "enhance", "role": "quality", "task": "enhance", "title": "Enhance (detail LoRA, partial redraw)",
     "image": "man.png", "mask": "man-face-mask.png", "request": {"detail": 0.4}},
    {"name": "sword-fast", "role": "fast", "task": "inpaint", "title": "Fast: masked replace", "image": "phone.png",
     "mask": "sword-mask.png", "request": {"prompt": SWORD}},
    {"name": "outpaint-fast", "role": "fast", "task": "outpaint", "title": "Fast: Specialized outpaint (fal LoRA)",
     "image": "garden-small.png", "request": {"prompt": "the rest of the garden", "left": 256, "right": 256,
                                              "specialized_outpaint": True}},
]


# ------------------------------------------------------------------ memory

class _RUsage(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size",
        "phys_footprint", "proc_start_abstime", "proc_exit_abstime", "child_user_time", "child_system_time",
        "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins", "child_elapsed_abstime", "diskio_bytesread",
        "diskio_byteswritten", "cpu_time_qos_default", "cpu_time_qos_maintenance", "cpu_time_qos_background",
        "cpu_time_qos_utility", "cpu_time_qos_legacy", "cpu_time_qos_user_initiated", "cpu_time_qos_user_interactive",
        "billed_system_time", "serviced_system_time", "logical_writes", "lifetime_max_phys_footprint", "instructions",
        "cycles", "billed_energy", "serviced_energy", "interval_max_phys_footprint", "runnable_time")]


_LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")


def footprint() -> int:
    """The whole rusage_info_v4 structure, or proc_pid_rusage writes past its end."""
    info = _RUsage()
    if _LIBPROC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
        raise OSError("proc_pid_rusage failed")
    return info.phys_footprint


class PeakMeter:
    """Samples the footprint every 20 ms in the background."""

    def __init__(self) -> None:
        self._peak = footprint()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            self._peak = max(self._peak, footprint())
            time.sleep(0.02)

    def reset(self) -> None:
        self._peak = footprint()

    def peak(self) -> float:
        self._peak = max(self._peak, footprint())
        return round(self._peak / GIB, 2)


# ------------------------------------------------------------------ assets

def prepare(output: Path) -> None:
    from app.compositing import _inner_feather

    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name in ("garden", "kitchen", "portrait"):
        with Image.open(V1 / f"{name}.png") as image:
            image.convert("RGB").save(assets / f"{name}.png")
        with Image.open(V1 / f"{name}.mask.png") as mask:
            mask.save(assets / f"{name}-mask.png")
    with Image.open(assets / "portrait.png") as image:
        image.save(assets / "woman.png")
    for name in ("man", "phone"):
        with Image.open(V2 / f"{name}.png") as image:
            image.convert("RGB").save(assets / f"{name}.png")
    with Image.open(assets / "kitchen.png") as image:
        mug = image.crop((600, 510, 1060, 930))
        size = image.size
    mug.thumbnail((512, 512), Image.Resampling.LANCZOS)
    mug.save(assets / "mug.png")
    box = Image.new("L", size)
    box.paste(255, (210, 700, 655, 1075))
    _inner_feather(box, 16).save(assets / "mug-mask.png")
    sword = Image.new("L", (832, 1216))
    sword.paste(255, (470, 60, 800, 880))
    _inner_feather(sword, 16).save(assets / "sword-mask.png")
    face = Image.new("L", (832, 1216))
    face.paste(255, (256, 64, 608, 512))
    _inner_feather(face, 16).save(assets / "man-face-mask.png")
    with Image.open(assets / "garden.png") as image:
        image.resize((768, 768), Image.Resampling.LANCZOS).save(assets / "garden-small.png")
    hashes = {path.name: sha256(path.read_bytes()).hexdigest() for path in sorted(assets.iterdir())}
    (output / "assets.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ one arm

class _Manager:
    def is_cancel_requested(self, job_id: str) -> bool:
        return False

    def update_progress(self, *args) -> None: ...
    def update_phase(self, *args) -> None: ...
    def update_step(self, *args) -> None: ...


def run_arm(output: Path, arm: str, only: set[str] | None) -> None:
    folder = output / arm
    data = folder / "data"
    # mlx-bf16 is a control: the MLX runtime without quantization, which isolates
    # what the implementation changes from what 8 or 4 bits change.
    os.environ.update({"MOSIMAGE_APPLE_WEIGHTS": "q8" if arm == "mlx-bf16" else arm, "MOSIMAGE_DATA_DIR": str(data), "MOSIMAGE_DB_PATH": str(data / "app.db"),
                       "MOSIMAGE_IMAGE_DIR": str(data / "images"), "MOSIMAGE_WARM_UP_ON_START": "false"})
    meter = PeakMeter()
    from app import documents, references, storage
    from app.config import settings
    from app.enhance import run_enhance
    from app.jobs import JobControl
    from app.models.registry import registry
    from app.runtime import release_cached_memory
    from app.schemas import EditRequest, EnhanceRequest, InpaintForm, LoadModelRequest, OutpaintForm, TextToImageRequest
    from app.services import run_image_edit, run_inpaint, run_outpaint, run_text_to_image

    assert settings.apple_silicon, "Run this on a Mac."
    if arm == "mlx-bf16":
        from app.models import klein_mlx
        for name in ("load_transformer", "load_text_encoder"):
            load = getattr(klein_mlx, name)
            setattr(klein_mlx, name, lambda folder, _precision, check, load=load: load(folder, "bf16", check))
    storage.ensure_storage()
    documents.ensure_document_storage()
    references.ensure_reference_storage()
    control = JobControl(_Manager(), "benchmark")
    assets = output / "assets"
    suite = {f["name"]: f for f in json.loads((V1 / "manifest.json").read_text())["fixtures"]}
    reference_ids = {name: references.create_reference(None, (assets / name).read_bytes(), name)["id"]
                     for name in ("woman.png", "man.png", "mug.png")}
    results: dict = {"arm": arm, "loads": {}, "renders": []}
    results_path = folder / "results.json"
    folder.mkdir(parents=True, exist_ok=True)
    renders = folder / "renders"
    renders.mkdir(exist_ok=True)

    for role in ("quality", "fast"):
        cases = [case for case in CASES if case["role"] == role and (not only or case["name"] in only)]
        if not cases:
            continue
        model = settings.quality_model if role == "quality" else settings.fast_model
        meter.reset()
        start = time.perf_counter()
        registry.preload(LoadModelRequest(model_name=model, task="inpaint", warm_up=True), control)
        release_cached_memory()  # The job runner does this after every GPU job.
        pipe = next(iter(registry.get(model)._pipes.values()))
        results["loads"][role] = {
            "model": model, "seconds": round(time.perf_counter() - start, 1), "peak_gib": meter.peak(),
            "resident_gib": round(footprint() / GIB, 2), **_pools(),
            "transformer_gib": round(_storage(pipe.transformer) / GIB, 2),
            "text_encoder_gib": round(_storage(pipe.text_encoder) / GIB, 2),
        }
        del pipe  # The provider alone must hold the model, or unloading it frees nothing.
        print(json.dumps({"load": results["loads"][role]}), flush=True)
        for case in cases:
            for seed in SEEDS:
                fields = {"seed": seed, "model_name": model, **case.get("request", {})}
                ids = [reference_ids[name] for name in case.get("references", ())]
                meter.reset()
                start = time.perf_counter()
                try:
                    if case["task"] == "generate":
                        generated = run_text_to_image(TextToImageRequest(**fields, reference_ids=ids), control)
                    elif case["task"] == "edit":
                        generated = run_image_edit(EditRequest(**fields, reference_ids=ids),
                                                   (assets / case["image"]).read_bytes(), case["image"], control)
                    elif case["task"] == "inpaint":
                        if "fixture" in case:
                            fixture = suite[case["fixture"]]
                            fields = {"prompt": fixture["prompt"], "edit_operation": fixture["operation"], **fields}
                        generated = run_inpaint(InpaintForm(**fields, reference_ids=ids), (assets / case["image"]).read_bytes(),
                                                case["image"], (assets / case["mask"]).read_bytes(), case["mask"], control)
                    elif case["task"] == "outpaint":
                        generated = run_outpaint(OutpaintForm(**fields), (assets / case["image"]).read_bytes(), control)
                    else:
                        generated = run_enhance(EnhanceRequest(**fields), (assets / case["image"]).read_bytes(), case["image"],
                                                (assets / case["mask"]).read_bytes(), control)
                    seconds = time.perf_counter() - start
                    record = storage.get_generation(generated[0])
                    target = renders / f"{case['name']}-{seed}.png"
                    target.write_bytes(storage.absolute_path(record["output_image_path"]).read_bytes())
                    row = {"case": case["name"], "seed": seed, "seconds": round(seconds, 2), "peak_gib": meter.peak(),
                           "output": str(target.relative_to(output))}
                except Exception as error:  # Retained: a failure is a result too.
                    row = {"case": case["name"], "seed": seed, "error": f"{type(error).__name__}: {error}"}
                finally:
                    release_cached_memory()
                if "output" in row:
                    row.update(after_gib=round(footprint() / GIB, 2), **_pools())
                print(json.dumps(row), flush=True)
                results["renders"].append(row)
                results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        results["loads"][role]["after_jobs_gib"] = round(footprint() / GIB, 2)
        results["loads"][role]["after_jobs_pools"] = _pools()
    registry.unload_all()
    results["after_unload_gib"] = round(footprint() / GIB, 2)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def _pools() -> dict[str, float]:
    """Where the footprint sits: PyTorch's Metal allocations and MLX's live and cached buffers."""
    import torch

    pools = {"mps_driver_gib": round(torch.mps.driver_allocated_memory() / GIB, 2)}
    mlx = sys.modules.get("mlx.core")
    if mlx is not None:
        pools.update(mlx_active_gib=round(mlx.get_active_memory() / GIB, 2), mlx_cache_gib=round(mlx.get_cache_memory() / GIB, 2))
    return pools


def _storage(module) -> int:
    if hasattr(module, "storage_bytes"):
        return module.storage_bytes
    inner = getattr(module, "inner", module)
    return sum(t.numel() * t.element_size() for t in (*inner.parameters(), *inner.buffers()))


# ------------------------------------------------------------------ comparison

def _ssim(first: np.ndarray, second: np.ndarray) -> float:
    """Mean SSIM on luminance with an 8 × 8 box window (the plain Wang et al. formula)."""
    import torch
    from torch.nn import functional as F

    def luma(image: np.ndarray) -> torch.Tensor:
        rgb = torch.from_numpy(image.astype(np.float64))
        return (rgb @ torch.tensor([0.299, 0.587, 0.114], dtype=torch.float64))[None, None]

    x, y = luma(first), luma(second)
    pool = lambda t: F.avg_pool2d(t, 8, 1)  # noqa: E731
    mx_, my = pool(x), pool(y)
    vx, vy, cov = pool(x * x) - mx_**2, pool(y * y) - my**2, pool(x * y) - mx_ * my
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = ((2 * mx_ * my + c1) * (2 * cov + c2)) / ((mx_**2 + my**2 + c1) * (vx + vy + c2))
    return float(ssim.mean())


def _region(case: dict, output: Path, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """The part of the picture a case changes: the selection's bounding box, or the added borders."""
    if case["task"] in ("inpaint", "enhance"):
        with Image.open(output / "assets" / case["mask"]) as mask:
            band = mask.getchannel("A") if "A" in mask.getbands() else mask.convert("L")
            box = band.point(lambda value: 255 if value > 8 else 0).getbbox()
        if box is None or band.size != size:
            return None
        return box
    return None


def _metrics(first: np.ndarray, second: np.ndarray) -> tuple[float | None, float]:
    mse = float(((first.astype(np.float64) - second) ** 2).mean())
    return (round(10 * np.log10(255**2 / mse), 2) if mse else None), round(_ssim(first, second), 4)


def compare(output: Path, arms: list[str]) -> None:
    results = {arm: json.loads((output / arm / "results.json").read_text()) for arm in arms}
    cases = {case["name"]: case for case in CASES}
    baseline = {(r["case"], r["seed"]): r for r in results["bf16"]["renders"]}
    rows = []
    for arm in arms:
        for render in results[arm]["renders"]:
            row = {"arm": arm, **render}
            reference = baseline.get((render["case"], render["seed"]))
            if arm != "bf16" and "output" in render and reference and "output" in reference:
                with Image.open(output / render["output"]) as a, Image.open(output / reference["output"]) as b:
                    first, second = np.asarray(a.convert("RGB")), np.asarray(b.convert("RGB"))
                row["psnr_vs_bf16"], row["ssim_vs_bf16"] = _metrics(first, second)
                # Masked edits keep every pixel outside the selection, which flatters whole-picture
                # numbers; measure the selection's box as well.
                box = _region(cases[render["case"]], output, (first.shape[1], first.shape[0]))
                if box is not None:
                    left, top, right, bottom = box
                    row["region_psnr_vs_bf16"], row["region_ssim_vs_bf16"] = _metrics(first[top:bottom, left:right],
                                                                                      second[top:bottom, left:right])
            rows.append(row)
    summary = {"loads": {arm: results[arm]["loads"] for arm in arms}, "renders": rows}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["loads"], indent=2))
    for case in dict.fromkeys(r["case"] for r in rows):
        line = [case]
        for arm in arms:
            mine = [r for r in rows if r["arm"] == arm and r["case"] == case]
            seconds = [r["seconds"] for r in mine if "seconds" in r]
            key = "region_" if any("region_psnr_vs_bf16" in r for r in mine) else ""
            psnr = [r[f"{key}psnr_vs_bf16"] for r in mine if r.get(f"{key}psnr_vs_bf16")]
            ssim = [r[f"{key}ssim_vs_bf16"] for r in mine if f"{key}ssim_vs_bf16" in r]
            errors = [r for r in mine if "error" in r]
            line.append(f"{arm}: {np.median(seconds) if seconds else '-'} s, peak {max((r['peak_gib'] for r in mine if 'peak_gib' in r), default='-')} GiB"
                        + (f", {'selection ' if key else ''}PSNR {np.mean(psnr):.1f} dB, SSIM {np.mean(ssim):.3f}" if psnr else "")
                        + (f", {len(errors)} failed" if errors else ""))
        print(" | ".join(map(str, line)))
    gallery(output, arms, rows)


def gallery(output: Path, arms: list[str], rows: list[dict]) -> None:
    """A local HTML page with every render side by side (JPEG thumbnails)."""
    thumbs = output / "thumbs"
    thumbs.mkdir(exist_ok=True)
    titles = {case["name"]: case["title"] for case in CASES}
    html = ["<!doctype html><meta charset=utf-8><title>Klein on Apple silicon: BF16 against MLX</title>",
            "<style>body{font:14px system-ui;margin:16px;background:#111;color:#eee}td{vertical-align:top;padding:4px}"
            "img{width:300px}h2{margin-top:28px}small{color:#aaa}</style><h1>Klein on Apple silicon: BF16 against MLX</h1>"]
    for case in dict.fromkeys(r["case"] for r in rows):
        html.append(f"<h2>{titles.get(case, case)}</h2><table><tr>" + "".join(f"<th>{arm}</th>" for arm in arms) + "</tr>")
        for seed in SEEDS:
            html.append("<tr>")
            for arm in arms:
                row = next((r for r in rows if r["arm"] == arm and r["case"] == case and r["seed"] == seed), None)
                if row is None or "output" not in row:
                    html.append(f"<td>{row.get('error', 'missing') if row else 'missing'}</td>")
                    continue
                thumb = thumbs / f"{arm}-{case}-{seed}.jpg"
                with Image.open(output / row["output"]) as image:
                    image = image.convert("RGB")
                    image.thumbnail((640, 640), Image.Resampling.LANCZOS)
                    image.save(thumb, quality=85)
                key = "region_" if "region_psnr_vs_bf16" in row else ""
                note = f"{row['seconds']} s" + (f", {'selection ' if key else ''}PSNR {row[key + 'psnr_vs_bf16']} dB, SSIM {row[key + 'ssim_vs_bf16']}"
                                                  if f"{key}psnr_vs_bf16" in row else "")
                html.append(f"<td><a href='{row['output']}'><img src='thumbs/{thumb.name}'></a><br><small>seed {seed}: {note}</small></td>")
            html.append("</tr>")
        html.append("</table>")
    (output / "gallery.html").write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "data/mlx-klein-2026-10-11")
    parser.add_argument("--arms", default="bf16,mlx-bf16,q8,q4", help="comma-separated: bf16, mlx-bf16 (control), q8, q4")
    parser.add_argument("--arm", help=argparse.SUPPRESS)  # internal: run one arm in this process
    parser.add_argument("--only", help="comma-separated case names")
    parser.add_argument("--compare", action="store_true", help="compare finished arms and write the gallery")
    args = parser.parse_args()
    output = args.output.resolve()
    arms = args.arms.split(",")
    only = set(args.only.split(",")) if args.only else None
    if args.arm:
        run_arm(output, args.arm, only)
    elif args.compare:
        compare(output, arms)
    else:
        if not (output / "assets.json").is_file():
            prepare(output)
        for arm in arms:
            # One process per arm: clean memory numbers, and the setting is read at import.
            command = [sys.executable, __file__, "--output", str(output), "--arm", arm] + (["--only", args.only] if args.only else [])
            subprocess.run(command, check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
        compare(output, arms)


if __name__ == "__main__":
    main()
