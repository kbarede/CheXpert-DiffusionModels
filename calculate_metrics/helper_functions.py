"""
Functions to compare generated chest-X-ray with FID and FRDv1.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, Sequence
import torch


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VALID_RESIZE_MODES = {"stretch", "pad", "crop"}
VALID_DEVICES = {"auto", "cpu", "cuda"}


def default_frd_workers() -> int:
    """Use Slurm CPU if available"""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try: return max(1, int(slurm_cpus))
        except ValueError: pass
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def find_images(directory: str | Path) -> list[Path]:
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    paths = sorted(
    path
    for path in directory.rglob("*")
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
    and not any(
        part.startswith(".")
        for part in path.relative_to(directory).parts
    ))

    if not paths:
        allowed = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise RuntimeError(f"No supported images found in {directory} ({allowed})")
    return paths


def choose_images( paths: Sequence[Path], count: int, seed: int, group_name: str) -> list[Path]:
    """Select fixed random subset from one image source -  number of selected image would match SAMPLE_SIZE"""
    if len(paths) < count:
        raise RuntimeError(
            f"{group_name} contains {len(paths)} images, but {count} are required."
        )
    selected = list(paths)
    random.Random(seed).shuffle(selected)
    return selected[:count]


def choose_real_subsets(
    paths: Sequence[Path],
    count: int,
    seed: int,
    calculate_real_baseline: bool,
) -> tuple[list[Path], list[Path] | None]:
    """slecte the fixed real reference (set A) and an optional disjoint baseline set (set B)"""
    if len(paths) < count:
        raise RuntimeError(
            f"The real test folder contains {len(paths)} images, but {count} are required."
        )

    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    reference = shuffled[:count]

    if calculate_real_baseline and len(paths) >= 2 * count:
        return reference, shuffled[count : 2 * count]

    if calculate_real_baseline:
        print(
            "Warning: real-vs-real baseline skipped; it needs "
            f"{2 * count} disjoint real images, but only {len(paths)} were found.",
            file=sys.stderr,
        )
    return reference, None


def resize_grayscale(image, size: int, mode: str):
    from PIL import Image, ImageOps

    image = image.convert("L")
    target = (size, size)
    resampling = Image.Resampling.BILINEAR

    if mode == "stretch":
        return image.resize(target, resample=resampling)
    if mode == "pad":
        return ImageOps.pad(image, target, method=resampling, color=0)
    if mode == "crop":
        return ImageOps.fit(image, target, method=resampling, centering=(0.5, 0.5))
    raise ValueError(f"Unknown resize mode: {mode}")


def standardize_images(
    source_paths: Sequence[Path],
    destination: Path,
    size: int,
    resize_mode: str,
) -> list[Path]:
    from PIL import Image

    destination.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    constant_images: list[Path] = []

    for index, source in enumerate(source_paths):
        output = destination / f"{index:05d}.png"
        try:
            with Image.open(source) as image:
                image.load()
                processed = resize_grayscale(image, size, resize_mode)
                minimum, maximum = processed.getextrema()
                if minimum == maximum:
                    constant_images.append(source)
                processed.save(output, format="PNG", compress_level=1)
        except Exception as exc:
            raise RuntimeError(f"Could not read image {source}: {exc}") from exc
        output_paths.append(output)

    if constant_images:
        examples = ", ".join(str(path) for path in constant_images[:3])
        print(
            f"Warning: {len(constant_images)} constant-intensity images found "
            f"(examples: {examples}). They are retained in the evaluation.",
            file=sys.stderr,
        )
    return output_paths


class StandardizedImageDataset:
    def __init__(self, paths: Sequence[Path]):
        self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import numpy as np
        import torch
        from PIL import Image

        with Image.open(self.paths[index]) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor.repeat(3, 1, 1)


def image_batches(paths: Sequence[Path], batch_size: int, workers: int, pin_memory: bool) -> Iterable:
    from torch.utils.data import DataLoader

    return DataLoader(
        StandardizedImageDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )


def feed_fid(metric, paths: Sequence[Path], real: bool, device, batch_size: int, workers: int) -> None:
    loader = image_batches(paths, batch_size=batch_size, workers=workers, pin_memory=device.type == "cuda")
    with torch.inference_mode():
        for batch in loader:
            metric.update(batch.to(device, non_blocking=True), real=real)


def compute_all_fid(
    reference_paths: Sequence[Path],
    comparison_paths: dict[str, Sequence[Path]],
    batch_size: int,
    workers: int,
    device_name: str,
) -> tuple[dict[str, float], str]:
    try:
        import torch
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as exc:
        raise RuntimeError(
            "FID dependencies are missing"
        ) from exc

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE='cuda', but CUDA is unavailable.")
    device = torch.device(device_name)
    metric = FrechetInceptionDistance(
        feature=2048,
        normalize=True,
        reset_real_features=False,
    ).set_dtype(torch.float64)
    metric = metric.to(device)

    print(f"Extracting real FID features on {device} ...")
    feed_fid(metric, reference_paths, True, device, batch_size, workers)

    scores: dict[str, float] = {}
    for name, paths in comparison_paths.items():
        print(f"Calculating FID for {name} ...")
        feed_fid(metric, paths, False, device, batch_size, workers)
        scores[name] = float(metric.compute().detach().cpu().item())
        metric.reset()

    return scores, str(device)


def compute_all_frd(
    reference_paths: Sequence[Path],
    comparison_paths: dict[str, Sequence[Path]],
    workers: int,
    verbose: bool,
    use_paper_log: bool,
) -> dict[str, float]:
    if sys.version_info < (3, 10):
        raise RuntimeError("The current official frd-score package requires Python >= 3.10")
    try:
        from frd_score import compute_frd
    except ImportError as exc:
        raise RuntimeError("FRD dependencies are missing.") from exc

    real_first = [str(path) for path in reference_paths]
    scores: dict[str, float] = {}
    for name, paths in comparison_paths.items():
        print(f"Calculating FRDv1 for {name} ...")
        value = compute_frd(
            [real_first, [str(path) for path in paths]],
            frd_version="v1",
            norm_ref="d1",
            num_workers=workers,
            verbose=verbose,
            use_paper_log=use_paper_log,
        )
        scores[name] = float(value)
    return scores


def serializable_number(value: float):
    return value if math.isfinite(value) else str(value)


def write_outputs(
    output_dir: Path,
    settings: dict,
    selected_originals: dict[str, Sequence[Path]],
    fid_scores: dict[str, float],
    frd_scores: dict[str, float],
) -> tuple[dict, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "selected_files.json"
    json_path = output_dir / "scores.json"
    csv_path = output_dir / "scores.csv"

    manifest = {
        name: [str(path.resolve()) for path in paths]
        for name, paths in selected_originals.items()
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    results = {
        "settings": settings,
        "scores": {
            name: {
                "fid": serializable_number(fid_scores[name]),
                "frd": serializable_number(frd_scores[name]),
                "n_reference": len(selected_originals["real_reference"]),
                "n_comparison": len(selected_originals[name]),
            }
            for name in fid_scores
        },
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("comparison", "fid", "frd", "n_reference", "n_comparison"),
        )
        writer.writeheader()
        for name, score in results["scores"].items():
            writer.writerow({"comparison": name, **score})

    return results, json_path, csv_path, manifest_path


def print_summary(fid_scores: dict[str, float], frd_scores: dict[str, float]) -> None:
    print("\nResults")
    print(f"{'Comparison':<22} {'FID':>14} {'FRDv1':>14}")
    print("-" * 52)
    for name in fid_scores:
        print(f"{name:<22} {fid_scores[name]:>14.6f} {frd_scores[name]:>14.6f}")

    model_names = [name for name in ("ddpm_vs_real", "ncsn_vs_real") if name in fid_scores]
    if len(model_names) == 2:
        best_fid = min(model_names, key=lambda name: fid_scores[name])
        best_frd = min(model_names, key=lambda name: frd_scores[name])
        print(f"\nLower FID: {best_fid.replace('_vs_real', '').upper()}")
        print(f"Lower FRD: {best_frd.replace('_vs_real', '').upper()}")


def validate_settings(
    sample_size: int,
    image_size: int,
    resize_mode: str,
    batch_size: int,
    loader_workers: int,
    frd_workers: int | None,
    device: str,
) -> None:
    for name, value in {
        "SAMPLE_SIZE": sample_size,
        "IMAGE_SIZE": image_size,
        "BATCH_SIZE": batch_size,
    }.items():
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if not isinstance(loader_workers, int) or loader_workers < 0:
        raise ValueError("LOADER_WORKERS must be zero or greater.")
    if frd_workers is not None and (
        not isinstance(frd_workers, int) or frd_workers < 1
    ):
        raise ValueError("FRD_WORKERS must be None or a positive integer.")
    if resize_mode not in VALID_RESIZE_MODES:
        raise ValueError(
            f"RESIZE_MODE must be one of {sorted(VALID_RESIZE_MODES)}, "
            f"got {resize_mode!r}."
        )
    if device not in VALID_DEVICES:
        raise ValueError(f"DEVICE must be one of {sorted(VALID_DEVICES)}, got {device!r}.")


def run_evaluation(
    *,
    real_dir: str | Path,
    ddpm_dir: str | Path,
    ncsn_dir: str | Path,
    ncsn_ema_dir: str | Path,
    ext_ncsn_dir: str | Path,
    ext_ncsn_ema_dir: str | Path,
    output_dir: str | Path,
    sample_size: int = 1000,
    image_size: int = 128,
    resize_mode: str = "stretch",
    batch_size: int = 32,
    loader_workers: int = 2,
    frd_workers: int | None = None,
    device: str = "auto",
    seed: int = 2026,
    calculate_real_baseline: bool = True,
    frd_paper_log: bool = True,
    verbose_frd: bool = False,
) -> dict:
    validate_settings(
        sample_size,
        image_size,
        resize_mode,
        batch_size,
        loader_workers,
        frd_workers,
        device,
    )
    if frd_workers is None:
        frd_workers = default_frd_workers()

    real_all = find_images(real_dir)
    ddpm_all = find_images(ddpm_dir)
    ncsn_all = find_images(ncsn_dir)
    ncsn_ema_all = find_images(ncsn_ema_dir)
    ext_ncsn_all = find_images(ext_ncsn_dir)
    ext_ncsn_ema_all = find_images(ext_ncsn_ema_dir)

    print(
        "Images found: "
        f"real={len(real_all)}, "
        f"DDPM={len(ddpm_all)}, " 
        f"NCSN={len(ncsn_all)}, " 
        f"NCSN_EMA={len(ncsn_ema_all)}, " 
        f"EXT_NCSN={len(ext_ncsn_all)}, " 
        f"EXT_NCSN_EMA={len(ext_ncsn_ema_all)}"
    )

    print(
        "Images Path: "
        f"real={real_dir}\n" 
        f"DDPM={ddpm_dir}\n" 
        f"NCSN={ncsn_dir}\n" 
        f"NCSN_EMA={ncsn_ema_dir}\n"  
        f"EXT_NCSN={ext_ncsn_dir}\n" 
        f"EXT_NCSN_EMA={ext_ncsn_ema_dir}"
    )

    real_reference, real_baseline = choose_real_subsets(
        real_all,
        sample_size,
        seed,
        calculate_real_baseline,
    )

    ddpm = choose_images(ddpm_all, sample_size, seed + 1, "DDPM")
    ncsn = choose_images(ncsn_all, sample_size, seed + 2, "NCSN")
    ncsn_ema = choose_images(ncsn_ema_all, sample_size, seed + 3, "NCSN-EMA")
    ext_ncsn = choose_images(ext_ncsn_all, sample_size, seed + 4, "extNCSN")
    ext_ncsn_ema = choose_images(ext_ncsn_ema_all, sample_size, seed + 5, "extNCSN-EMA")

    selected_originals: dict[str, Sequence[Path]] = {
        "real_reference": real_reference,
        "ddpm_vs_real": ddpm,
        "ncsn_vs_real": ncsn,
        "ncsnEma_vs_real": ncsn_ema,
        "extNcsn_vs_real": ext_ncsn,
        "extNcsnEma_vs_real": ext_ncsn_ema,
    }
    if real_baseline is not None:
        selected_originals["real_vs_real"] = real_baseline

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="cxr_fid_frd_") as temp_name:
        temp_root = Path(temp_name)
        standardized: dict[str, list[Path]] = {}
        print(
            f"Standardizing all sets to grayscale {image_size}x{image_size} "
            f"with resize_mode={resize_mode} ..."
        )
        for name, paths in selected_originals.items():
            standardized[name] = standardize_images(
                paths,
                temp_root / name,
                image_size,
                resize_mode,
            )

        comparison_paths = {
            name: paths
            for name, paths in standardized.items()
            if name != "real_reference"
        }
        fid_scores, actual_device = compute_all_fid(
            standardized["real_reference"],
            comparison_paths,
            batch_size=batch_size,
            workers=loader_workers,
            device_name=device,
        )
        frd_scores = compute_all_frd(
            standardized["real_reference"],
            comparison_paths,
            workers=frd_workers,
            verbose=verbose_frd,
            use_paper_log=frd_paper_log,
        )

    elapsed_seconds = time.perf_counter() - started
    settings = {
        "sample_size": sample_size,
        "image_size": image_size,
        "resize_mode": resize_mode,
        "seed": seed,
        "fid_feature_dimension": 2048,
        "fid_feature_extractor": "ImageNet Inception-v3",
        "frd_version": "v1",
        "frd_normalization_reference": "held-out real (d1)",
        "frd_log_convention": (
            "paper: log(sqrt(d_squared))"
            if frd_paper_log
            else "package default: log(d_squared)"
        ),
        "device": actual_device,
        "fid_batch_size": batch_size,
        "fid_loader_workers": loader_workers,
        "frd_workers": frd_workers,
        "elapsed_seconds": elapsed_seconds,
    }
    results, json_path, csv_path, manifest_path = write_outputs(
        output_dir,
        settings,
        selected_originals,
        fid_scores,
        frd_scores,
    )

    print_summary(fid_scores, frd_scores)
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved selection manifest: {manifest_path}")
    print(f"Elapsed time: {elapsed_seconds / 60.0:.1f} minutes")
    return results
