"""
src/preprocessing.py (refactored)

Preprocessing script for Pothole Detection (YOLO format).
- Reads preprocessing config (default: configs/preprocessing.yaml)
- Reads experiment config (default: configs/experiment.yaml) to configure MLflow
- Performs letterbox resize (preserve aspect ratio) to target size and padding color
- Updates YOLO-format labels (normalized x_center y_center width height)
- Saves processed images/labels to output_dir preserving train/val[/test] structure
- Logs params, metrics, tags, and lightweight artifacts to MLflow (metadata + samples only)

Usage:
    python src/preprocessing.py \
        --config configs/preprocessing.yaml \
        --experiment configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Tuple
import shutil

import cv2
import mlflow
import yaml

# --------------------------------------- Setup Logging ---------------------------------------

def setup_logging(log_dir: Path = Path("logs")) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(exist_ok=True, parents=True)
    logger = logging.getLogger("preprocessing")
    logger.setLevel(logging.INFO)

    # Clear previous handlers
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_dir / "preprocessing.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

# --------------------------------------- Helpers ---------------------------------------

def load_config(path: str | Path) -> Dict:
    """Load YAML configuration file."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {path}: {e}")


def try_git_commit(project_root: Path) -> str | None:
    """Return current git commit hash if available, else None."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(project_root))
        return commit.decode("utf-8").strip()
    except Exception:
        return None

# ----------------------------- Letterbox & label transform -----------------------------

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)) -> Tuple["cv2.Mat", float, Tuple[float, float]]:
    """Resize image with aspect ratio preservation and padding.

    Returns: processed_img, r, (dw, dh)
    """
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw *= 0.5
    dh *= 0.5

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

# ----------------------------- Core: dataset processing -----------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_images(folder: Path) -> Iterable[Path]:
    for ext in IMG_EXTS:
        yield from folder.glob(f"*{ext}")


def log_dataset_metadata(output_dir: Path, sample_limit: int = 5) -> None:
    """Create a small metadata file and log a few sample images to MLflow."""
    imgs = list(output_dir.rglob("images/*"))
    lbls = list(output_dir.rglob("labels/*.txt"))

    metadata = {
        "processed_root": str(output_dir.as_posix()),
        "images_count": len(imgs),
        "labels_count": len(lbls),
        "splits": sorted({p.parts[-3] for p in imgs if len(p.parts) >= 3}),
        "samples_logged": min(sample_limit, len(imgs)),
    }

    meta_path = output_dir / "dataset_info.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    mlflow.log_artifact(str(meta_path), artifact_path="dataset_info")

    # Log a few sample images only 
    for p in imgs[:sample_limit]:
        mlflow.log_artifact(str(p), artifact_path=f"samples/{p.parent.parent.name}")


def process_dataset(cfg: Dict, exp_cfg: Dict, logger: logging.Logger):
    """Main preprocessing function with error handling and metrics tracking."""
    input_dir = Path(cfg["input_dir"]).resolve()
    output_dir = Path(cfg["output_dir"]).resolve()
    overwrite = bool(cfg.get("overwrite", True))
    target_size = tuple(cfg.get("target_size", [640, 640]))
    padding_color = tuple(cfg.get("padding_color", [114, 114, 114]))

    # Validate input directory
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Setup MLflow 
    tracking_uri = exp_cfg.get("mlflow", {}).get("tracking_uri", "mlruns")
    experiment_name = exp_cfg.get("mlflow", {}).get("experiment_name", "pothole_detection_v2")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_name = "01_preprocessingv2_ft2" # Rename according to the stage being run
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"Starting preprocessing pipeline (run_id={run_id})…")

        # MLflow tags for discoverability
        exp_meta = exp_cfg.get("experiment", {})
        mlflow.set_tags({
            "stage": "preprocessing",
            "pipeline": "pothole-detection_v2",
            "dataset_version": str(exp_cfg.get("dataset", {}).get("version", cfg.get("version", "unknown"))),
            "experiment_name": exp_meta.get("name", "unknown"),
        })
        # git commit
        commit = try_git_commit(project_root=Path.cwd())
        if commit:
            mlflow.set_tag("git_commit", commit)

        # Log parameters (flat keys only)
        mlflow.log_params({
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "target_h": int(target_size[0]),
            "target_w": int(target_size[1]),
            "padding_color_r": int(padding_color[0]),
            "padding_color_g": int(padding_color[1]),
            "padding_color_b": int(padding_color[2]),
            "overwrite": overwrite,
        })

        # Prepare output root (clean only if overwrite)
        if output_dir.exists() and overwrite:
            for child in output_dir.iterdir():
                if child.is_file():
                    child.unlink(missing_ok=True)
                else:
                    shutil.rmtree(child)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Metrics
        total_images = 0
        processed_images = 0
        failed_images = 0
        total_labels = 0
        processed_labels = 0

        splits = [s for s in ("train", "val", "test") if (input_dir / s).exists()]
        if not splits:
            splits = [""]  # flat structure (no train/val folders)

        for split in splits:
            split_prefix = f"{split}/" if split else ""
            img_dir = input_dir / split / "images" if split else input_dir / "images"
            lbl_dir = input_dir / split / "labels" if split else input_dir / "labels"
            out_img_dir = output_dir / split / "images" if split else output_dir / "images"
            out_lbl_dir = output_dir / split / "labels" if split else output_dir / "labels"

            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)

            if not img_dir.exists():
                logger.warning(f"Image directory not found: {img_dir}")
                continue

            split_processed = 0
            split_failed = 0

            for img_file in iter_images(img_dir):
                total_images += 1
                try:
                    label_file = lbl_dir / f"{img_file.stem}.txt"
                    output_img_file = out_img_dir / img_file.name

                    # Skip if exists and not overwriting
                    if output_img_file.exists() and not overwrite:
                        processed_images += 1  # treat as OK in idempotent runs
                        split_processed += 1
                        continue

                    # Load and process image
                    img = cv2.imread(str(img_file))
                    if img is None:
                        logger.error(f"Failed to load image: {img_file}")
                        failed_images += 1
                        split_failed += 1
                        continue

                    processed_img, r, (dw, dh) = letterbox(img, new_shape=target_size, color=padding_color)
                    h, w = img.shape[:2]
                    new_w, new_h = processed_img.shape[1], processed_img.shape[0]

                    # Process labels if they exist
                    if label_file.exists():
                        total_labels += 1
                        try:
                            with label_file.open("r", encoding="utf-8") as f:
                                labels = f.readlines()

                            new_labels = []
                            for line in labels:
                                parts = line.strip().split()
                                if len(parts) != 5:
                                    logger.warning(f"Invalid label format in {label_file}: {line.strip()}")
                                    continue
                                cls, x, y, bw, bh = map(float, parts)
                                # YOLO xywh (normalized) -> pixels
                                x, y, bw, bh = x * w, y * h, bw * w, bh * h
                                # transform
                                x = (x * r + dw) / new_w
                                y = (y * r + dh) / new_h
                                bw = (bw * r) / new_w
                                bh = (bh * r) / new_h
                                new_labels.append(f"{int(cls)} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

                            with (out_lbl_dir / label_file.name).open("w", encoding="utf-8") as f:
                                f.writelines(new_labels)
                            processed_labels += 1
                        except Exception as e:
                            logger.error(f"Failed to process label {label_file}: {e}")

                    # Save processed image
                    success = cv2.imwrite(str(output_img_file), processed_img)
                    if not success:
                        logger.error(f"Failed to save image: {output_img_file}")
                        failed_images += 1
                        split_failed += 1
                    else:
                        processed_images += 1
                        split_processed += 1

                except Exception as e:
                    logger.error(f"Error processing {img_file}: {e}")
                    failed_images += 1
                    split_failed += 1

            logger.info(f"{split_prefix or '(root)'} split completed: {split_processed} processed, {split_failed} failed")

        # Log metrics to MLflow (namespaced keys)
        metrics = {
            "images.total": total_images,
            "images.processed": processed_images,
            "images.failed": failed_images,
            "images.success_rate": (processed_images / total_images) if total_images > 0 else 0.0,
            "labels.total": total_labels,
            "labels.processed": processed_labels,
        }
        mlflow.log_metrics(metrics)

        # Log lightweight artifacts only (metadata + a few samples)
        log_dataset_metadata(output_dir)

        logger.info(f"Preprocessing completed. Metrics: {metrics}")
        logger.info(f"MLflow run completed. Tracking URI: {tracking_uri} | Experiment: {experiment_name} | run_id={run_id}")

# ----------------------------- CLI -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess pothole detection dataset (MLflow-integrated)")
    parser.add_argument("--config", type=str, default="configs/experiment_stage1/preprocessing.yaml", help="Path to preprocessing config file")
    parser.add_argument("--experiment", type=str, default="configs/experiment_stage1/experiment.yaml", help="Path to experiment config file")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    logger.setLevel(getattr(logging, args.log_level))

    try:
        # Load configurations
        logger.info(f"Loading preprocessing config: {args.config}")
        preproc_cfg = load_config(args.config)

        logger.info(f"Loading experiment config: {args.experiment}")
        exp_cfg = load_config(args.experiment)

        # Run preprocessing
        process_dataset(preproc_cfg, exp_cfg, logger)
        logger.info("Preprocessing pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()