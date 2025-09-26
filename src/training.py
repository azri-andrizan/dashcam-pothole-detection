# src/training.py
"""
YOLOv8 Training Pipeline with MLflow Integration
Stage: 02_training
"""

import argparse
import logging
import csv
from pathlib import Path
import yaml
import mlflow
from ultralytics import YOLO
from typing import Dict, Any
import subprocess
import torch

torch.cuda.empty_cache()


# ----------------------------- Logging Setup -----------------------------
def setup_logging(log_dir: Path = Path("logs")) -> logging.Logger:
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ----------------------------- Helpers -----------------------------
def load_yaml(path: str) -> dict:
    """Load YAML file safely with error handling."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML file not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML: {path}\n{e}")


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def to_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in {"1", "true", "yes", "y", "on"}
    return default


def get_git_commit_hash() -> str:
    """Get current git commit hash if available."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def map_scheduler_flag(scheduler: str | None) -> Dict[str, Any]:
    """Map scheduler name to Ultralytics args."""
    sch = (scheduler or "").strip().lower()
    return {"cos_lr": sch == "cosine"}


# ----------------------------- Core Training -----------------------------
def train_from_config(exp_cfg: dict, logger: logging.Logger):
    logger.info("Starting YOLOv11 training...")

    # -------- Parse config --------
    exp_meta = exp_cfg.get("experiment", {})
    dataset = exp_cfg.get("dataset", {})
    training = exp_cfg.get("training", {})
    augmentation = training.get("augmentation", {})
    loss_cfg = training.get("loss", {})
    mlflow_cfg = exp_cfg.get("mlflow", {})
    output_cfg = exp_cfg.get("output", {})

    exp_name = exp_meta.get("name", "pothole_experiment")
    data_yaml = dataset.get("config", "configs/dataset_v2.yaml") # Rename according to the dataset version used
    dataset_version = dataset.get("version", "v2.0")

    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    # ---- Training params
    model_name = training.get("model", {}).get("name", "yolo11s.pt")
    epochs = int(training.get("epochs", 30))
    batch_size = int(training.get("batch_size", 8))
    img_size = int(training.get("img_size", 640))
    optimizer = training.get("optimizer", "AdamW")
    lr0 = float(training.get("lr0", 0.001))
    lrf = float(training.get("lrf", 0.01))
    weight_decay = float(training.get("weight_decay", 0.0005))
    scheduler = training.get("scheduler", "cosine")
    early_stopping = to_bool(training.get("early_stopping", True))
    patience = int(training.get("patience", 5))
    device = str(training.get("device", 0))
    workers = int(training.get("workers", 2))
    resume = to_bool(training.get("resume", False))

    # ---- Output dirs
    project_dir = Path(output_cfg.get("dir", "models"))
    ensure_dir(project_dir)

    # ---- MLflow config
    tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
    mlflow_experiment_name = mlflow_cfg.get("experiment_name", exp_name)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    run_name = "02_trainingv2_ft3" # Rename according to the stage being run

    # ----------------- MLflow Run -----------------
    with mlflow.start_run(run_name=run_name):
        # ---- Tags
        mlflow.set_tags({
            "stage": "training",
            "dataset_version": dataset_version,
            "pipeline": exp_name,
            "git_commit": get_git_commit_hash(),
        })

        # ---- Log Params
        params = {
            "exp_name": exp_name,
            "data_yaml": data_yaml,
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "optimizer": optimizer,
            "lr0": lr0,
            "lrf": lrf,
            "weight_decay": weight_decay,
            "scheduler": scheduler,
            "early_stopping": early_stopping,
            "patience": patience,
            "device": device,
            "workers": workers,
            "resume": resume,
            "augmentation": augmentation,
            "loss": loss_cfg,
            "project_dir": str(project_dir),
        }
        mlflow.log_params(params)

        # ---- Load Model
        logger.info(f"Loading model: {model_name}")
        model = YOLO(model_name)

        # ---- Compose Ultralytics train() kwargs
        train_kwargs = {
            "data": data_yaml,
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": img_size,
            "optimizer": optimizer,
            "lr0": lr0,
            "weight_decay": weight_decay,
            "device": device,
            "workers": workers,
            "resume": resume,
            "project": str(project_dir),
            "name": exp_name,
            "patience": patience if early_stopping else 0,
        }
        train_kwargs.update(map_scheduler_flag(scheduler))

        train_kwargs.update(augmentation)

        # ---- Start training
        logger.info(f"Training started for {epochs} epochs...")
        results = model.train(**train_kwargs)

        save_dir = Path(getattr(results, "save_dir", project_dir / exp_name))
        weights_dir = save_dir / "weights"
        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"
        results_csv = save_dir / "results.csv"

        logger.info(f"Outputs saved to: {save_dir}")

        # ---- Log metrics
        if results_csv.exists():
            with results_csv.open("r") as f:
                reader = csv.DictReader(f)
                final = list(reader)[-1]
                metrics_to_log = {
                    f"train.{k}": float(v) for k, v in final.items() if v.replace('.', '', 1).isdigit()
                }
                mlflow.log_metrics(metrics_to_log)
                logger.info(f"Logged {len(metrics_to_log)} metrics")

        # ---- Log artifacts
        if best_pt.exists():
            mlflow.log_artifact(str(best_pt), artifact_path="weights")
        if last_pt.exists():
            mlflow.log_artifact(str(last_pt), artifact_path="weights")

        for fname in ["results.csv", "hyp.yaml", "args.yaml", "train_batch0.jpg", "labels_correlogram.jpg"]:
            fpath = save_dir / fname
            if fpath.exists():
                mlflow.log_artifact(str(fpath), artifact_path="training_artifacts")

        logger.info("Training completed and logged to MLflow")


# ----------------------------- CLI -----------------------------
def main():
    parser = argparse.ArgumentParser(description="YOLO11 Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/experiment_stage1/experiment.yaml", help="Path to experiment YAML")
    args = parser.parse_args()

    logger = setup_logging()

    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    exp_cfg = load_yaml(args.config)
    train_from_config(exp_cfg, logger)


if __name__ == "__main__":
    main()