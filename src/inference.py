import cv2
import torch
import yaml
import argparse
import json
import mlflow
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from ultralytics import YOLO

# =========================================
# Utility Functions
# =========================================
def load_config(config_path, preset_name=None):
    """Load YAML config and merge with preset if provided."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if preset_name:
        if 'presets' not in config:
            raise ValueError("No presets defined in config file.")
        if preset_name not in config['presets']:
            raise ValueError(f"Preset '{preset_name}' not found in config file.")

        print(f"[INFO] Applying preset: {preset_name}")
        preset_values = config['presets'][preset_name]
        merged_config = deepcopy(config)
        for key, value in preset_values.items():
            merged_config[key] = value
        return merged_config

    return config


def get_video_writer(output_path, fps, frame_size):
    """Initialize a video writer object with avc1 codec."""
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def save_detections_json(detections, output_path):
    """Save detection results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(detections, f, indent=4)
    print(f"[INFO] Detection results saved to {output_path}")


# =========================================
# Main Inference Pipeline
# =========================================
def run_inference(config_path, video_input=None, video_output=None, preset=None):
    cfg = load_config(config_path, preset)
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------
    # Video Input
    # ------------------------
    if video_input is None:
        video_input = cfg.get("video_input", None)
    if video_input is None or not Path(video_input).exists():
        raise FileNotFoundError(f"Video file not found: {video_input}")

    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_input}")

    # Ambil informasi dasar video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        print("[WARNING] Invalid FPS, Using default 30 FPS")
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video loaded: {width}x{height} @ {fps:.2f} FPS, Total frames: {total_frames}")

    if cfg['skip_frames'] > 0:
        original_fps = fps
        fps = fps / (cfg['skip_frames'] + 1)
        print(f"[INFO] Adjusted FPS due to skip_frames: {original_fps:.2f} -> {fps:.2f}")

    # ------------------------
    # Output video writer
    # ------------------------
    if video_output is None:
        video_output = output_dir / f"inference_{preset or 'default'}.mp4" # Video output name can be customized
    writer = get_video_writer(str(video_output), fps, (width, height))

    # ------------------------
    # Load YOLO Model
    # ------------------------
    model_path = cfg['model']['path']
    print(f"[INFO] Loading YOLO model from: {model_path}")
    model = YOLO(model_path)

    device = "cuda" if (cfg['device'] == 'auto' and torch.cuda.is_available()) else cfg['device']
    model.to(device)

    # ------------------------
    # MLflow Setup
    # ------------------------
    if cfg['mlflow']['enabled']:
        mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
        mlflow.set_experiment(cfg['mlflow']['experiment_name'])

        run_name = f"inference_{preset or 'default'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        mlflow.log_param("preset", preset or "default")
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("confidence_threshold", cfg['confidence_threshold'])
        mlflow.log_param("iou_threshold", cfg['iou_threshold'])

    # ------------------------
    # Inference Loop
    # ------------------------
    detections_log = []
    frame_idx = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames if configured
        if cfg['skip_frames'] > 0 and frame_idx % (cfg['skip_frames'] + 1) != 0:
            frame_idx += 1
            continue

        # Run YOLO inference
        results = model.predict(
            source=frame,
            conf=cfg['confidence_threshold'],
            iou=cfg['iou_threshold'],
            device=device,
            half=cfg['half_precision'],
            verbose=cfg['verbose']
        )

        annotated_frame = frame.copy()
        frame_detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                              color=tuple(cfg['box_color']),
                              thickness=cfg['box_thickness'])

                label = f"{model.names[cls_id]} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                            getattr(cv2, cfg['font']),
                            cfg['text_scale'],
                            tuple(cfg['text_color']),
                            thickness=1)

                frame_detections.append({
                    "frame": frame_idx,
                    "class": model.names[cls_id],
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })

        # Log detections for this frame
        detections_log.extend(frame_detections)

        # Write annotated frame
        writer.write(annotated_frame)

        processed_frames += 1
        frame_idx += 1

    cap.release()
    writer.release()

    # ------------------------
    # Debugging and validation
    # ------------------------
    print(f"[DEBUG] Total frames input processed: {frame_idx}")
    print(f"[DEBUG] Total frames written to output: {processed_frames}")
    durasi_input = total_frames / (fps * (cfg['skip_frames'] + 1)) if total_frames > 0 else 0
    durasi_output = processed_frames / fps if fps > 0 else 0
    print(f"[DEBUG] Estimate input duration: {durasi_input:.2f} seconds")
    print(f"[DEBUG] Estimate output duration: {durasi_output:.2f} seconds")

    # ------------------------
    # Save JSON detection results
    # ------------------------
    if cfg['save_detections']:
        json_path = output_dir / f"inference_{preset or 'default'}.json"
        save_detections_json(detections_log, json_path)

    # ------------------------
    # Log ke MLflow
    # ------------------------
    if cfg['mlflow']['enabled']:
        if cfg['mlflow']['log_artifacts']:
            mlflow.log_artifact(str(video_output))
            if cfg['save_detections']:
                mlflow.log_artifact(str(json_path))

        if cfg['mlflow']['log_metrics']:
            mlflow.log_metrics({
                "total_frames_input": total_frames,
                "total_frames_output": processed_frames,
                "success_rate": processed_frames / total_frames if total_frames > 0 else 0
            })

        mlflow.end_run()

    print(f"[INFO] Inference complete. Output saved to {video_output}")


# =========================================
# Entry Point
# =========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pothole Detection Inference")
    parser.add_argument("--config", type=str, default="configs/inference.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--preset", type=str, choices=["fast", "balanced", "accurate"],
                        help="Preset mode for inference")
    parser.add_argument("--input", type=str,
                        help="Path to input video file (overrides YAML config)")
    parser.add_argument("--output", type=str,
                        help="Path to save annotated output video")

    args = parser.parse_args()
    run_inference(args.config, args.input, args.output, args.preset)