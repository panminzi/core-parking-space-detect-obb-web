from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CFG.save_dir = str(BASE_DIR / "output" / "train")

DATA_CONFIG = BASE_DIR / "dataset" / "small_dataset" / "data.yaml"
PRETRAINED_WEIGHTS = BASE_DIR / "weights" / "yolo11n-obb.pt"
MODEL_CONFIG = BASE_DIR / "code" / "models" / "yolo11n-obb-parking-attn.yaml"


def main():
    # Keep the official YOLO11n-OBB pretrained weights, but train a
    # parking-scene variant with one extra C2PSA attention block.
    model = YOLO(str(MODEL_CONFIG), task="obb").load(str(PRETRAINED_WEIGHTS))
    model.train(
        data=str(DATA_CONFIG),
        epochs=50,
        imgsz=640,
    )


if __name__ == "__main__":
    main()
