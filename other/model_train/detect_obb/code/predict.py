from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_WEIGHTS = BASE_DIR / "output" / "train" / "weights" / "best.pt"
IMAGE_DIR = BASE_DIR / "dataset" / "all_dataset" / "val" / "images"


model = YOLO(str(MODEL_WEIGHTS))
image_path = next(IMAGE_DIR.glob("*.*"))
results = model(str(image_path), save=True, project="output", name="predict", imgsz=640)

for result in results:
    xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
    xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
    names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
    confs = result.obb.conf  # confidence score of each box
