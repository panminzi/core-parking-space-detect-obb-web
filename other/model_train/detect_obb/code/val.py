from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_WEIGHTS = BASE_DIR / "output" / "train" / "weights" / "best.pt"
DATA_CONFIG = BASE_DIR / "dataset" / "all_dataset" / "data.yaml"


model = YOLO(str(MODEL_WEIGHTS))
metrics = model.val(data=str(DATA_CONFIG), project="output", name="val", split="val")
print(metrics.box.map)  # map50-95
print(metrics.box.map50)  # map50
print(metrics.box.map75)  # map75
print(metrics.box.maps)  # a list contains map50-95 of each category
