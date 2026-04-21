from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("output/train/weights/best.pt")

# 图片路径
image_path = "dataset/small_dataset/train/images/17_jpg.rf.31c5d0b06c377cacdb514cd655d5fef7.jpg"
# 预测图片
results = model(image_path, save=True, project="output", name="predict", imgsz=640)

# 预测结果
for result in results:
    xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
    xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
    names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
    confs = result.obb.conf  # confidence score of each box