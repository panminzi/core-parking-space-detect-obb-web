"""
停车位检测服务模块
"""
import os
import base64
import io
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from config import MODEL_CONFIGS, AVAILABLE_MODELS, CLASS_NAME_MAPPING
from service.detection_optimization import run_robust_obb_detection


_model_cache = {}


def load_model(model_name):
    """加载模型，使用缓存避免重复加载。"""
    if model_name not in _model_cache:
        model_config = MODEL_CONFIGS.get(model_name)
        if not model_config:
            raise ValueError(f"不支持的模型: {model_name}")

        model_path = model_config['model_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"模型文件不存在: {model_path}，请先执行训练脚本 code/train.py 生成权重文件"
            )

        print(f"加载模型: {model_name} from {model_path}")
        _model_cache[model_name] = YOLO(model_path)

    return _model_cache[model_name]


def get_models():
    """获取可用模型列表。"""
    return {'code': 200, 'data': AVAILABLE_MODELS}


def decode_base64_image(image_data):
    """解码 base64 图像数据。"""
    try:
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"图像解码失败: {str(e)}")


def draw_obb_detection_boxes(image, detections):
    """在图像上绘制 OBB 检测框和标签。"""
    try:
        img_array = np.array(image)
        img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        color_mapping = {
            'vacant': (0, 255, 0),
            'occupied': (0, 0, 255),
        }

        for detection in detections:
            polygon = detection['obb']['polygon']
            class_name = detection['class_name']
            confidence = detection['confidence']
            color = color_mapping.get(class_name, (128, 128, 128))

            points = []
            for i in range(0, len(polygon), 2):
                points.append([int(polygon[i]), int(polygon[i + 1])])
            points = np.array(points, np.int32)

            cv2.polylines(img_cv2, [points], True, color, 2)

            overlay = img_cv2.copy()
            cv2.fillPoly(overlay, [points], color)
            img_cv2 = cv2.addWeighted(img_cv2, 0.8, overlay, 0.2, 0)

            bbox = detection['bbox']
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            label = f"{class_name} {confidence:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

            cv2.rectangle(
                img_cv2,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            cv2.putText(
                img_cv2,
                label,
                (x1, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )

        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(img_rgb)

        buffer = io.BytesIO()
        result_image.save(buffer, format='JPEG', quality=90)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        raise ValueError(f"绘制 OBB 检测框失败: {str(e)}")


def detect_objects(model_name, image_data):
    """对图像进行停车位检测。"""
    try:
        if model_name not in MODEL_CONFIGS:
            return {'code': 400, 'message': f'不支持的模型: {model_name}'}

        model = load_model(model_name)
        image = decode_base64_image(image_data)

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            image.save(temp_file.name, 'JPEG')
            temp_path = temp_file.name

        try:
            img_width, img_height = image.size
            optimized_result = run_robust_obb_detection(
                model,
                temp_path,
                image_size=(img_width, img_height),
                class_name_mapping=CLASS_NAME_MAPPING,
                always_run_rescue=False,
                strict=True,
            )
            detections = optimized_result['detections']

            if len(detections) <= 1:
                relaxed_result = run_robust_obb_detection(
                    model,
                    temp_path,
                    image_size=(img_width, img_height),
                    class_name_mapping=CLASS_NAME_MAPPING,
                    always_run_rescue=True,
                    strict=False,
                )
                relaxed_detections = relaxed_result['detections']
                if len(relaxed_detections) > len(detections):
                    optimized_result = relaxed_result
                    optimized_result['optimization']['fallback_mode'] = 'relaxed-recall-recheck'
                    detections = relaxed_detections

            detection_image_base64 = None
            if detections:
                try:
                    detection_image_base64 = draw_obb_detection_boxes(image, detections)
                except Exception as e:
                    print(f"Warning: Failed to draw detection boxes: {e}")

            return {
                'code': 200,
                'data': {
                    'model': model_name,
                    'image_size': {
                        'width': img_width,
                        'height': img_height
                    },
                    'detections': detections,
                    'total_detections': len(detections),
                    'highest_confidence': detections[0] if detections else None,
                    'detection_image': detection_image_base64,
                    'optimization': optimized_result['optimization'],
                    'message': None if detections else '未检测到任何停车位'
                }
            }
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except FileNotFoundError as e:
        return {'code': 404, 'message': str(e)}
    except ValueError as e:
        return {'code': 400, 'message': str(e)}
    except Exception as e:
        return {'code': 500, 'message': f'停车位检测过程中发生错误: {str(e)}'}
