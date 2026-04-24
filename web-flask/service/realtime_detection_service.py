"""
实时摄像头检测服务模块
"""
import os
import base64
import io
import time
import threading
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from config import MODEL_CONFIGS, CLASS_NAME_MAPPING
from service.detection_optimization import ParkingSpaceMemoryTracker, run_robust_obb_detection


_model_cache = {}
_detection_sessions = {}


def load_model(model_name):
    """加载模型，使用缓存避免重复加载。"""
    if model_name not in _model_cache:
        model_config = MODEL_CONFIGS.get(model_name)
        if not model_config:
            raise ValueError(f"不支持的模型: {model_name}")

        model_path = model_config['model_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        print(f"加载实时检测模型: {model_name} from {model_path}")
        _model_cache[model_name] = YOLO(model_path)

    return _model_cache[model_name]


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


def encode_image_to_base64(image):
    """将 PIL 图像编码为 base64 字符串。"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=95, optimize=True)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


def draw_detections_on_image(image, detections):
    """在图像上绘制检测结果。"""
    draw = ImageDraw.Draw(image)
    color_mapping = {
        'vacant': '#00ff00',
        'occupied': '#ff0000',
    }
    font = ImageFont.load_default()

    for detection in detections:
        class_name = detection.get('class_name_en', detection.get('class_name'))
        color = color_mapping.get(class_name, '#808080')
        polygon = detection['obb']['polygon']
        coords = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]

        draw.polygon(coords, outline=color, width=2)

        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        if color.startswith('#'):
            color_rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
            overlay_draw.polygon(coords, fill=color_rgb + (50,))

        image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(image)

        label_suffix = " mem" if detection.get('recovered_from_memory') else ""
        label = f"{class_name}{label_suffix} {(detection['confidence'] * 100):.1f}%"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        label_x, label_y = coords[0]
        draw.rectangle(
            [label_x, label_y - text_height - 5, label_x + text_width + 10, label_y],
            fill=color
        )
        draw.text((label_x + 5, label_y - text_height - 2), label, fill='white', font=font)

    return image


def detect_objects_realtime(model_name, image_data, tracker=None):
    """对单帧图像进行实时停车位检测。"""
    started_at = time.time()
    try:
        if model_name not in MODEL_CONFIGS:
            return {'code': 400, 'message': f'不支持的模型: {model_name}'}

        model = load_model(model_name)
        image = decode_base64_image(image_data)
        img_array = np.array(image)

        optimized_result = run_robust_obb_detection(
            model,
            img_array,
            image_size=image.size,
            class_name_mapping=CLASS_NAME_MAPPING,
            always_run_rescue=False,
        )
        detections = optimized_result['detections']
        raw_detection_count = len(detections)
        tracking_info = None
        if tracker is not None:
            detections, tracking_info = tracker.update(detections)

        result_image = image.copy()
        if detections:
            result_image = draw_detections_on_image(result_image, detections)

        optimization = dict(optimized_result['optimization'])
        if tracking_info:
            optimization['tracking'] = tracking_info
            optimization['postprocess'] = 'multi-pass-merge+memory-tracking'

        return {
            'code': 200,
            'data': {
                'detections': detections,
                'raw_detections': raw_detection_count,
                'parking_spaces': len(detections),
                'processed_image': encode_image_to_base64(result_image),
                'optimization': optimization,
                'processing_time': round(time.time() - started_at, 4),
            }
        }
    except FileNotFoundError as e:
        return {'code': 404, 'message': str(e)}
    except ValueError as e:
        return {'code': 400, 'message': str(e)}
    except Exception as e:
        return {'code': 500, 'message': f'实时检测过程中发生错误: {str(e)}'}


class RealtimeDetectionSession:
    """实时检测会话类。"""

    def __init__(self, session_id, model_name):
        self.session_id = session_id
        self.model_name = model_name
        self.start_time = datetime.now()
        self.frame_count = 0
        self.detection_count = 0
        self.total_processing_time = 0
        self.recent_detections = []
        self.is_active = True
        self.tracker = ParkingSpaceMemoryTracker()

    def process_frame(self, image_data):
        """处理单帧。"""
        if not self.is_active:
            return {'code': 400, 'message': '会话已结束'}

        self.frame_count += 1
        result = detect_objects_realtime(self.model_name, image_data, self.tracker)

        if result['code'] == 200:
            data = result['data']
            self.total_processing_time += data.get('processing_time', 0)

            if data['detections']:
                self.detection_count += len(data['detections'])
                for detection in data['detections']:
                    self.recent_detections.append({
                        'timestamp': datetime.now().isoformat(),
                        'frame_number': self.frame_count,
                        **detection
                    })

                if len(self.recent_detections) > 100:
                    self.recent_detections = self.recent_detections[-100:]

        return result

    def get_statistics(self):
        """获取会话统计信息。"""
        runtime = (datetime.now() - self.start_time).total_seconds()
        return {
            'session_id': self.session_id,
            'model_name': self.model_name,
            'start_time': self.start_time.isoformat(),
            'runtime_seconds': runtime,
            'frame_count': self.frame_count,
            'detection_count': self.detection_count,
            'avg_processing_time': self.total_processing_time / self.frame_count if self.frame_count > 0 else 0,
            'fps': self.frame_count / runtime if runtime > 0 else 0,
            'recent_detections': self.recent_detections[-10:],
            'is_active': self.is_active
        }

    def stop(self):
        """停止会话。"""
        self.is_active = False


def start_detection_session(session_id, model_name):
    """开始检测会话。"""
    try:
        if model_name not in MODEL_CONFIGS:
            return {'code': 400, 'message': f'不支持的模型: {model_name}'}

        load_model(model_name)
        session = RealtimeDetectionSession(session_id, model_name)
        _detection_sessions[session_id] = session

        return {
            'code': 200,
            'message': '检测会话创建成功',
            'data': {
                'session_id': session_id,
                'model_name': model_name,
                'start_time': session.start_time.isoformat()
            }
        }
    except Exception as e:
        return {'code': 500, 'message': f'创建检测会话失败: {str(e)}'}


def process_frame_in_session(session_id, image_data):
    """在会话中处理帧。"""
    session = _detection_sessions.get(session_id)
    if not session:
        return {'code': 404, 'message': '检测会话不存在'}
    return session.process_frame(image_data)


def get_session_statistics(session_id):
    """获取会话统计信息。"""
    session = _detection_sessions.get(session_id)
    if not session:
        return {'code': 404, 'message': '检测会话不存在'}
    return {'code': 200, 'data': session.get_statistics()}


def stop_detection_session(session_id):
    """停止检测会话。"""
    session = _detection_sessions.get(session_id)
    if not session:
        return {'code': 404, 'message': '检测会话不存在'}

    session.stop()
    final_stats = session.get_statistics()
    del _detection_sessions[session_id]

    return {
        'code': 200,
        'message': '检测会话已停止',
        'data': final_stats
    }


def cleanup_inactive_sessions():
    """清理非活跃会话。"""
    current_time = datetime.now()
    inactive_sessions = []

    for session_id, session in _detection_sessions.items():
        if (current_time - session.start_time).total_seconds() > 3600:
            inactive_sessions.append(session_id)

    for session_id in inactive_sessions:
        if session_id in _detection_sessions:
            _detection_sessions[session_id].stop()
            del _detection_sessions[session_id]
            print(f"清理非活跃检测会话: {session_id}")

    return len(inactive_sessions)


def start_cleanup_task():
    """启动定期清理任务。"""
    def cleanup_worker():
        while True:
            try:
                cleanup_inactive_sessions()
                time.sleep(300)
            except Exception as e:
                print(f"清理任务错误: {e}")
                time.sleep(60)

    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    print("实时检测会话清理任务已启动")


start_cleanup_task()
