# MD5: 4ee205ed1c0ce729b7180d30a09644c5
"""
版权所有 © 2025 羊羊小栈 (GJQ)

作者：羊羊小栈
时间：2025-07-31 16:25:08

本系统为原创作品，禁止二次销售！禁止将系统对应的视频、文档进行二次发布！
违者需立即停止侵权行为，并按照【羊羊小栈系统版权声明及保护条款】中规定数额进行赔偿，并承担相应法律责任。
"""

"""
实时摄像头检测服务模块
"""
import os
import base64
import io
import time
import threading
from datetime import datetime
# import cv2  # 当前未使用，保留备用
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from config import MODEL_CONFIGS, CLASS_NAME_MAPPING


# 全局模型缓存
_model_cache = {}

# 实时检测会话缓存
_detection_sessions = {}


def load_model(model_name):
    """加载模型，使用缓存避免重复加载"""
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
    """解码base64图像数据"""
    try:
        # 移除data:image/xxx;base64,前缀
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # 解码base64
        image_bytes = base64.b64decode(image_data)
        
        # 转换为PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # 转换为RGB格式（如果需要）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        raise ValueError(f"图像解码失败: {str(e)}")


def encode_image_to_base64(image):
    """将PIL图像编码为base64字符串"""
    buffer = io.BytesIO()
    # 使用更高质量的JPEG编码
    image.save(buffer, format='JPEG', quality=95, optimize=True)
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


def draw_detections_on_image(image, detections):
    """在图像上绘制检测结果"""
    # 创建高质量绘图对象
    draw = ImageDraw.Draw(image)
    
    # 定义颜色映射
    color_mapping = {
        'vacant': '#00ff00',      # 绿色 - 空闲车位
        'occupied': '#ff0000',    # 红色 - 占用车位
    }
    
    # 使用默认字体
    font = ImageFont.load_default()
    
    for detection in detections:
        class_name = detection['class_name_en']
        color = color_mapping.get(class_name, '#808080')  # 默认灰色
        polygon = detection['obb']['polygon']
        
        # 将一维数组转换为坐标对
        coords = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
        
        # 绘制多边形边框 - 使用更平滑的绘制方式
        # 先绘制填充，再绘制边框以获得更好的效果
        draw.polygon(coords, outline=color, width=2)
        
        # 绘制半透明填充
        # PIL不直接支持alpha混合，所以我们创建一个临时图像
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # 将颜色转换为RGB值并添加透明度
        if color.startswith('#'):
            color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            fill_color = color_rgb + (50,)  # 添加alpha通道
            overlay_draw.polygon(coords, fill=fill_color)
        
        # 合并overlay到原图
        image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(image)  # 重新创建draw对象
        
        # 绘制标签
        label = f"{detection['class_name_en']} {(detection['confidence'] * 100):.1f}%"
        
        # 获取文本边界框
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 使用多边形第一个顶点作为标签位置
        label_x, label_y = coords[0]
        
        # 绘制标签背景
        label_bg = [label_x, label_y - text_height - 5, 
                   label_x + text_width + 10, label_y]
        draw.rectangle(label_bg, fill=color)
        
        # 绘制标签文本
        draw.text((label_x + 5, label_y - text_height - 2), label, fill='white', font=font)
    
    return image


def detect_objects_realtime(model_name, image_data):
    """
    对单帧图像进行实时停车位检测
    
    Args:
        model_name: 模型名称
        image_data: base64编码的图像数据
        
    Returns:
        dict: 检测结果
    """
    try:
        # 检查模型是否支持
        if model_name not in MODEL_CONFIGS:
            return {'code': 400, 'message': f'不支持的模型: {model_name}'}
        
        # 加载模型
        model = load_model(model_name)
        
        # 解码图像
        image = decode_base64_image(image_data)
        
        # 转换为numpy数组以便YOLO处理
        img_array = np.array(image)
        
        # 进行停车位检测预测
        results = model(img_array, verbose=False, imgsz=640)
        
        detections = []
        
        # 解析检测结果
        if results and len(results) > 0:
            result = results[0]
            
            # 获取OBB检测结果
            if (hasattr(result, 'obb') and 
                result.obb is not None and 
                hasattr(result.obb, '__len__') and 
                len(result.obb) > 0 and
                hasattr(result.obb, 'xyxyxyxy') and
                len(result.obb.xyxyxyxy) > 0):
                
                obb = result.obb
                
                # 遍历每个检测到的停车位
                num_detections = len(obb.xyxyxyxy) if hasattr(obb, 'xyxyxyxy') else 0
                for i in range(num_detections):
                    # 获取OBB坐标（xyxyxyxy格式 - 4个顶点）
                    xyxyxyxy = obb.xyxyxyxy[i].cpu().numpy()
                    
                    # 获取中心点坐标和旋转角度（xywhr格式）
                    xywhr = obb.xywhr[i].cpu().numpy()
                    center_x, center_y, width, height, rotation = xywhr
                    
                    # 获取置信度
                    confidence = float(obb.conf[i].cpu().numpy())
                    
                    # 获取类别
                    class_id = int(obb.cls[i].cpu().numpy())
                    
                    # 确保所有值都是标量而不是数组
                    if hasattr(center_x, '__len__'):
                        center_x = float(center_x[0]) if len(center_x) > 0 else 0.0
                    if hasattr(center_y, '__len__'):
                        center_y = float(center_y[0]) if len(center_y) > 0 else 0.0
                    if hasattr(width, '__len__'):
                        width = float(width[0]) if len(width) > 0 else 0.0
                    if hasattr(height, '__len__'):
                        height = float(height[0]) if len(height) > 0 else 0.0
                    if hasattr(rotation, '__len__'):
                        rotation = float(rotation[0]) if len(rotation) > 0 else 0.0
                    
                    class_name_en = result.names[class_id]
                    class_name_zh = CLASS_NAME_MAPPING.get(class_name_en, class_name_en)
                    
                    # 转换多边形坐标为一维数组
                    xyxyxyxy_flat = xyxyxyxy.flatten()
                    
                    detection_info = {
                        'class_name_en': class_name_en,
                        'class_name_zh': class_name_zh, 
                        'class_id': class_id,
                        'confidence': confidence,
                        'obb': {
                            'center_x': float(center_x),
                            'center_y': float(center_y),
                            'width': float(width),
                            'height': float(height),
                            'rotation': float(rotation),
                            'polygon': xyxyxyxy_flat.tolist()
                        }
                    }
                    detections.append(detection_info)
                
                # 按置信度排序
                detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 在图像上绘制检测结果
        result_image = image.copy()
        if detections:
            result_image = draw_detections_on_image(result_image, detections)
        
        # 将处理后的图像编码为base64
        result_image_base64 = encode_image_to_base64(result_image)
        
        return {
            'code': 200,
            'data': {
                'detections': detections,
                'processed_image': result_image_base64
            }
        }
        
    except FileNotFoundError as e:
        return {'code': 404, 'message': str(e)}
    except ValueError as e:
        return {'code': 400, 'message': str(e)}
    except Exception as e:
        return {'code': 500, 'message': f'实时检测过程中发生错误: {str(e)}'}


class RealtimeDetectionSession:
    """实时检测会话类"""
    
    def __init__(self, session_id, model_name):
        self.session_id = session_id
        self.model_name = model_name
        self.start_time = datetime.now()
        self.frame_count = 0
        self.detection_count = 0
        self.total_processing_time = 0
        self.recent_detections = []
        self.is_active = True
        
    def process_frame(self, image_data):
        """处理单帧"""
        if not self.is_active:
            return {'code': 400, 'message': '会话已结束'}
        
        self.frame_count += 1
        
        # 进行检测
        result = detect_objects_realtime(self.model_name, image_data)
        
        if result['code'] == 200:
            data = result['data']
            self.total_processing_time += data['processing_time']
            
            if data['detections']:
                self.detection_count += len(data['detections'])
                
                # 保存最近检测结果
                for detection in data['detections']:
                    self.recent_detections.append({
                        'timestamp': datetime.now().isoformat(),
                        'frame_number': self.frame_count,
                        **detection
                    })
                
                # 限制最近检测数量
                if len(self.recent_detections) > 100:
                    self.recent_detections = self.recent_detections[-100:]
        
        return result
    
    def get_statistics(self):
        """获取会话统计信息"""
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
            'recent_detections': self.recent_detections[-10:],  # 最近10个检测
            'is_active': self.is_active
        }
    
    def stop(self):
        """停止会话"""
        self.is_active = False


def start_detection_session(session_id, model_name):
    """开始检测会话"""
    try:
        # 检查模型是否存在
        if model_name not in MODEL_CONFIGS:
            return {'code': 400, 'message': f'不支持的模型: {model_name}'}
        
        # 预加载模型
        load_model(model_name)
        
        # 创建会话
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
    """在会话中处理帧"""
    session = _detection_sessions.get(session_id)
    if not session:
        return {'code': 404, 'message': '检测会话不存在'}
    
    return session.process_frame(image_data)


def get_session_statistics(session_id):
    """获取会话统计信息"""
    session = _detection_sessions.get(session_id)
    if not session:
        return {'code': 404, 'message': '检测会话不存在'}
    
    return {
        'code': 200,
        'data': session.get_statistics()
    }


def stop_detection_session(session_id):
    """停止检测会话"""
    session = _detection_sessions.get(session_id)
    if not session:
        return {'code': 404, 'message': '检测会话不存在'}
    
    session.stop()
    
    # 获取最终统计信息
    final_stats = session.get_statistics()
    
    # 清理会话
    del _detection_sessions[session_id]
    
    return {
        'code': 200,
        'message': '检测会话已停止',
        'data': final_stats
    }


def cleanup_inactive_sessions():
    """清理非活跃会话（定期调用）"""
    current_time = datetime.now()
    inactive_sessions = []
    
    for session_id, session in _detection_sessions.items():
        # 如果会话超过1小时未活动，标记为清理
        if (current_time - session.start_time).total_seconds() > 3600:
            inactive_sessions.append(session_id)
    
    for session_id in inactive_sessions:
        if session_id in _detection_sessions:
            _detection_sessions[session_id].stop()
            del _detection_sessions[session_id]
            print(f"清理非活跃检测会话: {session_id}")
    
    return len(inactive_sessions)


# 启动定期清理任务
def start_cleanup_task():
    """启动定期清理任务"""
    def cleanup_worker():
        while True:
            try:
                cleanup_inactive_sessions()
                time.sleep(300)  # 每5分钟清理一次
            except Exception as e:
                print(f"清理任务错误: {e}")
                time.sleep(60)  # 出错后等待1分钟再试
    
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    print("实时检测会话清理任务已启动")


# 模块初始化时启动清理任务
start_cleanup_task()