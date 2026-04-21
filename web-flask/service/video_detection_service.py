# MD5: 4ee205ed1c0ce729b7180d30a09644c5
"""
版权所有 © 2025 羊羊小栈 (GJQ)

作者：羊羊小栈
时间：2025-07-31 16:25:08

本系统为原创作品，禁止二次销售！禁止将系统对应的视频、文档进行二次发布！
违者需立即停止侵权行为，并按照【羊羊小栈系统版权声明及保护条款】中规定数额进行赔偿，并承担相应法律责任。
"""

"""
停车位视频检测服务模块
"""
import os
import cv2
import time
import threading
from ultralytics import YOLO
from config import MODEL_CONFIGS, CLASS_NAME_MAPPING


# 全局模型缓存
_model_cache = {}


def load_model(model_name):
    """加载模型，使用缓存避免重复加载"""
    if model_name not in _model_cache:
        model_config = MODEL_CONFIGS.get(model_name)
        if not model_config:
            raise ValueError(f"不支持的模型: {model_name}")
        
        model_path = model_config['model_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print(f"加载模型: {model_name} from {model_path}")
        _model_cache[model_name] = YOLO(model_path)
    
    return _model_cache[model_name]


def draw_obb_detection_boxes_on_frame(frame, detections):
    """
    在视频帧上绘制OBB检测框和标签（用于旋转边界框）
    
    Args:
        frame: OpenCV图像帧
        detections: OBB检测结果列表
        
    Returns:
        frame: 带OBB检测框的图像帧
    """
    if not detections:
        return frame
    
    import numpy as np
    
    # 定义颜色映射（BGR格式）
    color_mapping = {
        'vacant': (0, 255, 0),      # 绿色 - 空闲车位
        'occupied': (0, 0, 255),    # 红色 - 占用车位
    }
    
    for detection in detections:
        # 获取OBB信息
        if 'obb' in detection:
            obb = detection['obb']
            polygon = obb['polygon']
            
            # 将polygon坐标转换为OpenCV格式
            points = []
            for i in range(0, len(polygon), 2):
                points.append([int(polygon[i]), int(polygon[i+1])])
            points = np.array(points, np.int32)
            
            # 获取类别信息
            class_name = detection['class_name']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            # 选择颜色
            color = color_mapping.get(class_name, (128, 128, 128))  # 默认灰色
            
            # 绘制旋转的检测框（多边形）
            cv2.polylines(frame, [points], True, color, 2)
            
            # 可选：填充半透明颜色
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], color)
            frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
            
            # 计算标签位置（使用bbox的左上角）
            bbox = detection['bbox']
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
        else:
            # 兼容普通边界框格式（后备方案）
            bbox = detection['bbox']
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            
            class_name = detection['class_name']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            color = color_mapping.get(class_name, (128, 128, 128))  # 默认灰色
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签文本
        label = f"{class_name} {confidence:.2f}"
        
        # 计算文本尺寸
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # 绘制标签背景
        cv2.rectangle(frame, 
                     (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), 
                     color, -1)
        
        # 绘制标签文本
        cv2.putText(frame, label, 
                   (x1, y1 - 5), 
                   font, font_scale, 
                   (255, 255, 255), thickness)
    
    return frame


def detect_objects_in_frame(model, frame):
    """
    对单帧图像进行停车位检测
    
    Args:
        model: YOLO模型
        frame: OpenCV图像帧
        
    Returns:
        list: 检测结果列表
    """
    try:
        # 进行预测
        results = model(frame, verbose=False, imgsz=640)
        
        if not results or len(results) == 0:
            return []
        
        result = results[0]
        
        if (not hasattr(result, 'obb') or 
            result.obb is None or 
            not hasattr(result.obb, '__len__') or 
            len(result.obb) == 0 or
            not hasattr(result.obb, 'xyxyxyxy') or
            len(result.obb.xyxyxyxy) == 0):
            return []
        
        # 获取图像尺寸
        img_height, img_width = frame.shape[:2]
        
        detections = []
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
            
            # 计算边界框(用于显示)
            # 确保xyxyxyxy是一维数组
            xyxyxyxy_flat = xyxyxyxy.flatten()
            x_coords = xyxyxyxy_flat[::2]  # x坐标
            y_coords = xyxyxyxy_flat[1::2]  # y坐标
            x1, y1 = float(min(x_coords)), float(min(y_coords))
            x2, y2 = float(max(x_coords)), float(max(y_coords))
            
            # 归一化坐标
            center_x_norm = center_x / img_width
            center_y_norm = center_y / img_height
            width_norm = width / img_width
            height_norm = height / img_height
            
            detection_info = {
                'detection_id': i + 1,
                'class_name': class_name_en,
                'class_name_zh': class_name_zh,
                'class_id': class_id,
                'confidence': confidence,
                'percentage': f"{confidence * 100:.2f}%",
                'obb': {
                    'center_x': float(center_x),
                    'center_y': float(center_y),
                    'width': float(width),
                    'height': float(height),
                    'rotation': float(rotation),
                    'polygon': xyxyxyxy_flat.tolist()  # 4个顶点坐标
                },
                'bbox': {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'center_x': float(center_x),
                    'center_y': float(center_y),
                    'width': float(width),
                    'height': float(height)
                },
                'bbox_normalized': {
                    'center_x': float(center_x_norm),
                    'center_y': float(center_y_norm),
                    'width': float(width_norm),
                    'height': float(height_norm)
                }
            }
            detections.append(detection_info)
        
        # 按置信度排序
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections
        
    except Exception as e:
        print(f"帧检测错误: {e}")
        return []


def process_video(model_name, video_path, progress_callback=None):
    """
    处理视频，对每一帧进行停车位检测并绘制检测框
    
    Args:
        model_name: 模型名称
        video_path: 输入视频路径
        progress_callback: 进度回调函数
        
    Returns:
        dict: 处理结果
    """
    try:
        start_time = time.time()
        
        # 检查模型是否支持
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f'不支持的模型: {model_name}')
        
        # 加载模型
        if progress_callback:
            progress_callback(5, "正在加载模型...")
        model = load_model(model_name)
        
        # 打开视频文件
        if progress_callback:
            progress_callback(10, "正在读取视频文件...")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if progress_callback:
            progress_callback(15, f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
        
        # 创建输出视频文件
        output_path = video_path.replace('.', '_processed.')
        if progress_callback:
            progress_callback(12, "正在初始化视频编码器...")
        
        # 尝试使用H.264编码以确保浏览器兼容性
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264编码
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 如果H.264不可用，回退到XVID编码
        if not out.isOpened():
            print("avc1编码不可用，尝试XVID编码")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        # 最后回退选项：mp4v编码
        if not out.isOpened():
            print("XVID编码不可用，尝试mp4v编码")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        # 如果所有编码都失败，抛出错误
        if not out.isOpened():
            raise ValueError("无法创建视频输出文件，请检查系统是否支持视频编码")
        
        print(f"成功初始化视频编码器，输出路径: {output_path}")
        
        # 统计信息
        total_detections = 0
        frames_with_detections = 0
        current_frame = 0
        detection_stats = {}  # 各类别检测统计
        frame_detection_details = []  # 每帧检测详情
        
        if progress_callback:
            progress_callback(20, "开始处理视频帧...")
        
        # 逐帧处理
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame += 1
            
            # 更新进度
            frame_progress = 20 + (current_frame / total_frames) * 70  # 20-90%
            if progress_callback and current_frame % 10 == 0:  # 每10帧更新一次进度
                progress_callback(
                    frame_progress, 
                    f"处理第 {current_frame}/{total_frames} 帧"
                )
            
            # 对当前帧进行检测
            detections = detect_objects_in_frame(model, frame)
            
            # 统计检测结果
            frame_detection_info = {
                'frame_number': current_frame,
                'detections_count': len(detections),
                'detections': []
            }
            
            if detections:
                frames_with_detections += 1
                total_detections += len(detections)
                
                # 统计各类别检测数量
                for detection in detections:
                    class_name_en = detection['class_name']
                    class_name_zh = detection['class_name_zh']
                    confidence = detection['confidence']
                    
                    # 更新类别统计
                    if class_name_en not in detection_stats:
                        detection_stats[class_name_en] = {
                            'class_name_en': class_name_en,
                            'class_name_zh': class_name_zh,
                            'count': 0,
                            'total_confidence': 0,
                            'max_confidence': 0,
                            'min_confidence': 1.0,
                            'frames_appeared': set()
                        }
                    
                    stats = detection_stats[class_name_en]
                    stats['count'] += 1
                    stats['total_confidence'] += confidence
                    stats['max_confidence'] = max(stats['max_confidence'], confidence)
                    stats['min_confidence'] = min(stats['min_confidence'], confidence)
                    stats['frames_appeared'].add(current_frame)
                    
                    # 记录当前帧检测详情
                    frame_detection_info['detections'].append({
                        'class_name_en': class_name_en,
                        'class_name_zh': class_name_zh,
                        'confidence': confidence,
                        'bbox': detection['bbox']
                    })
            
            # 记录帧检测详情
            frame_detection_details.append(frame_detection_info)
            
            # 在帧上绘制检测框
            processed_frame = draw_obb_detection_boxes_on_frame(frame, detections)
            
            # 写入输出视频
            out.write(processed_frame)
        
        # 释放资源
        cap.release()
        out.release()
        
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        
        if progress_callback:
            progress_callback(95, "正在生成结果...")
        
        # 处理类别统计数据（转换set为list以便JSON序列化）
        processed_detection_stats = []
        for class_name_en, stats in detection_stats.items():
            avg_confidence = stats['total_confidence'] / stats['count'] if stats['count'] > 0 else 0
            frames_appeared = len(stats['frames_appeared'])
            
            processed_detection_stats.append({
                'class_name_en': stats['class_name_en'],
                'class_name_zh': stats['class_name_zh'],
                'count': stats['count'],
                'frames_appeared': frames_appeared,
                'frame_appearance_rate': (frames_appeared / total_frames) * 100 if total_frames > 0 else 0,
                'avg_confidence': round(avg_confidence, 3),
                'max_confidence': round(stats['max_confidence'], 3),
                'min_confidence': round(stats['min_confidence'], 3),
                'detection_rate': (stats['count'] / total_detections) * 100 if total_detections > 0 else 0
            })
        
        # 按检测数量排序
        processed_detection_stats.sort(key=lambda x: x['count'], reverse=True)
        
        # 生成处理结果
        result = {
            'processed_video_path': output_path,
            'total_frames': total_frames,
            'frames_with_detections': frames_with_detections,
            'total_detections': total_detections,
            'processing_time': processing_time,
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'duration': total_frames / fps if fps > 0 else 0
            },
            'detection_statistics': {
                'total_classes_detected': len(detection_stats),
                'detection_rate': (frames_with_detections / total_frames) * 100 if total_frames > 0 else 0,
                'avg_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0,
                'class_statistics': processed_detection_stats
            },
            # 可选：包含详细的帧检测信息（如果需要的话）
            'frame_details': frame_detection_details[:100]  # 限制数量以避免响应过大
        }
        
        if progress_callback:
            progress_callback(100, "处理完成！")
        
        return {'code': 200, 'data': result}
        
    except FileNotFoundError as e:
        return {'code': 404, 'message': str(e)}
    except ValueError as e:
        return {'code': 400, 'message': str(e)}
    except Exception as e:
        return {'code': 500, 'message': f'视频处理过程中发生错误: {str(e)}'}


class VideoProcessor:
    """视频处理器类，支持进度回调"""
    
    def __init__(self):
        self.progress = 0
        self.message = ""
        self.is_processing = False
        self.result = None
        self.error = None
    
    def update_progress(self, progress, message):
        """更新处理进度"""
        self.progress = progress
        self.message = message
        print(f"进度: {progress}% - {message}")
    
    def process_video_async(self, model_name, video_path):
        """异步处理视频"""
        self.is_processing = True
        self.progress = 0
        self.message = "开始处理..."
        
        try:
            result = process_video(model_name, video_path, self.update_progress)
            if result['code'] == 200:
                self.result = result['data']
            else:
                self.error = result['message']
        except Exception as e:
            self.error = str(e)
        finally:
            self.is_processing = False
    
    def get_status(self):
        """获取当前处理状态"""
        return {
            'is_processing': self.is_processing,
            'progress': self.progress,
            'message': self.message,
            'result': self.result,
            'error': self.error
        }


# 全局视频处理器实例
_video_processors = {}


def start_video_processing(session_id, model_name, video_path):
    """开始视频处理"""
    processor = VideoProcessor()
    _video_processors[session_id] = processor
    
    # 在后台线程中处理视频
    thread = threading.Thread(
        target=processor.process_video_async,
        args=(model_name, video_path)
    )
    thread.daemon = True
    thread.start()
    
    return processor


def get_processing_status(session_id):
    """获取视频处理状态"""
    processor = _video_processors.get(session_id)
    if not processor:
        return {'code': 404, 'message': '未找到处理任务'}
    
    status = processor.get_status()
    return {'code': 200, 'data': status}


def cleanup_processing_session(session_id):
    """清理处理会话"""
    if session_id in _video_processors:
        del _video_processors[session_id]