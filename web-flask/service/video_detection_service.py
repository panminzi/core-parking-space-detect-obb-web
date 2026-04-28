"""
停车位视频检测服务模块
"""
import os
import cv2
import time
import threading
import numpy as np
from copy import deepcopy
from ultralytics import YOLO
from config import MODEL_CONFIGS, CLASS_NAME_MAPPING
from service.detection_optimization import ParkingSpaceMemoryTracker, run_robust_obb_detection


_model_cache = {}
_video_processors = {}
VIDEO_DETECTION_INTERVAL = 3
VIDEO_STATE_CHANGE_VOTES = 3


class ParkingVideoStateStabilizer:
    """Give video detections fixed parking-space ids and debounce state changes."""

    def __init__(self, state_change_votes=VIDEO_STATE_CHANGE_VOTES):
        self.state_change_votes = state_change_votes
        self._states = {}
        self._last_detections = []

    def update(self, detections, frame_number):
        stabilized = []
        seen_ids = set()

        for detection in detections:
            parking_id = detection.get('parking_space_id') or detection.get('detection_id')
            if parking_id is None:
                parking_id = len(self._states) + 1
            seen_ids.add(parking_id)

            raw_state = detection.get('class_name_en', detection.get('class_name'))
            state = self._states.setdefault(parking_id, {
                'parking_space_id': parking_id,
                'stable_state': raw_state,
                'candidate_state': None,
                'candidate_votes': 0,
                'first_frame': frame_number,
                'last_frame': frame_number,
                'frames_seen': 0,
                'occupied_frames': 0,
                'vacant_frames': 0,
                'total_confidence': 0.0,
                'max_confidence': 0.0,
                'transitions': [],
            })

            if raw_state == state['stable_state']:
                state['candidate_state'] = None
                state['candidate_votes'] = 0
            else:
                if state['candidate_state'] == raw_state:
                    state['candidate_votes'] += 1
                else:
                    state['candidate_state'] = raw_state
                    state['candidate_votes'] = 1

                if state['candidate_votes'] >= self.state_change_votes:
                    old_state = state['stable_state']
                    state['stable_state'] = raw_state
                    state['candidate_state'] = None
                    state['candidate_votes'] = 0
                    state['transitions'].append({
                        'frame_number': frame_number,
                        'from': old_state,
                        'to': raw_state,
                    })

            stable_detection = deepcopy(detection)
            stable_state = state['stable_state']
            stable_detection['raw_class_name_en'] = raw_state
            stable_detection['raw_class_name_zh'] = detection.get('class_name_zh')
            stable_detection['class_name'] = stable_state
            stable_detection['class_name_en'] = stable_state
            stable_detection['class_name_zh'] = CLASS_NAME_MAPPING.get(stable_state, stable_state)
            stable_detection['stable_state'] = stable_state
            stable_detection['state_debounced'] = raw_state != stable_state
            stable_detection['state_vote_count'] = state['candidate_votes']
            stable_detection['parking_space_id'] = parking_id
            stable_detection['interpolated'] = False
            stabilized.append(stable_detection)

        self._last_detections = stabilized
        self._record_frame(stabilized, frame_number)
        return stabilized

    def reuse_last(self, frame_number):
        reused = []
        for detection in self._last_detections:
            cloned = deepcopy(detection)
            cloned['interpolated'] = True
            cloned['tracking_status'] = 'interpolated'
            reused.append(cloned)
        self._record_frame(reused, frame_number)
        return reused

    def _record_frame(self, detections, frame_number):
        for detection in detections:
            parking_id = detection.get('parking_space_id')
            if parking_id not in self._states:
                continue

            state = self._states[parking_id]
            stable_state = detection.get('stable_state', detection.get('class_name_en'))
            state['last_frame'] = frame_number
            state['frames_seen'] += 1
            state['total_confidence'] += detection.get('confidence', 0.0)
            state['max_confidence'] = max(state['max_confidence'], detection.get('confidence', 0.0))
            if stable_state == 'occupied':
                state['occupied_frames'] += 1
            elif stable_state == 'vacant':
                state['vacant_frames'] += 1

    def summary(self, fps):
        summaries = []
        for state in self._states.values():
            frames_seen = state['frames_seen']
            occupied_frames = state['occupied_frames']
            vacant_frames = state['vacant_frames']
            summaries.append({
                'parking_space_id': state['parking_space_id'],
                'current_state': state['stable_state'],
                'first_frame': state['first_frame'],
                'last_frame': state['last_frame'],
                'frames_seen': frames_seen,
                'occupied_frames': occupied_frames,
                'vacant_frames': vacant_frames,
                'occupied_seconds': round(occupied_frames / fps, 2) if fps > 0 else 0,
                'vacant_seconds': round(vacant_frames / fps, 2) if fps > 0 else 0,
                'avg_confidence': round(state['total_confidence'] / frames_seen, 3) if frames_seen else 0,
                'max_confidence': round(state['max_confidence'], 3),
                'state_changes': len(state['transitions']),
                'transitions': state['transitions'][:20],
            })

        summaries.sort(key=lambda item: item['parking_space_id'])
        return summaries


def load_model(model_name):
    """加载模型，使用缓存避免重复加载。"""
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
    """在视频帧上绘制 OBB 检测框和标签。"""
    if not detections:
        return frame

    color_mapping = {
        'vacant': (0, 255, 0),
        'occupied': (0, 0, 255),
    }

    for detection in detections:
        polygon = detection['obb']['polygon']
        points = []
        for i in range(0, len(polygon), 2):
            points.append([int(polygon[i]), int(polygon[i + 1])])
        points = np.array(points, np.int32)

        class_name = detection['class_name']
        confidence = detection['confidence']
        color = color_mapping.get(class_name, (128, 128, 128))

        cv2.polylines(frame, [points], True, color, 2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], color)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

        bbox = detection['bbox']
        x1, y1 = int(bbox['x1']), int(bbox['y1'])
        label_suffix = " mem" if detection.get('recovered_from_memory') else ""
        if detection.get('interpolated'):
            label_suffix += " hold"
        if detection.get('state_debounced'):
            label_suffix += " stable"
        parking_id = detection.get('parking_space_id')
        id_prefix = f"#{parking_id} " if parking_id else ""
        label = f"{id_prefix}{class_name}{label_suffix} {confidence:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

    return frame


def detect_objects_in_frame(model, frame, tracker=None):
    """对单帧图像进行停车位检测。"""
    try:
        img_height, img_width = frame.shape[:2]
        optimized_result = run_robust_obb_detection(
            model,
            frame,
            image_size=(img_width, img_height),
            class_name_mapping=CLASS_NAME_MAPPING,
            always_run_rescue=False,
        )
        detections = optimized_result['detections']
        tracking_info = None
        if tracker is not None:
            detections, tracking_info = tracker.update(detections)

        optimization = dict(optimized_result['optimization'])
        if tracking_info:
            optimization['tracking'] = tracking_info
            optimization['postprocess'] = 'multi-pass-merge+memory-tracking'

        return detections, optimization
    except Exception as e:
        print(f"帧检测错误: {e}")
        return [], {
            'primary_count': 0,
            'rescue_count': 0,
            'used_rescue_pass': False,
            'processing_time': 0,
            'postprocess': 'multi-pass-merge',
        }


def process_video(model_name, video_path, progress_callback=None):
    """处理视频，对每一帧进行停车位检测并绘制检测框。"""
    try:
        start_time = time.time()

        if model_name not in MODEL_CONFIGS:
            raise ValueError(f'不支持的模型: {model_name}')

        if progress_callback:
            progress_callback(5, "正在加载模型...")
        model = load_model(model_name)

        if progress_callback:
            progress_callback(10, "正在读取视频文件...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        detection_interval = VIDEO_DETECTION_INTERVAL

        if progress_callback:
            progress_callback(15, f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")

        output_path = video_path.replace('.', '_processed.')
        if progress_callback:
            progress_callback(18, "正在初始化视频编码器...")

        out = None
        for codec in ('avc1', 'XVID', 'mp4v'):
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
            if out.isOpened():
                break
        if out is None or not out.isOpened():
            raise ValueError("无法创建视频输出文件，请检查系统是否支持视频编码")

        total_detections = 0
        frames_with_detections = 0
        total_processing_time = 0
        rescue_frame_count = 0
        recovered_frame_count = 0
        model_inference_frames = 0
        interpolated_frames = 0
        debounced_detection_count = 0
        current_frame = 0
        detection_stats = {}
        frame_detection_details = []
        tracker = ParkingSpaceMemoryTracker(max_missing_frames=8, min_hits=2)
        state_stabilizer = ParkingVideoStateStabilizer()

        if progress_callback:
            progress_callback(20, "开始处理视频帧...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame += 1
            if progress_callback and current_frame % 10 == 0:
                frame_progress = 20 + (current_frame / max(total_frames, 1)) * 70
                progress_callback(frame_progress, f"处理第 {current_frame}/{total_frames} 帧")

            should_detect = current_frame == 1 or (current_frame - 1) % detection_interval == 0
            optimization = {
                'processing_time': 0,
                'used_rescue_pass': False,
                'tracking': {},
                'postprocess': 'interpolated-from-last-detection',
            }
            if should_detect:
                detections, optimization = detect_objects_in_frame(model, frame, tracker)
                detections = state_stabilizer.update(detections, current_frame)
                model_inference_frames += 1
                total_processing_time += optimization.get('processing_time', 0)
                if optimization.get('used_rescue_pass'):
                    rescue_frame_count += 1
                recovered_count = optimization.get('tracking', {}).get('recovered_count', 0)
                if recovered_count > 0:
                    recovered_frame_count += 1
            else:
                detections = state_stabilizer.reuse_last(current_frame)
                interpolated_frames += 1

            frame_detection_info = {
                'frame_number': current_frame,
                'detections_count': len(detections),
                'inference_frame': should_detect,
                'interpolated': not should_detect,
                'detections': [],
            }

            if detections:
                frames_with_detections += 1
                total_detections += len(detections)

                for detection in detections:
                    class_name_en = detection['class_name']
                    class_name_zh = detection['class_name_zh']
                    confidence = detection['confidence']

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
                    if detection.get('state_debounced'):
                        debounced_detection_count += 1

                    frame_detection_info['detections'].append({
                        'class_name_en': class_name_en,
                        'class_name_zh': class_name_zh,
                        'raw_class_name_en': detection.get('raw_class_name_en'),
                        'raw_class_name_zh': detection.get('raw_class_name_zh'),
                        'confidence': confidence,
                        'bbox': detection['bbox'],
                        'obb': detection['obb'],
                        'tracking_status': detection.get('tracking_status', 'live'),
                        'parking_space_id': detection.get('parking_space_id'),
                        'recovered_from_memory': detection.get('recovered_from_memory', False),
                        'interpolated': detection.get('interpolated', False),
                        'stable_state': detection.get('stable_state'),
                        'state_debounced': detection.get('state_debounced', False),
                    })

            frame_detection_details.append(frame_detection_info)
            processed_frame = draw_obb_detection_boxes_on_frame(frame, detections)
            out.write(processed_frame)

        cap.release()
        out.release()

        if progress_callback:
            progress_callback(95, "正在生成结果...")

        processed_detection_stats = []
        for _, stats in detection_stats.items():
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

        processed_detection_stats.sort(key=lambda x: x['count'], reverse=True)
        parking_space_summary = state_stabilizer.summary(fps)

        result = {
            'processed_video_path': output_path,
            'total_frames': total_frames,
            'frames_with_detections': frames_with_detections,
            'total_detections': total_detections,
            'processing_time': round(time.time() - start_time, 2),
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
                'class_statistics': processed_detection_stats,
                'avg_inference_time_per_frame': total_processing_time / total_frames if total_frames > 0 else 0,
                'avg_inference_time_per_model_frame': total_processing_time / model_inference_frames if model_inference_frames > 0 else 0,
                'rescue_frames': rescue_frame_count,
                'recovered_frames': recovered_frame_count,
                'model_inference_frames': model_inference_frames,
                'interpolated_frames': interpolated_frames,
                'detection_interval': detection_interval,
                'fixed_parking_spaces': len(parking_space_summary),
                'debounced_detections': debounced_detection_count,
                'state_change_votes': VIDEO_STATE_CHANGE_VOTES,
            },
            'parking_space_summary': parking_space_summary,
            'frame_details': frame_detection_details[:100]
        }

        if progress_callback:
            progress_callback(100, "处理完成")

        return {'code': 200, 'data': result}

    except FileNotFoundError as e:
        return {'code': 404, 'message': str(e)}
    except ValueError as e:
        return {'code': 400, 'message': str(e)}
    except Exception as e:
        return {'code': 500, 'message': f'视频处理过程中发生错误: {str(e)}'}


class VideoProcessor:
    """视频处理器类，支持进度回调。"""

    def __init__(self):
        self.progress = 0
        self.message = ""
        self.is_processing = False
        self.result = None
        self.error = None

    def update_progress(self, progress, message):
        self.progress = progress
        self.message = message
        print(f"进度: {progress}% - {message}")

    def process_video_async(self, model_name, video_path):
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
        return {
            'is_processing': self.is_processing,
            'progress': self.progress,
            'message': self.message,
            'result': self.result,
            'error': self.error
        }


def start_video_processing(session_id, model_name, video_path):
    """开始视频处理。"""
    processor = VideoProcessor()
    _video_processors[session_id] = processor

    thread = threading.Thread(
        target=processor.process_video_async,
        args=(model_name, video_path)
    )
    thread.daemon = True
    thread.start()

    return processor


def get_processing_status(session_id):
    """获取视频处理状态。"""
    processor = _video_processors.get(session_id)
    if not processor:
        return {'code': 404, 'message': '未找到处理任务'}

    return {'code': 200, 'data': processor.get_status()}


def cleanup_processing_session(session_id):
    """清理处理会话。"""
    if session_id in _video_processors:
        del _video_processors[session_id]
