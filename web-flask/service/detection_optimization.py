"""
Detection optimization helpers for occlusion-heavy parking-space scenes.
"""
from copy import deepcopy
import time


PRIMARY_INFERENCE_OPTIONS = {
    'conf': 0.20,
    'iou': 0.45,
    'imgsz': 768,
    'augment': False,
}

RESCUE_INFERENCE_OPTIONS = {
    'conf': 0.10,
    'iou': 0.55,
    'imgsz': 960,
    'augment': True,
}


def _safe_scalar(value):
    """Convert tensor/numpy-like scalars to plain float."""
    if hasattr(value, '__len__'):
        return float(value[0]) if len(value) > 0 else 0.0
    return float(value)


def _bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a['x1'], box_a['y1'], box_a['x2'], box_a['y2']
    bx1, by1, bx2, by2 = box_b['x1'], box_b['y1'], box_b['x2'], box_b['y2']

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _center_distance_ratio(det_a, det_b):
    dx = det_a['obb']['center_x'] - det_b['obb']['center_x']
    dy = det_a['obb']['center_y'] - det_b['obb']['center_y']
    distance = (dx * dx + dy * dy) ** 0.5
    scale = max(
        det_a['obb']['width'],
        det_a['obb']['height'],
        det_b['obb']['width'],
        det_b['obb']['height'],
        1.0,
    )
    return distance / scale


def _is_same_detection(det_a, det_b):
    if det_a['class_id'] != det_b['class_id']:
        return False
    if _bbox_iou(det_a['bbox'], det_b['bbox']) >= 0.45:
        return True
    return _center_distance_ratio(det_a, det_b) <= 0.35


def parse_obb_detections(result, image_size, class_name_mapping):
    """Normalize YOLO OBB output into the API detection structure."""
    if (
        not hasattr(result, 'obb')
        or result.obb is None
        or not hasattr(result.obb, '__len__')
        or len(result.obb) == 0
        or not hasattr(result.obb, 'xyxyxyxy')
        or len(result.obb.xyxyxyxy) == 0
    ):
        return []

    img_width, img_height = image_size
    obb = result.obb
    detections = []
    num_detections = len(obb.xyxyxyxy) if hasattr(obb, 'xyxyxyxy') else 0

    for i in range(num_detections):
        xyxyxyxy = obb.xyxyxyxy[i].cpu().numpy()
        xywhr = obb.xywhr[i].cpu().numpy()
        center_x, center_y, width, height, rotation = xywhr
        confidence = float(obb.conf[i].cpu().numpy())
        class_id = int(obb.cls[i].cpu().numpy())

        center_x = _safe_scalar(center_x)
        center_y = _safe_scalar(center_y)
        width = _safe_scalar(width)
        height = _safe_scalar(height)
        rotation = _safe_scalar(rotation)

        class_name_en = result.names[class_id]
        class_name_zh = class_name_mapping.get(class_name_en, class_name_en)

        xyxyxyxy_flat = xyxyxyxy.flatten()
        x_coords = xyxyxyxy_flat[::2]
        y_coords = xyxyxyxy_flat[1::2]
        x1, y1 = float(min(x_coords)), float(min(y_coords))
        x2, y2 = float(max(x_coords)), float(max(y_coords))

        detections.append({
            'detection_id': i + 1,
            'class_name': class_name_en,
            'class_name_en': class_name_en,
            'class_name_zh': class_name_zh,
            'class_id': class_id,
            'confidence': confidence,
            'percentage': f"{confidence * 100:.2f}%",
            'obb': {
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height,
                'rotation': rotation,
                'polygon': xyxyxyxy_flat.tolist()
            },
            'bbox': {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height
            },
            'bbox_normalized': {
                'center_x': center_x / img_width if img_width else 0.0,
                'center_y': center_y / img_height if img_height else 0.0,
                'width': width / img_width if img_width else 0.0,
                'height': height / img_height if img_height else 0.0
            }
        })

    detections.sort(key=lambda item: item['confidence'], reverse=True)
    for index, detection in enumerate(detections, start=1):
        detection['detection_id'] = index
    return detections


def merge_detections(*detection_groups):
    """Merge multi-pass detections while keeping rescued weak targets."""
    merged = []
    for group_index, group in enumerate(detection_groups):
        source_pass = 'primary' if group_index == 0 else 'rescue'
        for detection in group:
            candidate = detection.copy()
            candidate['source_passes'] = [source_pass]
            duplicate_index = next(
                (index for index, existing in enumerate(merged) if _is_same_detection(existing, candidate)),
                None
            )
            if duplicate_index is None:
                merged.append(candidate)
                continue

            existing = merged[duplicate_index]
            existing_passes = existing.setdefault('source_passes', [])
            if source_pass not in existing_passes:
                existing_passes.append(source_pass)

            if candidate['confidence'] > existing['confidence']:
                candidate['source_passes'] = existing_passes
                merged[duplicate_index] = candidate
            else:
                existing.setdefault('rescued_candidates', 0)
                existing['rescued_candidates'] += 1

    merged.sort(key=lambda item: item['confidence'], reverse=True)
    for index, detection in enumerate(merged, start=1):
        detection['detection_id'] = index
    return merged


def _touches_image_border(detection, margin=0.02):
    bbox = detection.get('bbox_normalized', {})
    half_width = bbox.get('width', 0.0) / 2.0
    half_height = bbox.get('height', 0.0) / 2.0
    center_x = bbox.get('center_x', 0.0)
    center_y = bbox.get('center_y', 0.0)
    x1 = center_x - half_width
    x2 = center_x + half_width
    y1 = center_y - half_height
    y2 = center_y + half_height
    return x1 <= margin or y1 <= margin or x2 >= (1.0 - margin) or y2 >= (1.0 - margin)


def _should_drop_strict_weak_detection(detection):
    source_passes = detection.get('source_passes', [])
    is_rescue_only = 'rescue' in source_passes and 'primary' not in source_passes
    if not is_rescue_only:
        return False

    class_name = detection.get('class_name_en', detection.get('class_name'))
    confidence = detection['confidence']
    bbox = detection.get('bbox_normalized', {})
    width = bbox.get('width', 0.0)
    height = bbox.get('height', 0.0)
    narrow_shape = min(width, height) < 0.09

    if confidence < 0.30:
        return True
    if class_name == 'vacant' and confidence < 0.60 and narrow_shape and _touches_image_border(detection):
        return True
    return False


def _resolve_state_conflicts(detections, strict=False):
    ranked = []
    discarded_indexes = set()

    for detection in detections:
        class_name = detection.get('class_name_en', detection.get('class_name'))
        occupied_bonus = 0.08 if class_name == 'occupied' else 0.0
        strict_penalty = 0.05 if strict and _should_drop_strict_weak_detection(detection) else 0.0
        ranked.append((detection['confidence'] + occupied_bonus - strict_penalty, detection))

    ranked.sort(key=lambda item: item[0], reverse=True)
    ordered = [item[1] for item in ranked]

    for index, current in enumerate(ordered):
        if index in discarded_indexes:
            continue

        current_class = current.get('class_name_en', current.get('class_name'))
        for other_index in range(index + 1, len(ordered)):
            if other_index in discarded_indexes:
                continue

            other = ordered[other_index]
            other_class = other.get('class_name_en', other.get('class_name'))
            if current_class == other_class:
                continue

            overlap = _bbox_iou(current['bbox'], other['bbox'])
            center_ratio = _center_distance_ratio(current, other)
            if overlap < 0.10 and center_ratio > 0.45:
                continue

            if current_class == 'occupied' and current['confidence'] + 0.08 >= other['confidence']:
                discarded_indexes.add(other_index)
                continue

            if other_class == 'occupied' and other['confidence'] + 0.08 >= current['confidence']:
                discarded_indexes.add(index)
                break

            discarded_indexes.add(other_index)

    refined = []
    for index, detection in enumerate(ordered):
        if index in discarded_indexes:
            continue
        if strict and _should_drop_strict_weak_detection(detection):
            continue
        refined.append(detection)

    refined.sort(key=lambda item: item['confidence'], reverse=True)
    for index, detection in enumerate(refined, start=1):
        detection['detection_id'] = index
    return refined


def should_run_rescue_pass(detections):
    """Run a second pass when results look weak or sparse."""
    if not detections:
        return True

    top_confidence = detections[0]['confidence']
    weak_count = sum(1 for item in detections if item['confidence'] < 0.35)
    return top_confidence < 0.55 or weak_count >= max(1, len(detections) // 2)


def _tracking_match_score(track_detection, detection):
    """Score how likely two detections are to be the same parking space."""
    iou = _bbox_iou(track_detection['bbox'], detection['bbox'])
    center_ratio = _center_distance_ratio(track_detection, detection)

    if iou < 0.15 and center_ratio > 0.65:
        return None

    return iou * 2.0 - center_ratio


def _format_percentage(confidence):
    return f"{confidence * 100:.2f}%"


class ParkingSpaceMemoryTracker:
    """
    Keep stable parking spaces alive for a few frames when they are briefly occluded.

    This does not create brand-new parking spaces from nothing. Instead, it remembers
    recently stable detections and surfaces them again for a short time if the next
    few frames fail to detect them.
    """

    def __init__(
        self,
        max_missing_frames=6,
        min_hits=2,
        recovery_confidence=0.24,
    ):
        self.max_missing_frames = max_missing_frames
        self.min_hits = min_hits
        self.recovery_confidence = recovery_confidence
        self._next_track_id = 1
        self._tracks = []

    def _new_track(self, detection):
        track = {
            'id': self._next_track_id,
            'last_detection': deepcopy(detection),
            'hits': 1,
            'misses': 0,
        }
        self._next_track_id += 1
        self._tracks.append(track)
        return track

    def _match_tracks(self, detections):
        candidates = []
        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                score = _tracking_match_score(track['last_detection'], detection)
                if score is not None:
                    candidates.append((score, track_index, detection_index))

        candidates.sort(reverse=True, key=lambda item: item[0])

        assigned_tracks = set()
        assigned_detections = set()
        matches = []

        for _, track_index, detection_index in candidates:
            if track_index in assigned_tracks or detection_index in assigned_detections:
                continue
            assigned_tracks.add(track_index)
            assigned_detections.add(detection_index)
            matches.append((track_index, detection_index))

        return matches, assigned_tracks, assigned_detections

    def update(self, detections):
        """
        Attach stable parking-space ids and recover recently occluded spaces.
        """
        live_detections = []
        recovered_detections = []
        fresh_track_ids = set()
        matches, matched_track_indexes, matched_detection_indexes = self._match_tracks(detections)

        for track_index, detection_index in matches:
            track = self._tracks[track_index]
            detection = deepcopy(detections[detection_index])

            track['last_detection'] = deepcopy(detection)
            track['hits'] += 1
            track['misses'] = 0

            detection['parking_space_id'] = track['id']
            detection['tracking_status'] = 'live'
            detection['recovered_from_memory'] = False
            detection['track_hits'] = track['hits']
            live_detections.append(detection)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indexes:
                continue

            track = self._new_track(detection)
            fresh_track_ids.add(track['id'])
            tracked_detection = deepcopy(detection)
            tracked_detection['parking_space_id'] = track['id']
            tracked_detection['tracking_status'] = 'live'
            tracked_detection['recovered_from_memory'] = False
            tracked_detection['track_hits'] = track['hits']
            live_detections.append(tracked_detection)

        active_tracks = []
        for track_index, track in enumerate(self._tracks):
            if track_index in matched_track_indexes:
                active_tracks.append(track)
                continue
            if track['id'] in fresh_track_ids:
                active_tracks.append(track)
                continue

            track['misses'] += 1
            if track['misses'] > self.max_missing_frames:
                continue

            active_tracks.append(track)
            if track['hits'] < self.min_hits:
                continue

            recovered_detection = deepcopy(track['last_detection'])
            recovered_detection['parking_space_id'] = track['id']
            recovered_detection['tracking_status'] = 'recovered'
            recovered_detection['recovered_from_memory'] = True
            recovered_detection['occluded_frames'] = track['misses']
            recovered_detection['track_hits'] = track['hits']
            recovered_confidence = max(
                min(recovered_detection['confidence'] * 0.8, self.recovery_confidence),
                0.12,
            )
            recovered_detection['confidence'] = recovered_confidence
            recovered_detection['percentage'] = _format_percentage(recovered_confidence)
            recovered_detections.append(recovered_detection)

        self._tracks = active_tracks

        combined = live_detections + recovered_detections
        combined.sort(key=lambda item: (item.get('recovered_from_memory', False), -item['confidence']))
        for index, detection in enumerate(combined, start=1):
            detection['detection_id'] = index

        return combined, {
            'live_count': len(live_detections),
            'recovered_count': len(recovered_detections),
            'active_tracks': len(self._tracks),
            'max_missing_frames': self.max_missing_frames,
            'min_hits': self.min_hits,
        }


def run_robust_obb_detection(
    model,
    source,
    image_size,
    class_name_mapping,
    always_run_rescue=False,
    strict=False,
):
    """
    Run primary inference and an optional rescue pass tuned for occlusion.
    """
    started_at = time.time()

    primary_results = model(source, verbose=False, **PRIMARY_INFERENCE_OPTIONS)
    primary_detections = []
    if primary_results and len(primary_results) > 0:
        primary_detections = parse_obb_detections(
            primary_results[0],
            image_size=image_size,
            class_name_mapping=class_name_mapping,
        )

    run_rescue = always_run_rescue or should_run_rescue_pass(primary_detections)
    rescue_detections = []
    if run_rescue:
        rescue_results = model(source, verbose=False, **RESCUE_INFERENCE_OPTIONS)
        if rescue_results and len(rescue_results) > 0:
            rescue_detections = parse_obb_detections(
                rescue_results[0],
                image_size=image_size,
                class_name_mapping=class_name_mapping,
            )

    detections = merge_detections(primary_detections, rescue_detections)
    detections = _resolve_state_conflicts(detections, strict=strict)
    return {
        'detections': detections,
        'optimization': {
            'primary_options': PRIMARY_INFERENCE_OPTIONS,
            'rescue_options': RESCUE_INFERENCE_OPTIONS if run_rescue else None,
            'primary_count': len(primary_detections),
            'rescue_count': len(rescue_detections),
            'used_rescue_pass': run_rescue,
            'postprocess': 'multi-pass-merge+state-conflict-filter' if strict else 'multi-pass-merge+state-conflict-resolve',
            'processing_time': round(time.time() - started_at, 4),
        }
    }
