"""SAM3のインスタンス予測を評価する。"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils


def _as_rle(mask: Any) -> dict[str, Any]:
    if isinstance(mask, dict):
        counts = mask["counts"]
        return {**mask, "counts": counts.encode("ascii") if isinstance(counts, str) else counts}
    array = np.asarray(mask, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError("maskは2次元配列またはCOCO RLE形式で指定してください。")
    return mask_utils.encode(np.asfortranarray(array))


def intersection_over_union(first: dict[str, Any], second: dict[str, Any], metric: str) -> float:
    if metric == "bbox":
        first_box = np.asarray(first["xyxy"], dtype=float)
        second_box = np.asarray(second["xyxy"], dtype=float)
        left = max(first_box[0], second_box[0])
        top = max(first_box[1], second_box[1])
        right = min(first_box[2], second_box[2])
        bottom = min(first_box[3], second_box[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first_box[2] - first_box[0]) * max(0.0, first_box[3] - first_box[1])
        second_area = max(0.0, second_box[2] - second_box[0]) * max(0.0, second_box[3] - second_box[1])
        union = first_area + second_area - intersection
        return float(intersection / union) if union else 0.0
    if metric != "mask":
        raise ValueError("metricは'bbox'または'mask'を指定してください。")
    return float(mask_utils.iou([_as_rle(first["mask"])], [_as_rle(second["mask"])], [0])[0][0])


def decode_mask(mask: Any) -> np.ndarray:
    return mask_utils.decode(_as_rle(mask)).astype(bool, copy=False)


def match_image(
    ground_truth: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    iou_threshold: float,
    metric: str,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    """画像内のカテゴリ一致した予測をconfidence降順で貪欲に対応付ける。"""
    candidates = [
        (index, prediction)
        for index, prediction in enumerate(predictions)
        if confidence_threshold is None or float(prediction["confidence"]) >= confidence_threshold
    ]
    candidates.sort(key=lambda item: float(item[1]["confidence"]), reverse=True)
    unmatched = set(range(len(ground_truth)))
    matches = []
    for prediction_index, prediction in candidates:
        best_index = None
        best_iou = -1.0
        for gt_index in unmatched:
            target = ground_truth[gt_index]
            if int(target["category_id"]) != int(prediction["category_id"]):
                continue
            iou = intersection_over_union(target, prediction, metric)
            if iou > best_iou:
                best_index, best_iou = gt_index, iou
        if best_index is not None and best_iou >= iou_threshold:
            unmatched.remove(best_index)
            matches.append(
                {
                    "prediction_index": prediction_index,
                    "ground_truth_index": best_index,
                    "iou": best_iou,
                    "status": "TP",
                }
            )
        else:
            matches.append(
                {
                    "prediction_index": prediction_index,
                    "ground_truth_index": None,
                    "iou": None if best_index is None else best_iou,
                    "status": "FP",
                }
            )
    return {
        "matches": matches,
        "false_negative_ground_truth_indices": sorted(unmatched),
        "tp": sum(item["status"] == "TP" for item in matches),
        "fp": sum(item["status"] == "FP" for item in matches),
        "fn": len(unmatched),
    }


def calculate_average_precision(
    ground_truth_by_image: dict[int, list[dict[str, Any]]],
    predictions_by_image: dict[int, list[dict[str, Any]]],
    category_id: int,
    iou_threshold: float,
    metric: str,
) -> float | None:
    ground_truth = {
        image_id: [item for item in items if int(item["category_id"]) == category_id]
        for image_id, items in ground_truth_by_image.items()
    }
    total_ground_truth = sum(map(len, ground_truth.values()))
    if total_ground_truth == 0:
        return None

    predictions = [
        (image_id, item)
        for image_id, items in predictions_by_image.items()
        for item in items
        if int(item["category_id"]) == category_id
    ]
    predictions.sort(key=lambda item: float(item[1]["confidence"]), reverse=True)
    matched: dict[int, set[int]] = {image_id: set() for image_id in ground_truth}
    true_positives = []
    false_positives = []
    for image_id, prediction in predictions:
        targets = ground_truth.get(image_id, [])
        best_index = None
        best_iou = -1.0
        for gt_index, target in enumerate(targets):
            if gt_index in matched.get(image_id, set()):
                continue
            iou = intersection_over_union(target, prediction, metric)
            if iou > best_iou:
                best_index, best_iou = gt_index, iou
        is_true_positive = best_index is not None and best_iou >= iou_threshold
        if is_true_positive:
            matched.setdefault(image_id, set()).add(best_index)
        true_positives.append(int(is_true_positive))
        false_positives.append(int(not is_true_positive))

    if not predictions:
        return 0.0
    tp = np.cumsum(true_positives, dtype=float)
    fp = np.cumsum(false_positives, dtype=float)
    recall = tp / total_ground_truth
    precision = tp / np.maximum(tp + fp, np.finfo(float).eps)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall_points = np.linspace(0.0, 1.0, 101)
    interpolated = [
        float(np.max(precision[recall >= point])) if np.any(recall >= point) else 0.0
        for point in recall_points
    ]
    return float(np.mean(interpolated))


def calculate_prf1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def calculate_prf1_by_image(
    ground_truth_by_image: dict[int, list[dict[str, Any]]],
    predictions_by_image: dict[int, list[dict[str, Any]]],
    category_id: int,
    iou_threshold: float,
    metric: str,
    confidence_threshold: float,
) -> dict[str, Any]:
    tp = fp = fn = 0
    image_matches = {}
    for image_id in set(ground_truth_by_image) | set(predictions_by_image):
        targets = [item for item in ground_truth_by_image.get(image_id, []) if int(item["category_id"]) == category_id]
        predictions = [item for item in predictions_by_image.get(image_id, []) if int(item["category_id"]) == category_id]
        result = match_image(targets, predictions, iou_threshold, metric, confidence_threshold)
        image_matches[image_id] = result
        tp += result["tp"]
        fp += result["fp"]
        fn += result["fn"]
    return {"counts": {"tp": tp, "fp": fp, "fn": fn}, **calculate_prf1(tp, fp, fn), "image_matches": image_matches}
