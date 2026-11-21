"""YOLOとSAM3の検出結果を画像に描画する。"""

from typing import Any

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

_MASK_COLOR = (255, 128, 0)
_YOLO_COLOR = (0, 255, 0)
_SAM3_COLOR = (0, 0, 255)


def _validate_bbox(bbox: ArrayLike) -> tuple[int, int, int, int]:
    """描画用bboxを検証し、整数座標に変換する。"""
    values = np.asarray(bbox)
    if values.shape != (4,) or values.dtype.kind not in "iuf":
        raise ValueError("bboxは有限な数値4個のxyxy形式で指定してください。")
    values = values.astype(float)
    if not np.all(np.isfinite(values)):
        raise ValueError("bboxの座標は有限値で指定してください。")
    return tuple(int(round(value)) for value in values)


def draw_detections(
    image: NDArray[np.uint8],
    yolo_detections: list[dict[str, Any]],
    sam3_detections: list[dict[str, Any]],
    iou_values: list[float],
) -> NDArray[np.uint8]:
    """検出mask・bboxとIoUを描画した画像を返す。

    Args:
        image: 描画元のBGR uint8画像。
        yolo_detections: YOLOのxyxy検出結果。
        sam3_detections: SAM3のxyxyとmaskを持つ検出結果。
        iou_values: 各SAM3検出に対応するIoU値。

    Returns:
        入力画像を変更せず、検出結果を重ねた画像。

    Raises:
        ValueError: 画像、検出結果、mask、IoUの形式が一致しない場合。
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("画像はuint8形式のH×W×3 NumPy配列で指定してください。")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("入力画像が空です。")
    if not isinstance(yolo_detections, list) or not isinstance(sam3_detections, list) or not isinstance(iou_values, list):
        raise ValueError("YOLO、SAM3検出結果とIoU値はリストで指定してください。")
    if len(sam3_detections) != len(iou_values):
        raise ValueError("SAM3検出結果とIoU値の件数を一致させてください。")

    result_image = image.copy()
    for detection in sam3_detections:
        if not isinstance(detection, dict) or "mask" not in detection:
            raise ValueError("SAM3検出結果にはmaskが必要です。")
        mask = np.asarray(detection["mask"])
        if mask.shape != image.shape[:2]:
            raise ValueError("SAM3のmask形状は入力画像と一致させてください。")
        mask = mask.astype(bool, copy=False)
        overlay = result_image.copy()
        overlay[mask] = _MASK_COLOR
        blended = cv2.addWeighted(result_image, 0.6, overlay, 0.4, 0)
        result_image[mask] = blended[mask]

    for index, detection in enumerate(yolo_detections, start=1):
        if not isinstance(detection, dict) or "xyxy" not in detection:
            raise ValueError("YOLO検出結果にはxyxy bboxが必要です。")
        x1, y1, x2, y2 = _validate_bbox(detection["xyxy"])
        cv2.rectangle(result_image, (x1, y1), (x2, y2), _YOLO_COLOR, 2)
        cv2.putText(
            result_image,
            f"YOLO {index}",
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _YOLO_COLOR,
            2,
        )

    for index, (detection, iou_value) in enumerate(zip(sam3_detections, iou_values, strict=True), start=1):
        if "xyxy" not in detection:
            raise ValueError("SAM3検出結果にはxyxy bboxが必要です。")
        if isinstance(iou_value, bool) or not isinstance(iou_value, (int, float, np.number)) or not np.isfinite(iou_value):
            raise ValueError("IoU値は有限な数値で指定してください。")
        x1, y1, x2, y2 = _validate_bbox(detection["xyxy"])
        cv2.rectangle(result_image, (x1, y1), (x2, y2), _SAM3_COLOR, 2)
        cv2.putText(
            result_image,
            f"SAM3 {index} IoU={iou_value:.3f}",
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _SAM3_COLOR,
            2,
        )

    return result_image
