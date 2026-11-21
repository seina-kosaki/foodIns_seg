"""xyxy形式bbox間のIntersection over Unionを計算する。"""

import numpy as np
from numpy.typing import ArrayLike


def calculate_iou(first_bbox: ArrayLike, second_bbox: ArrayLike) -> float:
    """2つのxyxy bboxのIoUを返す。

    Args:
        first_bbox: 1つ目のxyxy座標。
        second_bbox: 2つ目のxyxy座標。

    Returns:
        交差領域と和領域の面積比。和領域が空の場合は0.0。

    Raises:
        ValueError: 座標が有限な数値4個の正方向xyxy形式でない場合。
    """
    first = np.asarray(first_bbox)
    second = np.asarray(second_bbox)
    if first.shape != (4,) or first.dtype.kind not in "iuf":
        raise ValueError("bboxは有限な数値4個のxyxy形式で指定してください。")
    if second.shape != (4,) or second.dtype.kind not in "iuf":
        raise ValueError("bboxは有限な数値4個のxyxy形式で指定してください。")

    first = first.astype(float)
    second = second.astype(float)
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("bboxの座標は有限値で指定してください。")

    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second
    if first_x2 < first_x1 or first_y2 < first_y1 or second_x2 < second_x1 or second_y2 < second_y1:
        raise ValueError("bboxは正方向のxyxy座標で指定してください。")

    intersection_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection_area = intersection_width * intersection_height
    first_area = (first_x2 - first_x1) * (first_y2 - first_y1)
    second_area = (second_x2 - second_x1) * (second_y2 - second_y1)
    union_area = first_area + second_area - intersection_area
    return float(intersection_area / union_area) if union_area > 0 else 0.0
