"""SAM3推論の手動確認と結果保存を行う。"""

import json
import logging
import math
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sam3_inference.inference import Sam3Segmenter
from src.sam3_inference.iou import calculate_iou
from src.sam3_inference.visualization import draw_detections

LOGGER = logging.getLogger(__name__)
IMAGE_PATH = Path(__file__).resolve().parent / "6e69b38146aa4975b1feb2e1f4db17d6.jpg"
YOLO_RESULT_PATH = IMAGE_PATH.with_name(f"{IMAGE_PATH.stem}_yolo_result.json")
RESULT_PATH = IMAGE_PATH.with_name(f"{IMAGE_PATH.stem}_sam3_result.json")
RESULT_IMAGE_PATH = IMAGE_PATH.with_name(f"{IMAGE_PATH.stem}_sam3_result.jpg")


def main() -> None:
    """SAM3推論を実行し、検出結果と描画画像を保存する。"""
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"動作確認用の画像が見つかりません: {IMAGE_PATH}")
    if not YOLO_RESULT_PATH.is_file():
        raise FileNotFoundError(f"YOLO推論結果のJSONが見つかりません: {YOLO_RESULT_PATH}")

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise OSError(f"動作確認用の画像を読み込めませんでした: {IMAGE_PATH}")

    try:
        with YOLO_RESULT_PATH.open(encoding="utf-8") as result_file:
            yolo_detections = json.load(result_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"YOLO推論結果のJSONを読み込めないか、形式が正しくありません: {YOLO_RESULT_PATH}") from exc

    if not isinstance(yolo_detections, list) or not yolo_detections:
        raise ValueError(f"YOLO推論結果のJSONが空か、形式が正しくありません: {YOLO_RESULT_PATH}")

    bboxes = []
    for index, detection in enumerate(yolo_detections, start=1):
        if not isinstance(detection, dict):
            raise ValueError(f"YOLO推論結果のJSONに不正な検出結果があります（{index}件目）: {YOLO_RESULT_PATH}")
        bbox = detection.get("xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in bbox):
            raise ValueError(f"YOLO推論結果のJSONに有効なbbox(xyxy)がありません（{index}件目）: {YOLO_RESULT_PATH}")
        bboxes.append(bbox)

    if not bboxes:
        raise ValueError(f"YOLO推論結果のJSONにbboxがありません: {YOLO_RESULT_PATH}")

    segmenter = Sam3Segmenter()
    detections = segmenter.infer(image, bboxes)
    if not detections:
        LOGGER.info("検出結果はありません。")

    result_data = []
    bbox_ious = [calculate_iou(detection["source_xyxy"], detection["xyxy"]) for detection in detections]
    for index, (detection, bbox_iou) in enumerate(zip(detections, bbox_ious, strict=True), start=1):
        mask_filename = f"{IMAGE_PATH.stem}_sam3_mask_{index:03d}.png"
        mask_path = IMAGE_PATH.with_name(mask_filename)
        try:
            image_saved = cv2.imwrite(str(mask_path), detection["mask"].astype("uint8") * 255)
        except cv2.error as exc:
            raise OSError(f"SAM3のマスク画像を保存できませんでした: {mask_path}") from exc
        if not image_saved:
            raise OSError(f"SAM3のマスク画像を保存できませんでした（cv2.imwriteがFalseを返しました）: {mask_path}")

        LOGGER.info(
            "検出%d: 信頼度=%.3f、SAM bbox=%s、YOLO bbox=%s、矩形IoU=%.3f、mask=%s",
            index,
            detection["confidence"],
            detection["xyxy"],
            detection["source_xyxy"],
            bbox_iou,
            mask_filename,
        )
        result_data.append(
            {
                "confidence": detection["confidence"],
                "xyxy": detection["xyxy"],
                "source_xyxy": detection["source_xyxy"],
                "bbox_iou": bbox_iou,
                "mask_filename": mask_filename,
            }
        )

    result_image = draw_detections(image, yolo_detections, detections, bbox_ious)

    try:
        image_saved = cv2.imwrite(str(RESULT_IMAGE_PATH), result_image)
    except cv2.error as exc:
        raise OSError(f"SAM3推論結果の画像を保存できませんでした: {RESULT_IMAGE_PATH}") from exc
    if not image_saved:
        raise OSError(f"SAM3推論結果の画像を保存できませんでした（cv2.imwriteがFalseを返しました）: {RESULT_IMAGE_PATH}")
    LOGGER.info("推論結果の画像を保存しました: %s", RESULT_IMAGE_PATH)

    try:
        with RESULT_PATH.open("w", encoding="utf-8") as result_file:
            json.dump(result_data, result_file, ensure_ascii=False, indent=2)
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise OSError(f"SAM3推論結果のJSONを保存できませんでした: {RESULT_PATH}") from exc
    LOGGER.info("推論結果のJSONを保存しました: %s", RESULT_PATH)


if __name__ == "__main__":
    main()
