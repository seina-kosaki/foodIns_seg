"""YAML設定に従ってSAM3推論結果を画像単位でSQLiteへ保存する。"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from pycocotools import mask as mask_utils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sam3_inference.inference_categories import Sam3Segmenter
from src.sam3_inference.storage import create_database, increment_skipped, save_image_transaction

SETTINGS_PATH = PROJECT_ROOT / "configs" / "settings_sam3.yaml"
LOGGER = logging.getLogger(__name__)


def load_settings(settings_path: Path = SETTINGS_PATH) -> dict[str, Any]:
    try:
        with settings_path.open(encoding="utf-8") as file:
            settings = yaml.safe_load(file)
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"設定ファイルを読み込めません: {settings_path}") from exc
    if not isinstance(settings, dict):
        raise ValueError("設定ファイルのトップレベルは辞書形式にしてください。")
    for section in ("dataset", "evaluation", "output"):
        if not isinstance(settings.get(section), dict):
            raise ValueError(f"設定ファイルに '{section}' セクションが必要です。")
    return settings


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def coco_bbox_to_clipped_xyxy(bbox: list[Any], image_width: int, image_height: int) -> list[float]:
    try:
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("COCO bboxは数値4個で指定してください。") from exc
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("COCO bboxの座標は有限値で指定してください。")
    x1 = min(max(x, 0.0), float(image_width))
    y1 = min(max(y, 0.0), float(image_height))
    x2 = min(max(x + width, 0.0), float(image_width))
    y2 = min(max(y + height, 0.0), float(image_height))
    if x1 >= x2 or y1 >= y2:
        raise ValueError("COCO bboxは画像内に正の面積を持つ必要があります。")
    return [x1, y1, x2, y2]


def load_coco_annotations(annotation_path: Path) -> dict[str, Any]:
    try:
        with annotation_path.open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"アノテーションファイルを読み込めません: {annotation_path}") from exc
    if not isinstance(data, dict) or not {"images", "annotations", "categories"}.issubset(data):
        raise ValueError("COCO JSONにはimages、annotations、categoriesが必要です。")
    return data


def select_images(images: list[dict[str, Any]], dataset_settings: dict[str, Any]) -> list[dict[str, Any]]:
    selected = images
    image_ids = dataset_settings.get("image_ids")
    if image_ids is not None:
        selected_ids = {int(image_id) for image_id in image_ids}
        selected = [image for image in selected if int(image["id"]) in selected_ids]
    limit = dataset_settings.get("image_limit")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("dataset.image_limitは正の整数またはnullにしてください。")
        selected = selected[:limit]
    return selected


def build_annotations_by_image(annotations: list[dict[str, Any]], include_crowd: bool) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        if include_crowd or annotation.get("iscrowd", 0) != 1:
            grouped[int(annotation["image_id"])].append(annotation)
    return grouped


def annotation_to_mask(annotation: dict[str, Any], height: int, width: int) -> dict[str, Any]:
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, dict):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    elif isinstance(segmentation, list) and segmentation:
        rle = mask_utils.frPyObjects(segmentation, height, width)
        if isinstance(rle, list):
            rle = mask_utils.merge(rle)
    else:
        raise ValueError(f"アノテーションID={annotation.get('id')}に有効なsegmentationがありません。")
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    if decoded.shape != (height, width):
        raise ValueError(f"アノテーションID={annotation.get('id')}のmask寸法が画像と一致しません。")
    return _json_rle(mask_utils.encode(np.asfortranarray(decoded.astype(np.uint8))))


def _json_rle(rle: dict[str, Any]) -> dict[str, Any]:
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(value) for value in rle["size"]], "counts": str(counts)}


def limit_predictions_per_category(predictions: list[dict[str, Any]], limit: int = 100) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        grouped[int(prediction["category_id"])].append(prediction)
    return [item for category_predictions in grouped.values() for item in sorted(category_predictions, key=lambda value: float(value["confidence"]), reverse=True)[:limit]]


def prediction_to_record(image_id: int, mode: str, prediction: dict[str, Any]) -> dict[str, Any]:
    source_box = prediction.get("source_xyxy")
    return {
        "image_id": image_id,
        "mode": mode,
        "category_id": int(prediction["category_id"]),
        "category_name": prediction["category_name"],
        "prompt": prediction["prompt"],
        "confidence": float(prediction["confidence"]),
        "xyxy": [float(value) for value in prediction["xyxy"]],
        "source_xyxy": [float(value) for value in source_box] if source_box is not None else None,
    }


def _new_run(results_root: Path, configured_id: Any) -> tuple[str, Path]:
    results_root.mkdir(parents=True, exist_ok=True)
    if configured_id is None or configured_id == "auto":
        index = 1
        while (results_root / f"run_{index:05d}").exists():
            index += 1
        run_id = f"run_{index:05d}"
    else:
        if not isinstance(configured_id, str) or re.fullmatch(r"run_\d{5}", configured_id) is None:
            raise ValueError("run_idは'auto'またはrun_00001形式で指定してください。")
        run_id = configured_id
    run_directory = results_root / run_id
    try:
        run_directory.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"指定されたrunは既に存在します。追記できません: {run_id}") from exc
    return run_id, run_directory


def run_inference() -> None:
    settings = load_settings()
    dataset = settings["dataset"]
    evaluation = settings["evaluation"]
    output = settings["output"]
    annotation_value = dataset.get("annotation_path")
    image_dir_value = dataset.get("image_dir")
    if not isinstance(annotation_value, str) or not isinstance(image_dir_value, str):
        raise ValueError("dataset.annotation_pathとdataset.image_dirを指定してください。")
    annotation_path = resolve_project_path(annotation_value)
    image_dir = resolve_project_path(image_dir_value)
    coco = load_coco_annotations(annotation_path)
    images = select_images(coco["images"], dataset)
    if not images:
        raise ValueError("推論対象画像がありません。設定を確認してください。")
    modes = evaluation.get("modes", ["no_bbox"])
    if not isinstance(modes, list) or not modes or not set(modes).issubset({"no_bbox", "bbox"}):
        raise ValueError("evaluation.modesにはno_bbox、bboxのいずれかを指定してください。")
    thresholds = evaluation.get("iou_thresholds")
    report_thresholds = evaluation.get("report_thresholds")
    pr_threshold = evaluation.get("pr_confidence_threshold", 0.5)
    if not isinstance(thresholds, list) or not thresholds or not isinstance(report_thresholds, list) or not report_thresholds or any(value not in thresholds for value in report_thresholds):
        raise ValueError("evaluationのIoUしきい値設定が不正です。")
    for value in [*thresholds, *report_thresholds, pr_threshold]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("IoU・confidenceしきい値は0から1の数値にしてください。")
    results_value = output.get("results_dir", "results")
    if not isinstance(results_value, str):
        raise ValueError("output.results_dirは文字列で指定してください。")
    annotations_by_image = build_annotations_by_image(coco["annotations"], bool(dataset.get("include_crowd", False)))
    categories = coco["categories"]
    category_ids = {int(category["id"]) for category in categories}
    for annotation in coco["annotations"]:
        if int(annotation["category_id"]) not in category_ids:
            raise ValueError(f"categoriesに定義されていないcategory_idがあります: {annotation['category_id']}")
    started_at = datetime.now(UTC).isoformat()
    snapshot = {"started_at": started_at, "settings": settings}
    run_id, run_directory = _new_run(resolve_project_path(results_value), settings.get("run_id", "auto"))
    LOGGER.info("結果保存先: %s", run_directory)
    database_path = run_directory / "inference.sqlite3"
    connection = create_database(
        database_path,
        run_id,
        snapshot,
        str(annotation_path),
        str(image_dir),
        list(modes),
        "initializing",
        categories,
    )
    try:
        segmenter = Sam3Segmenter()
        connection.execute("UPDATE run_info SET device = ?", (segmenter.device,))
        connection.commit()
        skipped = 0
        for index, image_info in enumerate(images, start=1):
            image_id = int(image_info["id"])
            image_path = image_dir / str(image_info["file_name"])
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                LOGGER.warning("画像を読み込めないためスキップします: %s", image_path)
                increment_skipped(connection)
                skipped += 1
                continue
            height, width = image.shape[:2]
            annotations = annotations_by_image.get(image_id, [])
            ground_truths = []
            clipped_boxes = []
            for annotation in annotations:
                bbox = coco_bbox_to_clipped_xyxy(annotation["bbox"], width, height)
                clipped_boxes.append(bbox)
                ground_truths.append(
                    {
                        "annotation_id": int(annotation["id"]),
                        "category_id": int(annotation["category_id"]),
                        "xyxy": bbox,
                        "mask": annotation_to_mask(annotation, height, width),
                        "iscrowd": int(annotation.get("iscrowd", 0)),
                    }
                )
            predictions_for_image = []
            for mode in modes:
                if mode == "no_bbox":
                    prompt_ids = sorted({int(item["category_id"]) for item in annotations})
                    if not prompt_ids:
                        continue
                    predictions = segmenter.infer(image=image, category_ids=prompt_ids)
                else:
                    prompt_ids = [int(item["category_id"]) for item in annotations]
                    if not prompt_ids:
                        continue
                    predictions = segmenter.infer(image=image, category_ids=prompt_ids, bboxes=clipped_boxes)
                for prediction in limit_predictions_per_category(predictions):
                    mask = np.asarray(prediction["mask"], dtype=np.uint8)
                    if mask.shape != (height, width):
                        raise RuntimeError("SAM3のマスク形状が入力画像と一致しません。")
                    predictions_for_image.append(
                        {
                            "mode": mode,
                            "category_id": int(prediction["category_id"]),
                            "category_name": str(prediction["category_name"]),
                            "prompt": str(prediction["prompt"]),
                            "confidence": float(prediction["confidence"]),
                            "xyxy": [float(value) for value in prediction["xyxy"]],
                            "source_xyxy": ([float(value) for value in prediction["source_xyxy"]] if prediction.get("source_xyxy") is not None else None),
                            "mask": _json_rle(mask_utils.encode(np.asfortranarray(mask))),
                        }
                    )
            save_image_transaction(
                connection,
                {
                    "image_id": image_id,
                    "image_path": str(image_path.resolve()),
                    "width": width,
                    "height": height,
                    "processed_at": datetime.now(UTC).isoformat(),
                },
                ground_truths,
                predictions_for_image,
            )
            LOGGER.info("[%d/%d] image_id=%d を保存しました", index, len(images), image_id)
        connection.execute("UPDATE run_info SET state = 'complete'")
        connection.commit()
        run_info = {
            "run_id": run_id,
            "started_at": started_at,
            "state": "complete",
            "successful_image_count": connection.execute("SELECT successful_image_count FROM run_info").fetchone()[0],
            "skipped_image_count": skipped,
            "device": segmenter.device,
            "database": str(database_path),
        }
        with (run_directory / "run_info.json").open("w", encoding="utf-8") as file:
            json.dump(run_info, file, ensure_ascii=False, indent=2)
        LOGGER.info("推論完了: %s", run_directory)
    except Exception:
        connection.execute("UPDATE run_info SET state = 'incomplete'")
        connection.commit()
        raise
    finally:
        connection.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_inference()


if __name__ == "__main__":
    main()
