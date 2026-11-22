"""settings_sam3.yamlの設定に従ってSAM3推論を実行する。"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sam3_inference.evaluation import (
    calculate_average_precision,
    calculate_prf1_by_image,
)
from src.sam3_inference.inference_categories import Sam3Segmenter
from src.sam3_inference.reporting import (
    select_visualization_images,
    write_metrics_csv,
    write_overlays,
    write_summary_images,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "configs" / "settings_sam3.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger(__name__)


def load_settings(settings_path: Path) -> dict[str, Any]:
    """YAML設定ファイルを読み込む。

    Args:
        settings_path: 設定ファイルのパス。

    Returns:
        YAMLから読み込んだ設定辞書。

    Raises:
        RuntimeError: ファイルの読み込みまたはYAML解析に失敗した場合。
        ValueError: 設定のトップレベルが辞書でない場合。
    """
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
    """プロジェクトルートを基準にパスを解決する。

    Args:
        value: 絶対パス、またはプロジェクトルートからの相対パス。

    Returns:
        解決済みのパス。
    """
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def coco_bbox_to_clipped_xyxy(
    bbox: list[Any],
    image_width: int,
    image_height: int,
) -> list[float]:
    """COCO形式のbboxを画像内のXYXY座標に変換する。"""
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
    """COCO形式アノテーションJSONを読み込む。

    Args:
        annotation_path: アノテーションJSONのパス。

    Returns:
        COCOデータセット辞書。
    """
    try:
        with annotation_path.open(encoding="utf-8") as file:
            coco_data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"アノテーションファイルを読み込めません: {annotation_path}") from exc

    required_keys = {"images", "annotations", "categories"}
    if not isinstance(coco_data, dict) or not required_keys.issubset(coco_data):
        raise ValueError("COCO JSONにはimages、annotations、categoriesが必要です。")

    return coco_data


def select_images(
    images: list[dict[str, Any]],
    dataset_settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """image_idsとimage_limitの設定に従って推論対象画像を選択する。

    Args:
        images: COCO JSONのimages配列。
        dataset_settings: datasetセクションの設定。

    Returns:
        選択された画像情報のリスト。
    """
    selected_images = images
    image_ids = dataset_settings.get("image_ids")

    if image_ids is not None:
        selected_ids = {int(image_id) for image_id in image_ids}
        selected_images = [image for image in selected_images if int(image["id"]) in selected_ids]

    image_limit = dataset_settings.get("image_limit")
    if image_limit is not None:
        if isinstance(image_limit, bool) or not isinstance(image_limit, int) or image_limit < 1:
            raise ValueError("dataset.image_limitは正の整数またはnullにしてください。")
        selected_images = selected_images[:image_limit]

    return selected_images


def create_run_directory(results_root: Path) -> Path:
    """results/run_00001形式の未使用ディレクトリを作成する。

    Args:
        results_root: 実行結果ルートディレクトリ。

    Returns:
        新規作成した実行結果ディレクトリ。
    """
    results_root.mkdir(parents=True, exist_ok=True)

    run_index = 1
    while True:
        run_directory = results_root / f"run_{run_index:05d}"
        try:
            run_directory.mkdir()
            return run_directory
        except FileExistsError:
            run_index += 1


def build_annotations_by_image(
    annotations: list[dict[str, Any]],
    include_crowd: bool,
) -> dict[int, list[dict[str, Any]]]:
    """画像IDをキーとしてアノテーションをまとめる。

    Args:
        annotations: COCO JSONのannotations配列。
        include_crowd: iscrowd=1のアノテーションを含めるか。

    Returns:
        画像IDごとのアノテーション辞書。
    """
    grouped_annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for annotation in annotations:
        if not include_crowd and annotation.get("iscrowd", 0) == 1:
            continue
        grouped_annotations[int(annotation["image_id"])].append(annotation)

    return grouped_annotations


def prediction_to_record(
    image_id: int,
    mode: str,
    prediction: dict[str, Any],
) -> dict[str, Any]:
    """推論結果からマスクを除いたJSON保存用レコードを作成する。

    Args:
        image_id: COCO画像ID。
        mode: 推論モード。
        prediction: Sam3Segmenter.infer()の検出結果。

    Returns:
        JSONシリアライズ可能な予測レコード。
    """
    source_bbox = prediction["source_xyxy"]

    return {
        "image_id": image_id,
        "mode": mode,
        "category_id": int(prediction["category_id"]),
        "category_name": prediction["category_name"],
        "prompt": prediction["prompt"],
        "confidence": float(prediction["confidence"]),
        "xyxy": [float(value) for value in prediction["xyxy"]],
        "source_xyxy": ([float(value) for value in source_bbox] if source_bbox is not None else None),
    }


def annotation_to_mask(annotation: dict[str, Any], height: int, width: int) -> dict[str, Any]:
    """COCOアノテーションのsegmentationを圧縮RLEに変換する。"""
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
    return mask_utils.encode(np.asfortranarray(decoded.astype(np.uint8)))


def _validate_thresholds(settings: dict[str, Any]) -> tuple[list[float], list[float], float]:
    evaluation_settings = settings["evaluation"]
    thresholds = evaluation_settings.get("iou_thresholds")
    reports = evaluation_settings.get("report_thresholds")
    confidence = evaluation_settings.get("pr_confidence_threshold", 0.5)
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError("evaluation.iou_thresholdsは0から1の数値を含むリストにしてください。")
    if not isinstance(reports, list) or not reports:
        raise ValueError("evaluation.report_thresholdsはevaluation.iou_thresholdsに含まれる値を指定してください。")
    for value in [*thresholds, *reports, confidence]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("IoU・confidenceしきい値は0から1の数値にしてください。")
    thresholds = [float(value) for value in thresholds]
    reports = [float(value) for value in reports]
    if any(value not in thresholds for value in reports):
        raise ValueError("evaluation.report_thresholdsの値はevaluation.iou_thresholdsに含めてください。")
    return thresholds, reports, float(confidence)


def _build_metric_rows(
    modes: list[str],
    categories: list[dict[str, Any]],
    ground_truth_by_image: dict[int, list[dict[str, Any]]],
    predictions_by_mode: dict[str, dict[int, list[dict[str, Any]]]],
    thresholds: list[float],
    report_thresholds: list[float],
    pr_confidence_threshold: float,
    label_ja: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    category_names = {int(category["id"]): str(category["name"]) for category in categories}
    for mode in modes:
        predictions_by_image = predictions_by_mode[mode]
        for geometry in ("bbox", "mask"):
            per_category = {}
            for category in categories:
                category_id = int(category["id"])
                gt_count = sum(int(item["category_id"]) == category_id for items in ground_truth_by_image.values() for item in items)
                prediction_count = sum(int(item["category_id"]) == category_id for items in predictions_by_image.values() for item in items)
                ap_by_threshold = {
                    threshold: calculate_average_precision(
                        ground_truth_by_image,
                        predictions_by_image,
                        category_id,
                        threshold,
                        geometry,
                    )
                    for threshold in thresholds
                }
                pr_by_threshold = {
                    threshold: calculate_prf1_by_image(
                        ground_truth_by_image,
                        predictions_by_image,
                        category_id,
                        threshold,
                        geometry,
                        pr_confidence_threshold,
                    )
                    for threshold in report_thresholds
                }
                category_name = category_names[category_id]
                metric_data = {
                    "category_id": category_id,
                    "category_name": category_name,
                    "gt_count": gt_count,
                    "prediction_count": prediction_count,
                    "ap_by_threshold": ap_by_threshold,
                    "pr_by_threshold": pr_by_threshold,
                }
                per_category[category_id] = metric_data
                translated_name = label_ja.get(category_name, category_name)
                rows.append(
                    _metric_row(
                        mode,
                        geometry,
                        "category",
                        category_id,
                        translated_name,
                        [metric_data],
                        thresholds,
                        report_thresholds,
                        "bboxは正解bboxを入力する条件付き評価" if mode == "bbox" else "カテゴリプロンプトのみ",
                    )
                )

            included = [item for item in per_category.values() if item["gt_count"] > 0]
            rows.append(
                _metric_row(
                    mode,
                    geometry,
                    "overall",
                    None,
                    "全体",
                    included,
                    thresholds,
                    report_thresholds,
                    "bboxは正解bboxを入力する条件付き評価" if mode == "bbox" else "カテゴリプロンプトのみ",
                )
            )
    return rows


def _metric_row(
    mode: str,
    geometry: str,
    scope: str,
    category_id: int | None,
    category_name: str,
    items: list[dict[str, Any]],
    thresholds: list[float],
    report_thresholds: list[float],
    evaluation_condition: str,
) -> dict[str, Any]:
    ap_means = {threshold: _mean(item["ap_by_threshold"][threshold] for item in items) for threshold in thresholds}
    row: dict[str, Any] = {
        "mode": mode,
        "evaluation_condition": evaluation_condition,
        "geometry": geometry,
        "scope": scope,
        "category_id": category_id,
        "category_name": category_name,
        "gt_count": sum(item["gt_count"] for item in items),
        "prediction_count": sum(item["prediction_count"] for item in items),
        "ap50_95": _mean(ap_means.values()),
        "ap50": ap_means.get(0.5),
        "ap75": ap_means.get(0.75),
    }
    for threshold, value in ap_means.items():
        row[f"ap_iou_{threshold:.2f}"] = value
    for threshold in report_thresholds:
        key = f"{threshold:.2f}"
        pr_items = [item["pr_by_threshold"][threshold] for item in items]
        row[f"tp_iou_{key}"] = _mean(item["counts"]["tp"] for item in pr_items)
        row[f"fp_iou_{key}"] = _mean(item["counts"]["fp"] for item in pr_items)
        row[f"fn_iou_{key}"] = _mean(item["counts"]["fn"] for item in pr_items)
        for name in ("precision", "recall", "f1"):
            row[f"{name}_iou_{key}"] = _mean(item[name] for item in pr_items)
    if category_id is not None:
        item = items[0]
        row["ap50_95"] = _mean(item["ap_by_threshold"].values())
        row["ap50"] = item["ap_by_threshold"].get(0.5)
        row["ap75"] = item["ap_by_threshold"].get(0.75)
    if report_thresholds and not items:
        for threshold in report_thresholds:
            for name in ("tp", "fp", "fn", "precision", "recall", "f1"):
                row[f"{name}_iou_{threshold:.2f}"] = 0.0
    return row


def _mean(values: Any) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return float(np.mean(numeric)) if numeric else None


def limit_predictions_per_category(
    predictions: list[dict[str, Any]],
    limit: int = 100,
) -> list[dict[str, Any]]:
    """画像内のカテゴリごとにconfidence上位の予測を残す。"""
    by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        by_category[int(prediction["category_id"])].append(prediction)
    return [
        prediction
        for category_predictions in by_category.values()
        for prediction in sorted(
            category_predictions,
            key=lambda item: float(item["confidence"]),
            reverse=True,
        )[:limit]
    ]


def run_inference() -> None:
    """YAML設定に従って推論、評価、可視化を行う。"""
    settings = load_settings(SETTINGS_PATH)
    dataset_settings = settings["dataset"]
    evaluation_settings = settings["evaluation"]
    output_settings = settings["output"]

    annotation_value = dataset_settings.get("annotation_path")
    image_dir_value = dataset_settings.get("image_dir")
    if not isinstance(annotation_value, str) or not isinstance(image_dir_value, str):
        raise ValueError("dataset.annotation_pathとdataset.image_dirを指定してください。")

    annotation_path = resolve_project_path(annotation_value)
    image_dir = resolve_project_path(image_dir_value)
    coco_data = load_coco_annotations(annotation_path)

    images = select_images(coco_data["images"], dataset_settings)
    if not images:
        raise ValueError("推論対象画像がありません。設定を確認してください。")

    include_crowd = bool(dataset_settings.get("include_crowd", False))
    annotations_by_image = build_annotations_by_image(
        coco_data["annotations"],
        include_crowd,
    )

    modes = evaluation_settings.get("modes", ["no_bbox"])
    valid_modes = {"no_bbox", "bbox"}
    if not isinstance(modes, list) or not modes or not set(modes).issubset(valid_modes):
        raise ValueError("evaluation.modesにはno_bbox、bboxのいずれかを指定してください。")
    thresholds, report_thresholds, pr_confidence_threshold = _validate_thresholds(settings)

    results_dir_value = output_settings.get("results_dir", "results")
    if not isinstance(results_dir_value, str):
        raise ValueError("output.results_dirは文字列で指定してください。")
    visualization_samples = output_settings.get("visualization_samples", 20)
    worst_best_samples = output_settings.get("worst_best_samples", 10)
    for key, value in (("visualization_samples", visualization_samples), ("worst_best_samples", worst_best_samples)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"output.{key}は0以上の整数にしてください。")
    label_ja = output_settings.get("label_ja", {})
    if not isinstance(label_ja, dict):
        raise ValueError("output.label_jaはカテゴリ名と日本語ラベルの対応にしてください。")

    run_directory = create_run_directory(resolve_project_path(results_dir_value))
    LOGGER.info("結果保存先: %s", run_directory)
    LOGGER.info("対象画像数: %d", len(images))
    LOGGER.info("推論モード: %s", ", ".join(modes))

    segmenter = Sam3Segmenter()
    prediction_records: list[dict[str, Any]] = []
    predictions_by_mode: dict[str, dict[int, list[dict[str, Any]]]] = {mode: {int(image["id"]): [] for image in images} for mode in modes}
    ground_truth_by_image: dict[int, list[dict[str, Any]]] = {}
    processed_images = []
    processed_image_count = 0

    for image_index, image_info in enumerate(images, start=1):
        image_id = int(image_info["id"])
        image_path = image_dir / str(image_info["file_name"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            LOGGER.warning("画像を読み込めないためスキップします: %s", image_path)
            continue

        processed_image_count += 1
        processed_images.append(image_info)
        image_annotations = annotations_by_image.get(image_id, [])
        image_height, image_width = image.shape[:2]
        ground_truth = []
        clipped_bboxes = []
        for annotation_index, annotation in enumerate(image_annotations):
            try:
                bbox = coco_bbox_to_clipped_xyxy(
                    annotation["bbox"],
                    image_width,
                    image_height,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                LOGGER.error(
                    "無効なGT bboxです: image_id=%s, annotation_id=%s, index=%d, bbox=%s, image_size=(%d, %d)",
                    image_id,
                    annotation.get("id"),
                    annotation_index,
                    annotation.get("bbox"),
                    image_width,
                    image_height,
                )
                raise ValueError(f"画像内に正の面積を持たないGT bboxです: image_id={image_id}, annotation_id={annotation.get('id')}") from exc
            clipped_bboxes.append(bbox)
            ground_truth.append(
                {
                    "image_id": image_id,
                    "category_id": int(annotation["category_id"]),
                    "xyxy": bbox,
                    "mask": annotation_to_mask(annotation, image_height, image_width),
                }
            )
        ground_truth_by_image[image_id] = ground_truth

        LOGGER.info(
            "[%d/%d] image_id=%d file=%s",
            image_index,
            len(images),
            image_id,
            image_info["file_name"],
        )

        for mode in modes:
            if mode == "no_bbox":
                category_ids = sorted({int(annotation["category_id"]) for annotation in image_annotations})
                if not category_ids:
                    continue
                predictions = segmenter.infer(image=image, category_ids=category_ids)
            else:
                category_ids = []
                bboxes = clipped_bboxes
                category_ids.extend(int(annotation["category_id"]) for annotation in image_annotations)
                if not category_ids:
                    continue
                predictions = segmenter.infer(image=image, category_ids=category_ids, bboxes=bboxes)

            predictions = limit_predictions_per_category(predictions)
            for prediction in predictions:
                prediction_records.append(prediction_to_record(image_id, mode, prediction))
                prediction_with_mask = {
                    "image_id": image_id,
                    "category_id": int(prediction["category_id"]),
                    "confidence": float(prediction["confidence"]),
                    "xyxy": [float(value) for value in prediction["xyxy"]],
                    "source_xyxy": prediction.get("source_xyxy"),
                    "mask": mask_utils.encode(np.asfortranarray(np.asarray(prediction["mask"], dtype=np.uint8))),
                }
                predictions_by_mode[mode][image_id].append(prediction_with_mask)

    category_ids = {int(category["id"]) for category in coco_data["categories"]}
    unknown_ids = {int(item["category_id"]) for items in ground_truth_by_image.values() for item in items} - category_ids
    if unknown_ids:
        raise ValueError(f"categoriesに定義されていないcategory_idがあります: {sorted(unknown_ids)}")
    rows = _build_metric_rows(
        modes,
        coco_data["categories"],
        ground_truth_by_image,
        predictions_by_mode,
        thresholds,
        report_thresholds,
        pr_confidence_threshold,
        label_ja,
    )

    mode_directories = {}
    for mode in modes:
        mode_directory = run_directory / mode
        mode_directory.mkdir(parents=True, exist_ok=True)
        mode_directories[mode] = mode_directory
        mode_prediction_records = [record for record in prediction_records if record["mode"] == mode]
        with (mode_directory / "predictions.json").open("w", encoding="utf-8") as file:
            json.dump(mode_prediction_records, file, ensure_ascii=False, indent=2)
        write_metrics_csv(mode_directory / "metrics.csv", [row for row in rows if row["mode"] == mode])

    font_path = write_summary_images(
        run_directory,
        rows,
        output_settings,
        coco_data["categories"],
        PROJECT_ROOT,
        report_thresholds,
        thresholds,
        pr_confidence_threshold,
    )
    overlay_counts = {}
    category_names = {int(category["id"]): str(category["name"]) for category in coco_data["categories"]}
    per_mode_sample_counts = [visualization_samples // len(modes) + int(index < visualization_samples % len(modes)) for index in range(len(modes))]
    for mode_index, mode in enumerate(modes):
        per_mode_sample_count = per_mode_sample_counts[mode_index]
        per_mode_worst_best = min(worst_best_samples, per_mode_sample_count // 2)
        selected_for_visualization = (
            select_visualization_images(
                processed_images,
                ground_truth_by_image,
                predictions_by_mode[mode],
                per_mode_sample_count,
                per_mode_worst_best,
                pr_confidence_threshold,
            )
            if per_mode_sample_count
            else []
        )
        overlay_counts[mode] = write_overlays(
            run_directory,
            mode,
            selected_for_visualization,
            image_dir,
            ground_truth_by_image,
            predictions_by_mode[mode],
            category_names,
            label_ja,
            font_path,
            int(output_settings.get("dpi", 150)),
            pr_confidence_threshold,
        )

    run_info = {
        "annotation_path": str(annotation_path),
        "image_dir": str(image_dir),
        "modes": modes,
        "mode_conditions": {
            "no_bbox": "画像のGTカテゴリのみをプロンプトとして評価",
            "bbox": "GT bboxとカテゴリを入力する条件付き評価（通常の検出性能ではない）",
        },
        "selected_image_count": len(images),
        "processed_image_count": processed_image_count,
        "prediction_count": len(prediction_records),
        "confidence_threshold": segmenter.confidence_threshold,
        "pr_confidence_threshold": pr_confidence_threshold,
        "iou_thresholds": thresholds,
        "report_thresholds": report_thresholds,
        "prediction_count_by_mode": {mode: sum(len(items) for items in predictions_by_mode[mode].values()) for mode in modes},
        "overlay_count_by_mode": overlay_counts,
        "counts_by_mode": {
            mode: {
                "selected_image_count": len(images),
                "processed_image_count": processed_image_count,
                "ground_truth_count": sum(len(items) for items in ground_truth_by_image.values()),
                "prediction_count": sum(len(items) for items in predictions_by_mode[mode].values()),
                "overlay_count": overlay_counts[mode],
            }
            for mode in modes
        },
        "mode_output_directories": {mode: str(directory) for mode, directory in mode_directories.items()},
        "japanese_font_path": str(font_path),
        "device": segmenter.device,
    }
    run_info_path = run_directory / "run_info.json"
    with run_info_path.open("w", encoding="utf-8") as file:
        json.dump(run_info, file, ensure_ascii=False, indent=2)

    LOGGER.info("推論完了: %d件の予測", len(prediction_records))
    LOGGER.info("条件別成果物: %s", run_directory)
    LOGGER.info("条件比較画像: %s", run_directory / "comparison_summary.png")
    LOGGER.info("実行情報: %s", run_info_path)


def main() -> None:
    """ログを初期化して推論を実行する。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_inference()


if __name__ == "__main__":
    main()
