"""保存済みSAM3推論DBのみを用いて評価・可視化を行う。"""

from __future__ import annotations

import json
import csv
import logging
import re
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.main_infer import SETTINGS_PATH, load_settings, resolve_project_path
from src.sam3_inference.evaluation import (
    calculate_average_precision,
    calculate_prf1,
    calculate_prf1_by_image,
    intersection_over_union,
)
from src.sam3_inference.reporting import (
    select_visualization_images,
    write_metrics_csv,
    write_overlays,
    write_summary_images,
    write_comparison_outputs,
)
from src.sam3_inference.storage import SCHEMA_VERSION

LOGGER = logging.getLogger(__name__)
_REQUIRED_COLUMNS = {
    "run_info": {"run_id", "schema_version", "started_at", "settings_json", "annotation_path", "image_dir", "modes_json", "state", "device", "successful_image_count", "skipped_image_count"},
    "images": {"image_id", "image_path", "width", "height", "status", "processed_at"},
    "categories": {"category_id", "name"},
    "ground_truths": {"annotation_id", "image_id", "category_id", "bbox_json", "mask_json", "iscrowd"},
    "predictions": {"image_id", "mode", "category_id", "category_name", "prompt", "confidence", "bbox_json", "source_bbox_json", "mask_json"},
}


def _read_database(database_path: Path) -> dict[str, Any]:
    if not database_path.is_file():
        raise RuntimeError(f"指定runの推論DBが見つかりません: {database_path}")
    try:
        connection = sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        tables = {row["name"] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        missing_tables = set(_REQUIRED_COLUMNS) - tables
        if missing_tables:
            raise RuntimeError(f"推論DBのスキーマが不一致です。不足テーブル: {sorted(missing_tables)}")
        for table, expected in _REQUIRED_COLUMNS.items():
            columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
            if not expected.issubset(columns):
                raise RuntimeError(f"推論DBのスキーマが不一致です: {table}")
        run_info = connection.execute("SELECT * FROM run_info").fetchone()
        if run_info is None:
            raise RuntimeError("推論DBにrun_infoがありません。")
        if int(run_info["schema_version"]) != SCHEMA_VERSION:
            raise RuntimeError(f"未対応の推論DBスキーマです: {run_info['schema_version']}")
        settings = json.loads(run_info["settings_json"])
        modes = json.loads(run_info["modes_json"])
        if not isinstance(settings, dict) or not isinstance(modes, list) or not modes or not set(modes).issubset({"no_bbox", "bbox"}):
            raise RuntimeError("推論DBの設定スナップショットが不正です。")
        categories = [{"id": int(row["category_id"]), "name": str(row["name"])} for row in connection.execute("SELECT category_id, name FROM categories ORDER BY category_id")]
        images = [dict(row) for row in connection.execute("SELECT image_id, image_path, width, height, processed_at FROM images ORDER BY image_id")]
        ground_truth_by_image: dict[int, list[dict[str, Any]]] = {int(item["image_id"]): [] for item in images}
        image_sizes = {int(item["image_id"]): [int(item["height"]), int(item["width"])] for item in images}
        for row in connection.execute("SELECT * FROM ground_truths ORDER BY image_id, annotation_id"):
            item = {
                "annotation_id": int(row["annotation_id"]),
                "image_id": int(row["image_id"]),
                "category_id": int(row["category_id"]),
                "xyxy": json.loads(row["bbox_json"]),
                "mask": json.loads(row["mask_json"]),
                "iscrowd": int(row["iscrowd"]),
            }
            if (
                not isinstance(item["xyxy"], list)
                or len(item["xyxy"]) != 4
                or not isinstance(item["mask"], dict)
                or not isinstance(item["mask"].get("size"), list)
                or not isinstance(item["mask"].get("counts"), str)
                or item["mask"]["size"] != image_sizes[int(row["image_id"])]
            ):
                raise RuntimeError("推論DBのGT bboxまたは圧縮RLEが不正です。")
            ground_truth_by_image[int(row["image_id"])].append(item)
        predictions_by_mode: dict[str, dict[int, list[dict[str, Any]]]] = {mode: {int(image["image_id"]): [] for image in images} for mode in modes}
        for row in connection.execute("SELECT * FROM predictions ORDER BY prediction_id"):
            mode = str(row["mode"])
            if mode not in predictions_by_mode:
                raise RuntimeError(f"推論DBに設定外のmodeがあります: {mode}")
            item = {
                "category_id": int(row["category_id"]),
                "confidence": float(row["confidence"]),
                "xyxy": json.loads(row["bbox_json"]),
                "source_xyxy": json.loads(row["source_bbox_json"]) if row["source_bbox_json"] is not None else None,
                "mask": json.loads(row["mask_json"]),
            }
            if (
                not isinstance(item["xyxy"], list)
                or len(item["xyxy"]) != 4
                or not isinstance(item["mask"], dict)
                or not isinstance(item["mask"].get("size"), list)
                or not isinstance(item["mask"].get("counts"), str)
                or item["mask"]["size"] != image_sizes[int(row["image_id"])]
            ):
                raise RuntimeError("推論DBの予測bboxまたは圧縮RLEが不正です。")
            predictions_by_mode[mode][int(row["image_id"])].append(item)
        return {
            "run_info": dict(run_info),
            "settings": settings,
            "modes": modes,
            "categories": categories,
            "images": images,
            "ground_truth_by_image": ground_truth_by_image,
            "predictions_by_mode": predictions_by_mode,
        }
    except RuntimeError:
        raise
    except (sqlite3.Error, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"推論DBが破損しているか読み込めません: {database_path}") from exc
    finally:
        if "connection" in locals():
            connection.close()


def _build_metric_rows(
    modes: list[str],
    categories: list[dict[str, Any]],
    ground_truth_by_image: dict[int, list[dict[str, Any]]],
    predictions_by_mode: dict[str, dict[int, list[dict[str, Any]]]],
    thresholds: list[float],
    report_thresholds: list[float],
    pr_confidence_threshold: float,
    label_ja: dict[str, str],
    supplementary_exclude_categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    category_names = {int(item["id"]): str(item["name"]) for item in categories}
    for mode in modes:
        predictions = predictions_by_mode[mode]
        for geometry in ("bbox", "mask"):
            metrics = []
            for category in categories:
                category_id = int(category["id"])
                gt_count = sum(int(item["category_id"]) == category_id for values in ground_truth_by_image.values() for item in values)
                prediction_count = sum(int(item["category_id"]) == category_id for values in predictions.values() for item in values)
                ap = {threshold: calculate_average_precision(ground_truth_by_image, predictions, category_id, threshold, geometry) for threshold in thresholds}
                pr = {
                    threshold: calculate_prf1_by_image(
                        ground_truth_by_image,
                        predictions,
                        category_id,
                        threshold,
                        geometry,
                        pr_confidence_threshold,
                    )
                    for threshold in report_thresholds
                }
                record = {
                    "category_id": category_id,
                    "category_name": category_names[category_id],
                    "gt_count": gt_count,
                    "prediction_count": prediction_count,
                    "ap": ap,
                    "pr": pr,
                }
                metrics.append(record)
                rows.append(
                    _metric_row(
                        mode,
                        geometry,
                        "category",
                        category_id,
                        label_ja.get(category_names[category_id], category_names[category_id]),
                        [record],
                        thresholds,
                        report_thresholds,
                    )
                )
            included = [record for record in metrics if record["gt_count"] > 0]
            rows.append(_metric_row(mode, geometry, "overall", None, "全体", included, thresholds, report_thresholds))
            if included:
                micro = {
                    "mode": mode,
                    "evaluation_condition": "bboxは正解bboxを入力する条件付き評価" if mode == "bbox" else "カテゴリプロンプトのみ",
                    "geometry": geometry,
                    "scope": "micro",
                    "category_id": None,
                    "category_name": "全GTマイクロ平均",
                    "gt_count": sum(record["gt_count"] for record in included),
                    "prediction_count": sum(record["prediction_count"] for record in included),
                }
                gt_counts = np.asarray([record["gt_count"] for record in included], dtype=float)
                for threshold in thresholds:
                    micro[f"ap_iou_{threshold:.2f}"] = float(np.average(
                        [record["ap"][threshold] for record in included],
                        weights=gt_counts,
                    ))
                micro["ap50_95"] = float(np.mean([
                    micro[f"ap_iou_{threshold:.2f}"] for threshold in thresholds
                ]))
                micro["ap50"] = micro.get("ap_iou_0.50")
                micro["ap75"] = micro.get("ap_iou_0.75")
                for threshold in report_thresholds:
                    tp = sum(record["pr"][threshold]["counts"]["tp"] for record in included)
                    fp = sum(record["pr"][threshold]["counts"]["fp"] for record in included)
                    fn = sum(record["pr"][threshold]["counts"]["fn"] for record in included)
                    pr = calculate_prf1(tp, fp, fn)
                    key = f"{threshold:.2f}"
                    micro[f"tp_iou_{key}"] = tp
                    micro[f"fp_iou_{key}"] = fp
                    micro[f"fn_iou_{key}"] = fn
                    micro[f"precision_iou_{key}"] = pr["precision"]
                    micro[f"recall_iou_{key}"] = pr["recall"]
                    micro[f"f1_iou_{key}"] = pr["f1"]
                rows.append(micro)
            excluded_names = set(supplementary_exclude_categories or [])
            supplementary = [
                record for record in included
                if record["category_name"] not in excluded_names
            ]
            if excluded_names and supplementary:
                rows.append(
                    _metric_row(
                        mode,
                        geometry,
                        "supplementary",
                        None,
                        "補足除外集計",
                        supplementary,
                        thresholds,
                        report_thresholds,
                    )
                )
    return rows


def _mean(values: Any) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return float(np.mean(numbers)) if numbers else None


def _metric_row(
    mode: str,
    geometry: str,
    scope: str,
    category_id: int | None,
    category_name: str,
    items: list[dict[str, Any]],
    thresholds: list[float],
    report_thresholds: list[float],
) -> dict[str, Any]:
    ap_means = {threshold: _mean(item["ap"][threshold] for item in items) for threshold in thresholds}
    row: dict[str, Any] = {
        "mode": mode,
        "evaluation_condition": "bboxは正解bboxを入力する条件付き評価" if mode == "bbox" else "カテゴリプロンプトのみ",
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
        pr_items = [item["pr"][threshold] for item in items]
        for metric in ("tp", "fp", "fn"):
            row[f"{metric}_iou_{key}"] = _mean(item["counts"][metric] for item in pr_items)
        for metric in ("precision", "recall", "f1"):
            row[f"{metric}_iou_{key}"] = _mean(item[metric] for item in pr_items)
    if category_id is not None:
        item = items[0]
        row["ap50_95"] = _mean(item["ap"].values())
        row["ap50"] = item["ap"].get(0.5)
        row["ap75"] = item["ap"].get(0.75)
    if not items:
        for threshold in report_thresholds:
            for metric in ("tp", "fp", "fn", "precision", "recall", "f1"):
                row[f"{metric}_iou_{threshold:.2f}"] = 0.0
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _gt_size(area: float) -> str:
    if area < 32 * 32:
        return "small"
    if area <= 96 * 96:
        return "medium"
    return "large"


def _build_per_gt_rows(
    mode: str,
    ground_truth_by_image: dict[int, list[dict[str, Any]]],
    predictions_by_mode: dict[int, list[dict[str, Any]]],
    category_names: dict[int, str],
    label_ja: dict[str, str],
    comparison_threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for image_id, ground_truths in ground_truth_by_image.items():
        image_gt_count = len(ground_truths)
        predictions = predictions_by_mode.get(image_id, [])
        for gt in ground_truths:
            same_category = [
                prediction for prediction in predictions
                if int(prediction["category_id"]) == int(gt["category_id"])
            ]
            bbox_ious = [
                intersection_over_union(gt, prediction, "bbox")
                for prediction in same_category
            ]
            mask_ious = [
                intersection_over_union(gt, prediction, "mask")
                for prediction in same_category
            ]
            bbox_iou = max(bbox_ious, default=0.0)
            mask_iou = max(mask_ious, default=0.0)
            state = (
                "見逃し" if not same_category
                else "成功" if mask_iou >= comparison_threshold
                else "低IoU"
            )
            source_iou = None
            if mode == "bbox":
                gt_box = {"xyxy": gt["xyxy"]}
                source_ious = []
                for prediction in same_category:
                    source_box = prediction.get("source_xyxy")
                    if source_box is not None:
                        source_ious.append(
                            intersection_over_union(
                                gt_box,
                                {"xyxy": source_box},
                                "bbox",
                            )
                        )
                if source_ious:
                    source_iou = max(source_ious)
            category_name = category_names[int(gt["category_id"])]
            x1, y1, x2, y2 = gt["xyxy"]
            area = max(0.0, float(x2 - x1) * float(y2 - y1))
            rows.append({
                "annotation_id": int(gt["annotation_id"]),
                "image_id": int(image_id),
                "mode": mode,
                "category_id": int(gt["category_id"]),
                "category_name": label_ja.get(category_name, category_name),
                "size": _gt_size(area),
                "density": "dense" if image_gt_count >= 5 else "sparse",
                "bbox_iou": bbox_iou,
                "mask_iou": mask_iou,
                "source_bbox_iou": source_iou,
                "state": state,
                "success_at_0_50": int(mask_iou >= 0.50),
                "success_at_0_75": int(mask_iou >= 0.75),
            })
    return rows


def _comparison_rows(
    a_rows: list[dict[str, Any]],
    b_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    equal_margin: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    a_gt = {int(row["annotation_id"]): row for row in a_rows}
    b_gt = {int(row["annotation_id"]): row for row in b_rows}
    comparison_gt = []
    for annotation_id in sorted(set(a_gt) | set(b_gt)):
        a = a_gt.get(annotation_id)
        b = b_gt.get(annotation_id)
        base = a or b
        comparison_gt.append({
            "annotation_id": annotation_id,
            "category_name": base["category_name"],
            "size": base["size"],
            "density": base["density"],
            "A_mask_iou": None if a is None else a["mask_iou"],
            "A_bbox_iou": None if a is None else a["bbox_iou"],
            "A_state": None if a is None else a["state"],
            "B_mask_iou": None if b is None else b["mask_iou"],
            "B_bbox_iou": None if b is None else b["bbox_iou"],
            "B_state": None if b is None else b["state"],
        })

    transition = []
    states = ("成功", "低IoU", "見逃し")
    total = len(comparison_gt)
    for a_state in states:
        for b_state in states:
            count = sum(
                row["A_state"] == a_state and row["B_state"] == b_state
                for row in comparison_gt
            )
            transition.append({
                "A_state": a_state,
                "B_state": b_state,
                "count": count,
                "ratio": count / total if total else 0.0,
            })

    comparison_metrics = []
    a_metrics = {
        (row["geometry"], row["scope"]): row
        for row in metric_rows if row["mode"] == "no_bbox"
    }
    b_metrics = {
        (row["geometry"], row["scope"]): row
        for row in metric_rows if row["mode"] == "bbox"
    }
    for geometry in ("bbox", "mask"):
        for scope in ("overall", "micro", "supplementary"):
            a = a_metrics.get((geometry, scope))
            b = b_metrics.get((geometry, scope))
            if not a and not b:
                continue
            keys = sorted(set(a or {}) | set(b or {}))
            for key in keys:
                if key not in {"ap50_95", "ap50", "ap75"} and not (
                    key.startswith(("precision_", "recall_", "f1_", "tp_", "fp_", "fn_"))
                ):
                    continue
                a_value = (a or {}).get(key)
                b_value = (b or {}).get(key)
                if a_value is None or b_value is None:
                    continue
                diff = float(b_value) - float(a_value)
                if abs(diff) < equal_margin:
                    decision = "同等"
                elif diff > 0:
                    decision = "bbox優位"
                else:
                    decision = "bboxなし優位"
                comparison_metrics.append({
                    "geometry": geometry,
                    "scope": scope,
                    "metric": key,
                    "A_no_bbox": float(a_value),
                    "B_bbox": float(b_value),
                    "difference": diff,
                    "decision": decision,
                })
    return comparison_gt, transition, comparison_metrics


def _clean_analysis_directory(analysis_directory: Path) -> None:
    if analysis_directory.is_symlink():
        raise RuntimeError(f"解析出力先がシンボリックリンクのため安全に更新できません: {analysis_directory}")
    if analysis_directory.exists():
        shutil.rmtree(analysis_directory)
    analysis_directory.mkdir(parents=True)


def run_analysis() -> None:
    settings = load_settings()
    run_id = settings.get("analysis_run_id")
    if not isinstance(run_id, str) or re.fullmatch(r"run_\d{5}", run_id) is None:
        raise ValueError("analysis_run_idはrun_00001形式で指定してください。")
    output = settings["output"]
    results_value = output.get("results_dir", "results")
    if not isinstance(results_value, str):
        raise ValueError("output.results_dirは文字列で指定してください。")
    run_directory = resolve_project_path(results_value) / run_id
    data = _read_database(run_directory / "inference.sqlite3")
    if data["run_info"]["run_id"] != run_id:
        raise RuntimeError("指定したrunと推論DB内のrun_idが一致しません。")
    snapshot = data["settings"]
    evaluation = snapshot.get("evaluation")
    if not isinstance(evaluation, dict):
        raise RuntimeError("推論DBの設定スナップショットに評価条件がありません。")
    thresholds = evaluation.get("iou_thresholds")
    report_thresholds = evaluation.get("report_thresholds")
    pr_confidence = evaluation.get("pr_confidence_threshold", 0.5)
    if not isinstance(thresholds, list) or not isinstance(report_thresholds, list):
        raise RuntimeError("推論DBの評価しきい値が不正です。")
    try:
        thresholds = [float(value) for value in thresholds]
        report_thresholds = [float(value) for value in report_thresholds]
        pr_confidence = float(pr_confidence)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("推論DBの評価しきい値が不正です。") from exc
    if (
        not thresholds
        or not report_thresholds
        or any(not np.isfinite(value) or not 0 <= value <= 1 for value in thresholds + report_thresholds)
        or any(value not in thresholds for value in report_thresholds)
        or not np.isfinite(pr_confidence)
        or not 0 <= pr_confidence <= 1
    ):
        raise RuntimeError("推論DBの評価しきい値が不正です。")
    output_settings = settings["output"]
    label_ja = output_settings.get("label_ja", {})
    if not isinstance(label_ja, dict):
        raise ValueError("output.label_jaはカテゴリ名と日本語ラベルの対応にしてください。")
    excluded_categories = evaluation.get("supplementary_exclude_categories", ["other ingredients"])
    if not isinstance(excluded_categories, list) or not all(isinstance(item, str) for item in excluded_categories):
        raise ValueError("evaluation.supplementary_exclude_categoriesは文字列の配列にしてください。")
    comparison_settings = snapshot.get("comparison", {})
    if not isinstance(comparison_settings, dict):
        raise RuntimeError("推論DBのcomparison設定が不正です。")
    comparison_threshold = float(comparison_settings.get("comparison_iou_threshold", 0.5))
    equal_margin = float(comparison_settings.get("equal_margin", 0.005))
    if not 0 <= comparison_threshold <= 1 or not 0 <= equal_margin <= 1:
        raise RuntimeError("推論DBのcomparisonしきい値が不正です。")
    analysis_directory = run_directory / "analysis"
    _clean_analysis_directory(analysis_directory)
    rows = _build_metric_rows(
        data["modes"],
        data["categories"],
        data["ground_truth_by_image"],
        data["predictions_by_mode"],
        thresholds,
        report_thresholds,
        float(pr_confidence),
        label_ja,
        excluded_categories,
    )
    for mode in data["modes"]:
        mode_directory = analysis_directory / mode
        mode_directory.mkdir(parents=True, exist_ok=True)
        write_metrics_csv(mode_directory / "metrics.csv", [row for row in rows if row["mode"] == mode])
    font_path = write_summary_images(
        analysis_directory,
        rows,
        output_settings,
        data["categories"],
        PROJECT_ROOT,
        report_thresholds,
        thresholds,
        float(pr_confidence),
    )
    category_names = {int(item["id"]): str(item["name"]) for item in data["categories"]}
    per_gt_by_mode: dict[str, list[dict[str, Any]]] = {}
    for mode in data["modes"]:
        mode_directory = analysis_directory / mode
        mode_rows = [row for row in rows if row["mode"] == mode]
        _write_csv(
            mode_directory / "metrics_by_category.csv",
            [row for row in mode_rows if row["scope"] == "category"],
        )
        per_gt = _build_per_gt_rows(
            mode,
            data["ground_truth_by_image"],
            data["predictions_by_mode"][mode],
            category_names,
            label_ja,
            comparison_threshold,
        )
        per_gt_by_mode[mode] = per_gt
        _write_csv(mode_directory / "per_gt.csv", per_gt)
        size_rows = []
        for size in ("small", "medium", "large"):
            for density in ("sparse", "dense"):
                items = [row for row in per_gt if row["size"] == size and row["density"] == density]
                if items:
                    size_rows.append({
                        "size": size,
                        "density": density,
                        "gt_count": len(items),
                        "bbox_mean_iou": float(np.mean([row["bbox_iou"] for row in items])),
                        "mask_mean_iou": float(np.mean([row["mask_iou"] for row in items])),
                        "success_at_0_50": float(np.mean([row["success_at_0_50"] for row in items])),
                        "success_at_0_75": float(np.mean([row["success_at_0_75"] for row in items])),
                    })
        _write_csv(mode_directory / "metrics_by_size.csv", size_rows)

    if "no_bbox" in per_gt_by_mode and "bbox" in per_gt_by_mode:
        comparison_gt, transition, comparison_metrics = _comparison_rows(
            per_gt_by_mode["no_bbox"],
            per_gt_by_mode["bbox"],
            rows,
            equal_margin,
        )
        comparison_directory = analysis_directory / "comparison"
        write_comparison_outputs(
            comparison_directory,
            comparison_metrics,
            comparison_gt,
            transition,
            font_path,
            int(output_settings.get("dpi", 150)),
        )
    image_dir = resolve_project_path(str(data["run_info"]["image_dir"]))
    images = [{"id": int(item["image_id"]), "file_name": str(item["image_path"])} for item in data["images"]]
    for mode_index, mode in enumerate(data["modes"]):
        sample_total = int(output_settings.get("visualization_samples", 20))
        sample_count = sample_total // len(data["modes"]) + int(mode_index < sample_total % len(data["modes"]))
        worst_best = min(int(output_settings.get("worst_best_samples", 10)), sample_count // 2)
        selected = (
            select_visualization_images(
                images,
                data["ground_truth_by_image"],
                data["predictions_by_mode"][mode],
                sample_count,
                worst_best,
                float(pr_confidence),
            )
            if sample_count
            else []
        )
        available = []
        for image in selected:
            if (image_dir / str(image["file_name"])).is_file():
                available.append(image)
            else:
                LOGGER.warning("元画像がないためoverlayを生成できません: %s", image["file_name"])
        count = write_overlays(
            analysis_directory,
            mode,
            available,
            image_dir,
            data["ground_truth_by_image"],
            data["predictions_by_mode"][mode],
            category_names,
            label_ja,
            font_path,
            int(output_settings.get("dpi", 150)),
            float(pr_confidence),
        )
        LOGGER.info("%s: overlay %d件を生成しました", mode, count)
    LOGGER.info("解析成果物を生成しました: %s", analysis_directory)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_analysis()


if __name__ == "__main__":
    main()
