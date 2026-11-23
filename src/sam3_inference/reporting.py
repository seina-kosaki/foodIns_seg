"""評価結果のCSV、表画像、overlay画像を出力する。"""

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import to_hex, to_rgb
from matplotlib.ft2font import FT2Font
from PIL import Image, ImageDraw, ImageFont

from src.sam3_inference.evaluation import decode_mask, match_image

_CSV_COLUMN_LABELS = {
    "mode": "推論条件",
    "evaluation_condition": "評価条件",
    "geometry": "評価対象",
    "scope": "集計区分",
    "category_id": "カテゴリID",
    "category_name": "カテゴリ名",
    "gt_count": "正解数",
    "prediction_count": "予測数",
    "ap50_95": "AP50-95",
    "ap50": "AP50",
    "ap75": "AP75",
}
_METRIC_LABELS = {
    "tp": "TP",
    "fp": "FP",
    "fn": "FN",
    "precision": "適合率",
    "recall": "再現率",
    "f1": "F1値",
    "ap": "AP",
}
_MODE_LABELS = {"no_bbox": "bboxなし", "bbox": "GT bboxあり"}
_GEOMETRY_LABELS = {"bbox": "bbox", "mask": "マスク"}
_SCOPE_LABELS = {"overall": "全体", "category": "カテゴリ別"}


def _configure_japanese_font(font_path: Path) -> font_manager.FontProperties:
    font_manager.fontManager.addfont(str(font_path))
    font = font_manager.FontProperties(fname=str(font_path))
    plt.rcParams["font.family"] = font.get_name()
    plt.rcParams["font.sans-serif"] = [font.get_name()]
    plt.rcParams["axes.unicode_minus"] = False
    return font


def _apply_font_to_axis(axis: Any, font: font_manager.FontProperties) -> None:
    axis.title.set_fontproperties(font)
    axis.xaxis.label.set_fontproperties(font)
    axis.yaxis.label.set_fontproperties(font)
    for tick in axis.get_xticklabels() + axis.get_yticklabels():
        tick.set_fontproperties(font)


def _find_japanese_font(output_settings: dict[str, Any], category_names: list[str], project_root: Path) -> Path:
    required = "".join(category_names) + (
        "食事データ評価bboxは正解bboxを入力する条件付き評価"
        "推論条件評価対象カテゴリ別全体カテゴリ正解数GT数予測数適合率再現率F1値"
        "TPFPFN緑赤青マスクAPbboxAPAP50-95AP50AP75比較指標"
        "PrecisionRecallP/R/F1@.0123456789–"
        "推論条件別AP比較bboxありはGT bboxを入力した条件付き評価ですカテゴリ別指標"
        "生画像予測正解塗りつぶし輪郭実線破線"
        "信頼度しきい値"
    )
    configured = output_settings.get("font_path")
    candidates = []
    if configured is not None:
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError("output.font_pathはフォントファイルのパスまたはnullにしてください。")
        path = Path(configured)
        candidates.append(path if path.is_absolute() else project_root / path)
    else:
        known_japanese_fonts = (
            "NotoSansJP-VF.ttf",
            "NotoSerifJP-VF.ttf",
            "YuGothR.ttc",
            "YuGothM.ttc",
            "meiryo.ttc",
            "msgothic.ttc",
            "msyh.ttc",
        )
        windows_fonts = Path("C:/Windows/Fonts")
        candidates.extend(windows_fonts / name for name in known_japanese_fonts)
        candidates.extend(Path(path) for path in font_manager.findSystemFonts())
        fonts_directory = project_root / "assets" / "fonts"
        if fonts_directory.is_dir():
            candidates.extend(fonts_directory.glob("*.ttf"))
            candidates.extend(fonts_directory.glob("*.otf"))
            candidates.extend(fonts_directory.glob("*.ttc"))

    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            charmap = FT2Font(str(candidate)).get_charmap()
            if all(ord(character) in charmap or character.isspace() for character in required):
                return candidate
        except (OSError, RuntimeError, ValueError):
            continue
    raise RuntimeError("日本語を表示できるフォントが見つかりません。output.font_pathに日本語フォントを指定してください。")


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [_csv_column_label(key) for key in rows[0]] if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    _csv_column_label(key): _csv_value(key, value)
                    for key, value in row.items()
                }
            )


def write_comparison_outputs(
    comparison_directory: Path,
    comparison_metrics: list[dict[str, Any]],
    comparison_gt: list[dict[str, Any]],
    transition: list[dict[str, Any]],
    font_path: Path,
    dpi: int,
) -> None:
    comparison_directory.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("comparison_metrics.csv", comparison_metrics),
        ("comparison_gt.csv", comparison_gt),
        ("transition_matrix.csv", transition),
    ):
        if not rows:
            continue
        with (comparison_directory / name).open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    font = _configure_japanese_font(font_path)
    charts_directory = comparison_directory / "charts"
    charts_directory.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    for axis, geometry in zip(axes, ("bbox", "mask")):
        rows = [row for row in comparison_gt if row.get(f"{'B' if geometry == 'bbox' else 'B'}_{geometry}_iou") is not None]
        x = [float(row[f"A_{geometry}_iou"]) for row in rows if row.get(f"A_{geometry}_iou") is not None]
        y = [float(row[f"B_{geometry}_iou"]) for row in rows if row.get(f"B_{geometry}_iou") is not None]
        axis.scatter(x, y, s=18, alpha=0.65, color="#2878B5")
        axis.plot([0, 1], [0, 1], "--", color="#777777")
        axis.set_title(f"{geometry} GT別IoU", fontproperties=font)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("bboxなし 最良IoU", fontproperties=font)
        axis.set_ylabel("bboxあり 最良IoU", fontproperties=font)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(charts_directory / "scatter_ab_iou.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    matrix = np.full((3, 3), 0.0)
    states = ("成功", "低IoU", "見逃し")
    total = len(comparison_gt)
    for row in transition:
        i = states.index(row["A_state"])
        j = states.index(row["B_state"])
        matrix[i, j] = float(row["count"])
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    axis.set_xticks(range(3), states)
    axis.set_yticks(range(3), states)
    axis.set_xlabel("bboxあり", fontproperties=font)
    axis.set_ylabel("bboxなし", fontproperties=font)
    axis.set_title("状態遷移（件数 / 割合）", fontproperties=font)
    _apply_font_to_axis(axis, font)
    for i in range(3):
        for j in range(3):
            count = int(matrix[i, j])
            ratio = count / total if total else 0.0
            axis.text(j, i, f"{count}\n{ratio:.1%}", ha="center", va="center",
                      fontproperties=font)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(charts_directory / "transition_matrix.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    rescued = sum(
        row["A_state"] in ("低IoU", "見逃し") and row["B_state"] == "成功"
        for row in comparison_gt
    )
    rescued_ratio = rescued / len(comparison_gt) if comparison_gt else 0.0
    figure = plt.figure(figsize=(15, 10))
    axis = figure.add_subplot(111)
    axis.axis("off")
    axis.text(
        0.01,
        0.96,
        f"bboxにより救われるGT: {rescued}件（{rescued_ratio:.1%}）",
        fontproperties=font,
        fontsize=16,
        va="top",
    )
    table_rows = [
        row for row in comparison_metrics
        if row["metric"] in ("ap50_95", "ap50", "ap75",
                             "recall_iou_0.50", "f1_iou_0.50")
    ]
    if table_rows:
        table = axis.table(
            cellText=[
                [row["geometry"], row["metric"], f"{row['A_no_bbox']:.3f}",
                 f"{row['B_bbox']:.3f}", f"{row['difference']:+.3f}", row["decision"]]
                for row in table_rows[:12]
            ],
            colLabels=["評価対象", "指標", "A: bboxなし", "B: bboxあり", "差(B-A)", "判定"],
            loc="upper left",
            bbox=(0.01, 0.08, 0.98, 0.78),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for key, cell in table.get_celld().items():
            cell.get_text().set_fontproperties(font)
            cell.get_text().set_fontsize(9)
    figure.savefig(comparison_directory / "summary.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _csv_column_label(key: str) -> str:
    if "_iou_" in key and key.startswith("ap_"):
        _, threshold = key.split("_iou_", maxsplit=1)
        return f"AP@IoU {threshold}"
    if "_iou_" in key:
        metric, threshold = key.split("_iou_", maxsplit=1)
        if metric in _METRIC_LABELS:
            return f"{_METRIC_LABELS[metric]}@IoU {threshold}"
    return _CSV_COLUMN_LABELS.get(key, key)


def _csv_value(key: str, value: Any) -> Any:
    if key == "mode":
        return _MODE_LABELS.get(value, value)
    if key == "geometry":
        return _GEOMETRY_LABELS.get(value, value)
    if key == "scope":
        return _SCOPE_LABELS.get(value, value)
    return value


def write_summary_images(
    run_directory: Path,
    rows: list[dict[str, Any]],
    output_settings: dict[str, Any],
    categories: list[dict[str, Any]],
    project_root: Path,
    report_thresholds: list[float],
    iou_thresholds: list[float] | None = None,
    pr_confidence_threshold: float = 0.5,
) -> Path:
    label_ja = output_settings.get("label_ja", {})
    category_names = [
        str(label_ja.get(category["name"], category["name"])) if isinstance(label_ja, dict) else str(category["name"])
        for category in categories
    ]
    font_path = _find_japanese_font(output_settings, category_names, project_root)
    dpi = output_settings.get("dpi", 150)
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("output.dpiは正の整数にしてください。")
    mode_names = list(dict.fromkeys(row["mode"] for row in rows))
    if iou_thresholds is None:
        iou_thresholds = sorted(
            {
                float(key.removeprefix("ap_iou_"))
                for row in rows
                for key in row
                if key.startswith("ap_iou_")
            }
        )
    _write_comparison_summary(
        run_directory,
        rows,
        mode_names,
        font_path,
        dpi,
        iou_thresholds,
        report_thresholds,
        pr_confidence_threshold,
    )

    translated_names = {
        int(category["id"]): (
            str(label_ja.get(category["name"], category["name"]))
            if isinstance(label_ja, dict)
            else str(category["name"])
        )
        for category in categories
    }
    for mode in mode_names:
        mode_rows = [
            row for row in rows
            if row["mode"] == mode and row["scope"] == "category"
        ]
        category_ids = sorted({int(row["category_id"]) for row in mode_rows})
        mode_directory = run_directory / mode
        mode_directory.mkdir(parents=True, exist_ok=True)
        for page_index, start in enumerate(range(0, len(category_ids), 20), start=1):
            page_ids = category_ids[start : start + 20]
            for geometry in ("mask", "bbox"):
                _write_category_metrics_chart(
                    mode_directory / f"category_ap_{geometry}_{page_index:03d}.png",
                    mode,
                    page_ids,
                    mode_rows,
                    translated_names,
                    report_thresholds,
                    font_path,
                    dpi,
                    geometry=geometry,
                )
    return font_path


def _write_comparison_summary(
    run_directory: Path,
    rows: list[dict[str, Any]],
    modes: list[str],
    font_path: Path,
    dpi: int,
    iou_thresholds: list[float],
    report_thresholds: list[float],
    pr_confidence_threshold: float,
) -> None:
    font = _configure_japanese_font(font_path)
    mode_colors = {"no_bbox": "#2878B5", "bbox": "#E07A24"}
    mode_styles = {"no_bbox": "-", "bbox": "--"}
    metric_styles = {
        "precision": ("#2878B5", "o", "適合率"),
        "recall": ("#48A36D", "s", "再現率"),
        "f1": ("#E07A24", "^", "F1"),
    }
    figure, axes = plt.subplots(2, 2, figsize=(15, 11), squeeze=False)
    for axis, geometry, title in (
        (axes[0][0], "mask", "マスク mAP（IoUごと）"),
        (axes[0][1], "bbox", "bbox mAP（IoUごと）"),
    ):
        overall_by_mode = {
            row["mode"]: row
            for row in rows
            if row["scope"] == "overall" and row["geometry"] == geometry
        }
        x_values = np.asarray(iou_thresholds, dtype=float)
        for mode in modes:
            summary = overall_by_mode.get(mode, {})
            values = [summary.get(f"ap_iou_{threshold:.2f}") for threshold in iou_thresholds]
            numeric_values = [np.nan if value is None else float(value) for value in values]
            axis.plot(
                x_values,
                numeric_values,
                marker="o",
                linestyle=mode_styles.get(mode, "-"),
                label=_MODE_LABELS.get(mode, mode),
                color=mode_colors.get(mode, "#777777"),
            )
        axis.set_title(title, fontproperties=font, fontsize=14)
        axis.set_xticks(x_values, [f"{value:.2f}" for value in iou_thresholds])
        axis.set_xlabel("IoU閾値", fontproperties=font)
        axis.set_ylabel("カテゴリ平均 mAP", fontproperties=font)
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(prop=font)
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            tick.set_fontproperties(font)

    for axis, geometry, title in (
        (axes[1][0], "mask", "マスク P/R/F1（カテゴリ平均）"),
        (axes[1][1], "bbox", "bbox P/R/F1（カテゴリ平均）"),
    ):
        overall_by_mode = {
            row["mode"]: row
            for row in rows
            if row["scope"] == "overall" and row["geometry"] == geometry
        }
        x_values = np.asarray(report_thresholds, dtype=float)
        for mode in modes:
            summary = overall_by_mode.get(mode, {})
            for metric, (color, marker, metric_label) in metric_styles.items():
                values = [
                    summary.get(f"{metric}_iou_{threshold:.2f}")
                    for threshold in report_thresholds
                ]
                numeric_values = [np.nan if value is None else float(value) for value in values]
                axis.plot(
                    x_values,
                    numeric_values,
                    color=color,
                    marker=marker,
                    linestyle=mode_styles.get(mode, "-"),
                    label=f"{_MODE_LABELS.get(mode, mode)}・{metric_label}",
                )
        axis.set_title(title, fontproperties=font, fontsize=14)
        axis.set_xticks(x_values, [f"{value:.2f}" for value in report_thresholds])
        axis.set_xlabel("IoU閾値", fontproperties=font)
        axis.set_ylabel("指標値", fontproperties=font)
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(prop=font, fontsize=8, ncol=2)
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            tick.set_fontproperties(font)

    figure.suptitle(
        f"推論条件比較（P/R/F1の信頼度しきい値 pr_confidence_threshold: {pr_confidence_threshold}）",
        fontproperties=font,
        fontsize=15,
    )
    figure.text(
        0.5,
        0.015,
        "bboxありはGT bboxを入力した条件付き評価です。",
        ha="center",
        fontproperties=font,
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(run_directory / "comparison_summary.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _write_category_metrics_chart(
    path: Path,
    mode: str,
    category_ids: list[int],
    rows: list[dict[str, Any]],
    category_names: dict[int, str],
    report_thresholds: list[float],
    font_path: Path,
    dpi: int,
    *,
    geometry: str,
) -> None:
    font = _configure_japanese_font(font_path)
    geometry_rows = {
        int(row["category_id"]): row
        for row in rows
        if row["geometry"] == geometry and row["category_id"] is not None
    }
    names = []
    for category_id in category_ids:
        name = category_names.get(category_id, str(category_id))
        row = geometry_rows.get(category_id)
        gt_count = row.get("gt_count") if row else None
        names.append(f"{name} (GT {int(gt_count)})" if gt_count is not None else name)
    figure_height = max(8, 0.5 * len(category_ids) + 3)
    figure, axis = plt.subplots(figsize=(8, figure_height))
    category_rows = [geometry_rows.get(category_id, {}) for category_id in category_ids]
    _plot_grouped_metrics(
        axis,
        category_rows,
        names,
        ("ap50_95", "ap50", "ap75"),
        ("AP50-95", "AP50", "AP75"),
        ("#2878B5", "#48A36D", "#E07A24"),
        font,
    )
    title = "マスク" if geometry == "mask" else "bbox"
    axis.set_title(f"{title} mAP", fontproperties=font)
    figure.suptitle(
        f"{_MODE_LABELS.get(mode, mode)} {title}カテゴリ別指標（{category_ids[0]}–{category_ids[-1]}）",
        fontproperties=font,
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    heatmap_path = Path(
        str(path)
        .replace("category_ap_", "category_heatmap_", 1)
        .replace("category_metrics_", "category_heatmap_", 1)
    )
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(max(12, len(report_thresholds) * 3), figure_height))
    _plot_category_heatmap(
        axis,
        category_rows,
        names,
        report_thresholds,
        font,
    )
    axis.set_title(f"{title} P/R/F1ヒートマップ", fontproperties=font)
    figure.suptitle(
        f"{_MODE_LABELS.get(mode, mode)} {title}カテゴリ別P/R/F1（{category_ids[0]}–{category_ids[-1]}）",
        fontproperties=font,
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(heatmap_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_category_heatmap(
    axis: Any,
    rows: list[dict[str, Any]],
    names: list[str],
    report_thresholds: list[float],
    font: font_manager.FontProperties,
) -> None:
    columns = []
    labels = []
    metric_labels = (
        ("precision", "適合率"),
        ("recall", "再現率"),
        ("f1", "F1"),
    )
    for metric, metric_label in metric_labels:
        for threshold in report_thresholds:
            columns.append(f"{metric}_iou_{threshold:.2f}")
            labels.append(f"{metric_label}@{threshold:.2f}")

    matrix = np.asarray([
        [
            np.nan if row.get(column) is None else float(row[column])
            for column in columns
        ]
        for row in rows
    ], dtype=float)
    image = axis.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.invert_yaxis()
    axis.set_title("P/R/F1ヒートマップ", fontproperties=font)
    for tick in axis.get_xticklabels() + axis.get_yticklabels():
        tick.set_fontproperties(font)
        tick.set_fontsize(8)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontproperties=font,
                    color="white" if value >= 0.75 else "black",
                )
    axis.set_xlabel("指標とIoU閾値", fontproperties=font)
    axis.figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)


def _plot_grouped_metrics(
    axis: Any,
    rows: list[dict[str, Any]],
    names: list[str],
    columns: tuple[str, ...],
    labels: tuple[str, ...],
    colors: tuple[str, ...],
    font: font_manager.FontProperties,
) -> None:
    positions = np.arange(len(names), dtype=float)
    series_count = max(len(columns), 1)
    bar_height = min(0.8 / series_count, 0.22)
    for index, (column, label) in enumerate(zip(columns, labels)):
        values = [
            np.nan if row.get(column) is None else float(row[column])
            for row in rows
        ]
        offset = (index - (series_count - 1) / 2) * bar_height
        color = colors[index % len(colors)]
        axis.barh(
            positions + offset,
            values,
            height=bar_height,
            label=label,
            color=color,
        )
        zero_positions = positions[np.asarray(values) == 0] + offset
        if zero_positions.size:
            axis.scatter(
                np.full(zero_positions.shape, 0.008),
                zero_positions,
                marker="|",
                s=60,
                linewidths=1.5,
                color=color,
                zorder=3,
                clip_on=False,
            )
    axis.set_yticks(positions, names)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.grid(axis="x", alpha=0.2)
    legend = axis.legend(loc="lower right", prop=font, ncol=2)
    for text in legend.get_texts():
        text.set_fontsize(9)
    for tick in axis.get_xticklabels():
        tick.set_fontproperties(font)
        tick.set_fontsize(10)
    for tick in axis.get_yticklabels():
        tick.set_fontproperties(font)
        tick.set_fontsize(11)


def _blend_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    overlay = Image.new("RGB", image.size, color)
    current = np.asarray(image).copy()
    overlay_array = np.asarray(overlay)
    current[mask] = (current[mask].astype(float) * 0.55 + overlay_array[mask].astype(float) * 0.45).astype(np.uint8)
    image.paste(Image.fromarray(current))


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
) -> None:
    distance = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    if distance == 0:
        return
    for offset in range(0, int(distance) + 1, 10):
        end_offset = min(offset + 5, distance)
        first = (
            round(start[0] + (end[0] - start[0]) * offset / distance),
            round(start[1] + (end[1] - start[1]) * offset / distance),
        )
        second = (
            round(start[0] + (end[0] - start[0]) * end_offset / distance),
            round(start[1] + (end[1] - start[1]) * end_offset / distance),
        )
        draw.line((first, second), fill=color, width=width)


def _draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = bounds
    for start, end in (
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ):
        _draw_dashed_line(draw, start, end, color, 3)


def write_overlays(
    run_directory: Path,
    mode: str,
    selected_images: list[dict[str, Any]],
    image_dir: Path,
    ground_truth_by_image: dict[int, list[dict[str, Any]]],
    predictions_by_image: dict[int, list[dict[str, Any]]],
    category_names: dict[int, str],
    label_ja: dict[str, str],
    font_path: Path,
    dpi: int,
    confidence_threshold: float,
) -> int:
    colors = {"TP": (0, 180, 0), "FP": (230, 40, 40), "FN": (30, 90, 240)}
    font = ImageFont.truetype(str(font_path), 16)
    legend_font = ImageFont.truetype(str(font_path), 12)
    output_directory = run_directory / mode / "overlays"
    output_directory.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, image_info in enumerate(selected_images, start=1):
        image_id = int(image_info["id"])
        with Image.open(image_dir / str(image_info["file_name"])) as source:
            original = source.convert("RGB")
        bbox_panel = original.copy()
        mask_panel = original.copy()
        gt = ground_truth_by_image.get(image_id, [])
        predictions = predictions_by_image.get(image_id, [])
        bbox_matches = match_image(gt, predictions, 0.5, "bbox", confidence_threshold)
        mask_matches = match_image(gt, predictions, 0.5, "mask", confidence_threshold)
        bbox_prediction_status = {
            item["prediction_index"]: item["status"] for item in bbox_matches["matches"]
        }
        mask_prediction_status = {
            item["prediction_index"]: item["status"] for item in mask_matches["matches"]
        }
        bbox_gt_status = {
            item["ground_truth_index"]
            for item in bbox_matches["matches"]
            if item["status"] == "TP"
        }
        mask_gt_status = {
            item["ground_truth_index"]: item["status"]
            for item in mask_matches["matches"]
            if item["status"] == "TP"
        }

        for gt_index, item in enumerate(gt):
            bbox_status = "TP" if gt_index in bbox_gt_status else "FN"
            mask_status = "TP" if gt_index in mask_gt_status else "FN"
            x1, y1, x2, y2 = (int(round(value)) for value in item["xyxy"])
            category_name = category_names[int(item["category_id"])]
            label = label_ja.get(category_name, category_name)
            bbox_draw = ImageDraw.Draw(bbox_panel)
            bbox_draw.rectangle((x1, y1, x2, y2), outline=colors[bbox_status], width=3)
            bbox_draw.text(
                (x1, max(0, y1 - 20)),
                f"{bbox_status} 正解 {label}",
                fill=colors[bbox_status],
                font=font,
                stroke_width=1,
                stroke_fill=(255, 255, 255),
            )
            _blend_mask(mask_panel, decode_mask(item["mask"]), colors[mask_status])
            mask_draw = ImageDraw.Draw(mask_panel)
            mask_draw.text(
                (x1, max(0, y1 - 20)),
                f"{mask_status} 正解 {label}",
                fill=colors[mask_status],
                font=font,
                stroke_width=1,
                stroke_fill=(255, 255, 255),
            )

        for prediction_index, item in enumerate(predictions):
            if float(item["confidence"]) < confidence_threshold:
                continue
            bbox_status = bbox_prediction_status.get(prediction_index, "FP")
            mask_status = mask_prediction_status.get(prediction_index, "FP")
            x1, y1, x2, y2 = (int(round(value)) for value in item["xyxy"])
            category_name = category_names[int(item["category_id"])]
            label = label_ja.get(category_name, category_name)
            bbox_draw = ImageDraw.Draw(bbox_panel)
            _draw_dashed_rectangle(bbox_draw, (x1, y1, x2, y2), colors[bbox_status])
            bbox_draw.text(
                (x1, min(bbox_panel.height - 20, y2)),
                f"{bbox_status} 予測 {label}",
                fill=colors[bbox_status],
                font=font,
                stroke_width=1,
                stroke_fill=(255, 255, 255),
            )

            mask_array = decode_mask(item["mask"]).astype(np.uint8)
            contours, _ = cv2.findContours(mask_array, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            mask_pixels = np.asarray(mask_panel).copy()
            cv2.drawContours(mask_pixels, contours, -1, colors[mask_status], 2)
            mask_panel = Image.fromarray(mask_pixels)
            mask_draw = ImageDraw.Draw(mask_panel)
            mask_draw.text(
                (x1, min(mask_panel.height - 20, y2)),
                f"{mask_status} 予測 {label}",
                fill=colors[mask_status],
                font=font,
                stroke_width=1,
                stroke_fill=(255, 255, 255),
            )

        panel_titles = ("(a) 生画像", "(b) bbox", "(c) mask")
        panel_height = original.height + 76
        composite = Image.new("RGB", (original.width * 3, panel_height), (255, 255, 255))
        composite_draw = ImageDraw.Draw(composite)
        for panel_index, (panel, title) in enumerate(
            zip((original, bbox_panel, mask_panel), panel_titles)
        ):
            x_offset = panel_index * original.width
            composite_draw.text((x_offset + 8, 5), title, fill=(0, 0, 0), font=font)
            composite.paste(panel, (x_offset, 28))
        legend_y = 32 + original.height
        composite_draw.text(
            (original.width + 8, legend_y),
            "TP: 緑 FP: 赤 FN: 青",
            fill=(0, 0, 0),
            font=legend_font,
        )
        composite_draw.text(
            (original.width + 8, legend_y + 18),
            "正解: 実線 / 予測: 破線",
            fill=(0, 0, 0),
            font=legend_font,
        )
        composite_draw.text(
            (original.width * 2 + 8, legend_y),
            "TP: 緑 FP: 赤 FN: 青",
            fill=(0, 0, 0),
            font=legend_font,
        )
        composite_draw.text(
            (original.width * 2 + 8, legend_y + 18),
            "正解: 塗りつぶし / 予測: 輪郭",
            fill=(0, 0, 0),
            font=legend_font,
        )
        sample_group = image_info.get("_sample_group", "sample")
        destination = output_directory / f"{sample_group}_{index:02d}_image_{image_id}.png"
        composite.save(destination, dpi=(dpi, dpi))
        written += 1
    return written


def select_visualization_images(
    selected_images: list[dict[str, Any]],
    ground_truth_by_image: dict[int, list[dict[str, Any]]],
    predictions_by_image: dict[int, list[dict[str, Any]]],
    sample_count: int,
    worst_best_count: int,
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    cases = []
    image_by_id = {int(image["id"]): image for image in selected_images}
    for image_id in image_by_id:
        gt = ground_truth_by_image.get(image_id, [])
        predictions = predictions_by_image.get(image_id, [])
        result = match_image(gt, predictions, 0.5, "mask", confidence_threshold)
        for item in result["matches"]:
            cases.append((0.0 if item["iou"] is None else float(item["iou"]), image_id))
        for _ in result["false_negative_ground_truth_indices"]:
            cases.append((0.0, image_id))
    cases.sort(key=lambda item: item[0])

    selected = []
    ranked_groups = (
        (
            ("worst", cases[:worst_best_count]),
            ("best", list(reversed(cases[-worst_best_count:]))),
        )
        if worst_best_count
        else ()
    )
    selected_image_ids: set[int] = set()
    for group, candidates in ranked_groups:
        for _, image_id in candidates:
            if len(selected) >= sample_count:
                break
            if image_id in selected_image_ids:
                continue
            selected.append({**image_by_id[image_id], "_sample_group": group})
            selected_image_ids.add(image_id)
    for image in selected_images:
        if len(selected) >= sample_count:
            break
        image_id = int(image["id"])
        if image_id in selected_image_ids:
            continue
        selected.append({**image, "_sample_group": "sample"})
        selected_image_ids.add(image_id)
    return selected[:sample_count]
