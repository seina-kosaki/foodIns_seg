"""SAM3評価指標の小規模な動作確認。"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.main import (
    _build_metric_rows,
    coco_bbox_to_clipped_xyxy,
    limit_predictions_per_category,
    prediction_to_record,
)
from src.sam3_inference.evaluation import (
    calculate_average_precision,
    calculate_prf1_by_image,
    match_image,
)
from src.sam3_inference import reporting
from src.sam3_inference.reporting import write_metrics_csv
from src.sam3_inference.reporting import write_overlays


def test_coco_bbox_to_clipped_xyxy_clips_rounding_error_and_image_edges():
    assert coco_bbox_to_clipped_xyxy(
        [2.0, -8.881784197001252e-16, 10.0, 10.0],
        image_width=10,
        image_height=10,
    ) == [2.0, 0.0, 10.0, 10.0]
    assert coco_bbox_to_clipped_xyxy(
        [-1.0, 1.0, 12.0, 12.0],
        image_width=10,
        image_height=10,
    ) == [0.0, 1.0, 10.0, 10.0]


def test_coco_bbox_to_clipped_xyxy_rejects_zero_area_after_clipping():
    import pytest

    with pytest.raises(ValueError, match="正の面積"):
        coco_bbox_to_clipped_xyxy(
            [12.0, 1.0, 2.0, 2.0],
            image_width=10,
            image_height=10,
        )


def _instance(category_id, xyxy, mask, confidence=1.0):
    return {
        "category_id": category_id,
        "xyxy": xyxy,
        "mask": np.asarray(mask, dtype=bool),
        "confidence": confidence,
    }


def test_mask_ap_uses_score_order_and_101_point_interpolation():
    mask = np.array([[1, 0], [0, 0]], dtype=bool)
    ground_truth = {1: [_instance(3, [0, 0, 1, 1], mask)]}
    predictions = {
        1: [
            _instance(3, [1, 1, 2, 2], np.array([[0, 0], [0, 1]]), 0.9),
            _instance(3, [0, 0, 1, 1], mask, 0.8),
        ]
    }

    ap = calculate_average_precision(ground_truth, predictions, 3, 0.5, "mask")

    assert ap == 0.5


def test_mask_and_bbox_matching_are_evaluated_independently():
    ground_truth = [_instance(1, [0, 0, 4, 4], np.ones((4, 4), dtype=bool))]
    prediction = _instance(1, [0, 0, 4, 4], np.eye(4, dtype=bool))

    assert match_image(ground_truth, [prediction], 0.5, "bbox")["tp"] == 1
    result = match_image(ground_truth, [prediction], 0.5, "mask")
    assert result["tp"] == 0
    assert result["fp"] == 1
    assert result["fn"] == 1


def test_prf1_matching_is_per_image_category_and_one_to_one():
    mask = np.array([[1, 0], [0, 0]], dtype=bool)
    ground_truth = {1: [_instance(2, [0, 0, 1, 1], mask)]}
    predictions = {
        1: [
            _instance(2, [0, 0, 1, 1], mask, 0.9),
            _instance(2, [0, 0, 1, 1], mask, 0.8),
            _instance(99, [0, 0, 1, 1], mask, 0.95),
        ],
        2: [_instance(2, [0, 0, 1, 1], mask, 0.7)],
    }

    result = calculate_prf1_by_image(ground_truth, predictions, 2, 0.5, "mask", 0.5)

    assert result["counts"] == {"tp": 1, "fp": 2, "fn": 0}
    assert result["precision"] == 1 / 3
    assert result["recall"] == 1.0
    assert result["f1"] == 0.5


def test_normal_macro_average_includes_every_gt_category():
    mask = np.array([[1, 0], [0, 0]], dtype=bool)
    ground_truth = {
        1: [
            _instance(1, [0, 0, 1, 1], mask),
            _instance(2, [0, 0, 1, 1], mask),
        ]
    }
    predictions = {
        1: [
            _instance(1, [1, 1, 2, 2], np.array([[0, 0], [0, 1]])),
            _instance(2, [0, 0, 1, 1], mask),
        ]
    }
    categories = [{"id": 1, "name": "other ingredients"}, {"id": 2, "name": "rice"}]

    rows = _build_metric_rows(
        ["no_bbox"],
        categories,
        ground_truth,
        {"no_bbox": predictions},
        [0.5],
        [0.5],
        0.5,
        {},
    )

    overall = next(row for row in rows if row["scope"] == "overall" and row["geometry"] == "mask")
    category_rows = [row for row in rows if row["scope"] == "category" and row["geometry"] == "mask"]
    assert overall["gt_count"] == 2
    assert overall["ap50"] == 0.5
    assert overall["ap_iou_0.50"] == 0.5
    assert {row["category_name"] for row in category_rows} == {"other ingredients", "rice"}
    assert not any(row["scope"] == "supplementary" for row in rows)


def test_metric_rows_keep_ap_for_each_iou_threshold():
    mask = np.ones((4, 4), dtype=bool)
    ground_truth = {1: [_instance(1, [0, 0, 4, 4], mask)]}
    predictions = {
        1: [_instance(1, [0, 0, 2, 4], mask, 0.9)]
    }

    rows = _build_metric_rows(
        ["no_bbox"],
        [{"id": 1, "name": "rice"}],
        ground_truth,
        {"no_bbox": predictions},
        [0.5, 0.75],
        [0.5],
        0.5,
        {},
    )

    bbox_overall = next(
        row for row in rows if row["scope"] == "overall" and row["geometry"] == "bbox"
    )
    assert bbox_overall["ap_iou_0.50"] == 1.0
    assert bbox_overall["ap_iou_0.75"] == 0.0


def test_persistent_prediction_record_omits_mask():
    prediction = {
        "category_id": 1,
        "category_name": "rice",
        "prompt": "rice",
        "confidence": 0.8,
        "xyxy": [1, 2, 3, 4],
        "source_xyxy": None,
        "mask": np.ones((2, 2), dtype=bool),
    }

    record = prediction_to_record(8, "no_bbox", prediction)

    assert "mask" not in record
    assert record["image_id"] == 8


def test_prediction_limit_is_applied_per_category():
    predictions = [
        {
            "category_id": 1,
            "confidence": confidence / 101,
        }
        for confidence in range(101)
    ]
    predictions.append({"category_id": 2, "confidence": 0.1})

    limited = limit_predictions_per_category(predictions)

    category_one = [item for item in limited if item["category_id"] == 1]
    category_two = [item for item in limited if item["category_id"] == 2]
    assert len(category_one) == 100
    assert min(item["confidence"] for item in category_one) == 1 / 101
    assert len(category_two) == 1


def test_metrics_csv_has_japanese_headers_and_values(tmp_path):
    path = tmp_path / "metrics.csv"
    write_metrics_csv(
        path,
        [
            {
                "mode": "no_bbox",
                "geometry": "mask",
                "scope": "overall",
                "category_name": "りんご",
                "ap50_95": 0.75,
                "ap_iou_0.50": 0.7,
                "precision_iou_0.50": 0.8,
            }
        ],
    )

    content = path.read_bytes()
    decoded = content.decode("utf-8-sig")

    assert content.startswith(b"\xef\xbb\xbf")
    assert "推論条件" in decoded
    assert "AP@IoU 0.50" in decoded
    assert "適合率@IoU 0.50" in decoded
    assert "bboxなし" in decoded
    assert "マスク" in decoded
    assert "りんご" in decoded


def test_summary_handles_single_mode_none_ap_and_splits_category_charts(tmp_path, monkeypatch):
    categories = [{"id": category_id, "name": f"category-{category_id}"} for category_id in range(1, 22)]
    chart_calls = []
    original_write_chart = reporting._write_category_metrics_chart

    def capture_chart(*args, **kwargs):
        chart_calls.append((kwargs["geometry"], args[2]))
        return original_write_chart(*args, **kwargs)

    monkeypatch.setattr(reporting, "_write_category_metrics_chart", capture_chart)
    rows = [
        {
            "mode": "no_bbox",
            "geometry": geometry,
            "scope": "overall",
            "category_id": None,
            "category_name": "全体",
            "ap50_95": None,
            "ap50": None,
            "ap75": None,
        }
        for geometry in ("mask", "bbox")
    ]
    for geometry in ("mask", "bbox"):
        for category in categories:
            rows.append(
                {
                    "mode": "no_bbox",
                    "geometry": geometry,
                    "scope": "category",
                    "category_id": category["id"],
                    "category_name": category["name"],
                    "gt_count": 0,
                    "ap50_95": None,
                    "ap50": None,
                    "ap75": None,
                    "precision_iou_0.50": 0.0,
                    "recall_iou_0.50": 0.0,
                    "f1_iou_0.50": 0.0,
                }
            )

    reporting.write_summary_images(
        tmp_path,
        rows,
        {"label_ja": {}, "dpi": 60},
        categories,
        tmp_path,
        [0.5],
        [0.5, 0.75],
        0.42,
    )

    assert (tmp_path / "comparison_summary.png").is_file()
    for geometry in ("mask", "bbox"):
        assert (tmp_path / "no_bbox" / f"category_metrics_{geometry}_001.png").is_file()
        assert (tmp_path / "no_bbox" / f"category_metrics_{geometry}_002.png").is_file()
    assert chart_calls == [
        ("mask", list(range(1, 21))),
        ("bbox", list(range(1, 21))),
        ("mask", [21]),
        ("bbox", [21]),
    ]
    assert not (tmp_path / "bbox").exists()


def test_category_charts_separate_geometry_and_pr_metric_colors(tmp_path, monkeypatch):
    captured = []
    original_subplots = reporting.plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        captured.append((figure, axes))
        return figure, axes

    monkeypatch.setattr(reporting.plt, "subplots", capture_subplots)
    rows = [
        {
            "geometry": geometry,
            "category_id": 1,
            "ap50_95": 0.5 if geometry == "mask" else 0.1,
            "ap50": 0.6 if geometry == "mask" else 0.2,
            "ap75": 0.4 if geometry == "mask" else 0.3,
            **{
                f"{metric}_iou_{threshold:.2f}": value
                for metric, values in (
                    ("precision", (0.4, 0.5)),
                    ("recall", (0.6, 0.7)),
                    ("f1", (0.5, 0.6)),
                )
                for threshold, value in zip((0.5, 0.75), values)
            },
        }
        for geometry in ("mask", "bbox")
    ]
    font_path = reporting._find_japanese_font({}, ["rice"], tmp_path)

    for geometry in ("mask", "bbox"):
        reporting._write_category_metrics_chart(
            tmp_path / f"category_metrics_{geometry}_001.png",
            "no_bbox",
            [1],
            rows,
            {1: "rice"},
            [0.5, 0.75],
            font_path,
            60,
            geometry=geometry,
        )

    assert len(captured) == 2
    for (figure, axes), geometry in zip(captured, ("mask", "bbox")):
        assert axes.shape == (1, 2)
        assert len(axes[0][0].containers) == 3
        expected_ap = (0.5, 0.6, 0.4) if geometry == "mask" else (0.1, 0.2, 0.3)
        assert tuple(
            container.patches[0].get_width()
            for container in axes[0][0].containers
        ) == expected_ap
        pr_axis = axes[0][1]
        labels = [text.get_text() for text in pr_axis.get_legend().get_texts()]
        assert labels == [
            "Precision@0.50",
            "Recall@0.50",
            "F1@0.50",
            "Precision@0.75",
            "Recall@0.75",
            "F1@0.75",
        ]
        colors = [
            container.patches[0].get_facecolor()
            for container in pr_axis.containers
        ]
        assert len({tuple(color) for color in colors}) == 6
        for metric_indexes in ((0, 3), (1, 4), (2, 5)):
            first, second = (colors[index] for index in metric_indexes)
            assert tuple(first) != tuple(second)
            assert np.argmax(first[:3]) == np.argmax(second[:3])
        assert [np.argmax(color[:3]) for color in colors[:3]] == [2, 1, 0]
        assert (tmp_path / f"category_metrics_{geometry}_001.png").is_file()
        assert figure.axes[0].get_title().startswith(
            "マスク" if geometry == "mask" else "bbox"
        )
        assert figure.axes[1].get_title().endswith("P/R/F1")


def test_category_chart_renders_labels_and_marks_zero_metrics(tmp_path, monkeypatch):
    captured = []
    original_subplots = reporting.plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        captured.append((figure, axes))
        return figure, axes

    monkeypatch.setattr(reporting.plt, "subplots", capture_subplots)
    category_ids = list(range(1, 21))
    category_names = {category_id: f"category-{category_id}" for category_id in category_ids}
    category_names.update(
        {
            4: "チョコレート",
            5: "ビスケット",
            9: "チーズ・バター",
        }
    )
    ap_values = {
        4: (0.850495, 1.0, 1.0),
        5: (0.711663, 1.0, 0.789879),
        9: (0.220892, 0.317954, 0.274181),
    }
    rows = [
        {
            "geometry": geometry,
            "category_id": category_id,
            "ap50_95": ap_values.get(category_id, (None, None, None))[0],
            "ap50": ap_values.get(category_id, (None, None, None))[1],
            "ap75": ap_values.get(category_id, (None, None, None))[2],
            **{
                f"{metric}_iou_{threshold:.2f}": (
                    None if category_id == 1 else 0.0
                )
                for metric in ("precision", "recall", "f1")
                for threshold in (0.5, 0.75)
            },
        }
        for geometry in ("mask", "bbox")
        for category_id in category_ids
    ]
    font_path = reporting._find_japanese_font(
        {}, list(category_names.values()), tmp_path
    )

    for geometry in ("mask", "bbox"):
        reporting._write_category_metrics_chart(
            tmp_path / f"category_metrics_{geometry}_001.png",
            "no_bbox",
            category_ids,
            rows,
            category_names,
            [0.5, 0.75],
            font_path,
            150,
            geometry=geometry,
        )

    assert len(captured) == 2
    for (figure, axes), geometry in zip(captured, ("mask", "bbox")):
        image_path = tmp_path / f"category_metrics_{geometry}_001.png"
        image = Image.open(image_path).convert("RGB")
        assert image.width > 1500
        assert image.height > 1500

        ap_axis, pr_axis = axes[0]
        assert [tick.get_text() for tick in ap_axis.get_yticklabels()] == [
            category_names[category_id] for category_id in category_ids
        ]
        assert [tick.get_text() for tick in ap_axis.get_xticklabels()] == [
            "0.0",
            "0.2",
            "0.4",
            "0.6",
            "0.8",
            "1.0",
        ]
        assert [text.get_text() for text in ap_axis.get_legend().get_texts()] == [
            "AP50-95",
            "AP50",
            "AP75",
        ]
        assert [tick.get_text() for tick in pr_axis.get_yticklabels()] == [
            category_names[category_id] for category_id in category_ids
        ]
        assert [text.get_text() for text in pr_axis.get_legend().get_texts()] == [
            "Precision@0.50",
            "Recall@0.50",
            "F1@0.50",
            "Precision@0.75",
            "Recall@0.75",
            "F1@0.75",
        ]

        # Six zero-valued P/R/F1 series have a visible marker for each non-None
        # category; the None row must not be presented as zero.
        assert len(pr_axis.collections) == 6
        assert [collection.get_offsets().shape[0] for collection in pr_axis.collections] == [
            19,
        ] * 6
        assert all(
            tick.get_fontsize() >= 10
            for tick in ap_axis.get_xticklabels() + ap_axis.get_yticklabels()
        )
        assert all(
            text.get_fontsize() >= 9
            for text in ap_axis.get_legend().get_texts()
            + pr_axis.get_legend().get_texts()
        )

        # Check the saved PNG margins where category names, x tick values and
        # the two legends are rendered (not just the corresponding text artists).
        for box in (
            (0, 100, 160, 1800),
            (150, 1850, 1000, 1910),
            (700, 1740, 950, 1850),
            (1810, 1740, 2180, 1850),
        ):
            pixels = np.asarray(image.crop(box))
            assert np.count_nonzero(np.any(pixels < 100, axis=2)) >= 100


def test_comparison_summary_plots_each_threshold_and_confidence(monkeypatch, tmp_path):
    captured = {}
    original_subplots = reporting.plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        captured["figure"] = figure
        captured["axes"] = axes
        return figure, axes

    monkeypatch.setattr(reporting.plt, "subplots", capture_subplots)
    font_path = reporting._find_japanese_font({}, [], tmp_path)
    rows = []
    for mode, value in (("no_bbox", 0.4), ("bbox", 0.8)):
        for geometry in ("mask", "bbox"):
            row = {
                "mode": mode,
                "geometry": geometry,
                "scope": "overall",
                "ap_iou_0.50": value,
                "ap_iou_0.75": value / 2,
            }
            row.update(
                {
                    f"{metric}_iou_{threshold:.2f}": value
                    for metric in ("precision", "recall", "f1")
                    for threshold in (0.5, 0.75)
                }
            )
            rows.append(row)

    reporting._write_comparison_summary(
        tmp_path,
        rows,
        ["no_bbox", "bbox"],
        font_path,
        60,
        [0.5, 0.75],
        [0.5, 0.75],
        0.42,
    )

    axes = captured["axes"]
    assert len(axes[0][0].lines) == 2
    assert list(axes[0][0].lines[0].get_xdata()) == [0.5, 0.75]
    assert list(axes[0][0].lines[0].get_ydata()) == [0.4, 0.2]
    assert len(axes[1][0].lines) == 6
    assert [tick.get_text() for tick in axes[1][0].get_xticklabels()] == ["0.50", "0.75"]
    assert any("0.42" in text.get_text() for text in captured["figure"].texts)
    assert any("条件付き評価" in text.get_text() for text in captured["figure"].texts)
    assert (tmp_path / "comparison_summary.png").is_file()


def test_overlay_is_saved_inside_condition_directory(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (4, 4), "white").save(image_dir / "sample.png")
    font_path = reporting._find_japanese_font({}, ["rice"], tmp_path)
    mask = np.ones((4, 4), dtype=bool)
    instance = _instance(1, [0, 0, 4, 4], mask)
    written = write_overlays(
        tmp_path,
        "no_bbox",
        [{"id": 1, "file_name": "sample.png"}],
        image_dir,
        {1: [instance]},
        {1: [instance]},
        {1: "rice"},
        {},
        font_path,
        60,
        0.5,
    )

    assert written == 1
    overlays = list((tmp_path / "no_bbox" / "overlays").glob("*.png"))
    assert len(overlays) == 1
    with Image.open(overlays[0]) as composite:
        assert composite.size == (12, 80)
