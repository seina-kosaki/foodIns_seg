"""推論DB保存とDB解析の局所テスト。"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import analyze, main, main_infer
from src.sam3_inference.storage import create_database, save_image_transaction


def _database(path: Path):
    return create_database(
        path,
        "run_00001",
        {
            "started_at": "2026-01-01T00:00:00+00:00",
            "settings": {
                "evaluation": {
                    "modes": ["no_bbox"],
                    "iou_thresholds": [0.5],
                    "report_thresholds": [0.5],
                    "pr_confidence_threshold": 0.5,
                }
            },
        },
        "annotations.json",
        "images",
        ["no_bbox"],
        "cpu",
        [{"id": 1, "name": "rice"}],
    )


def _mask_record():
    encoded = mask_utils.encode(np.asfortranarray(np.ones((2, 2), dtype=np.uint8)))
    if isinstance(encoded["counts"], bytes):
        encoded["counts"] = encoded["counts"].decode("ascii")
    return {"size": encoded["size"], "counts": encoded["counts"]}


def _image(image_id):
    return {
        "image_id": image_id,
        "image_path": f"image_{image_id}.png",
        "width": 2,
        "height": 2,
        "processed_at": "2026-01-01T00:00:00+00:00",
    }


def _ground_truth():
    return [{
        "annotation_id": 11,
        "category_id": 1,
        "xyxy": [0.0, 0.0, 2.0, 2.0],
        "mask": _mask_record(),
        "iscrowd": 0,
    }]


def test_image_results_commit_as_separate_transactions_and_rle_is_json(tmp_path):
    connection = _database(tmp_path / "inference.sqlite3")
    save_image_transaction(connection, _image(1), _ground_truth(), [])
    assert connection.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1
    save_image_transaction(connection, _image(2), _ground_truth(), [])
    assert connection.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 2
    mask_json = connection.execute("SELECT mask_json FROM ground_truths").fetchone()[0]
    decoded = json.loads(mask_json)
    assert decoded["size"] == [2, 2]
    assert isinstance(decoded["counts"], str)
    connection.close()


def test_failed_image_transaction_leaves_no_partial_records(tmp_path):
    connection = _database(tmp_path / "inference.sqlite3")
    prediction = {
        "mode": "no_bbox",
        "category_id": 999,
        "category_name": "unknown",
        "prompt": "unknown",
        "confidence": 0.9,
        "xyxy": [0, 0, 1, 1],
        "source_xyxy": None,
        "mask": _mask_record(),
    }
    with pytest.raises(sqlite3.IntegrityError):
        save_image_transaction(connection, _image(1), _ground_truth(), [prediction])
    assert connection.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM ground_truths").fetchone()[0] == 0
    assert connection.execute(
        "SELECT successful_image_count FROM run_info"
    ).fetchone()[0] == 0
    connection.close()


def test_per_gt_and_comparison_transition_are_generated():
    mask = _mask_record()
    gt = {1: [{
        "annotation_id": 11,
        "image_id": 1,
        "category_id": 1,
        "xyxy": [0.0, 0.0, 2.0, 2.0],
        "mask": mask,
        "iscrowd": 0,
    }]}
    no_bbox = analyze._build_per_gt_rows(
        "no_bbox", gt, {}, {1: "rice"}, {}, 0.5
    )
    bbox = analyze._build_per_gt_rows(
        "bbox",
        gt,
        {1: [{
            "category_id": 1,
            "confidence": 0.9,
            "xyxy": [0.0, 0.0, 2.0, 2.0],
            "source_xyxy": [0.0, 0.0, 2.0, 2.0],
            "mask": mask,
        }]},
        {1: "rice"},
        {},
        0.5,
    )
    comparison_gt, transition, _ = analyze._comparison_rows(no_bbox, bbox, [], 0.005)
    assert no_bbox[0]["state"] == "見逃し"
    assert bbox[0]["state"] == "成功"
    assert comparison_gt[0]["A_state"] == "見逃し"
    assert comparison_gt[0]["B_state"] == "成功"
    rescued = next(
        row for row in transition
        if row["A_state"] == "見逃し" and row["B_state"] == "成功"
    )
    assert rescued["count"] == 1


def test_analysis_reads_saved_rle_without_reconstructing_annotations(tmp_path):
    database_path = tmp_path / "inference.sqlite3"
    connection = _database(database_path)
    save_image_transaction(
        connection,
        _image(1),
        _ground_truth(),
        [{
            "mode": "no_bbox",
            "category_id": 1,
            "category_name": "rice",
            "prompt": "rice",
            "confidence": 0.8,
            "xyxy": [0.0, 0.0, 2.0, 2.0],
            "source_xyxy": None,
            "mask": _mask_record(),
        }],
    )
    connection.close()
    data = analyze._read_database(database_path)
    gt = data["ground_truth_by_image"][1][0]
    pred = data["predictions_by_mode"]["no_bbox"][1][0]
    assert gt["mask"]["size"] == [2, 2]
    assert pred["mask"]["counts"] == _mask_record()["counts"]
    assert data["categories"] == [{"id": 1, "name": "rice"}]


def test_analysis_writes_outputs_under_configured_run_only(tmp_path, monkeypatch):
    results = tmp_path / "results"
    run_directory = results / "run_00001"
    run_directory.mkdir(parents=True)
    analysis_directory = run_directory / "analysis"
    stale_directory = analysis_directory / "no_bbox" / "overlays"
    stale_directory.mkdir(parents=True)
    (stale_directory / "old_visualization.png").write_bytes(b"old")
    import cv2

    assert cv2.imwrite(str(tmp_path / "image_1.png"), np.zeros((2, 2, 3), dtype=np.uint8))
    connection = _database(run_directory / "inference.sqlite3")
    connection.execute("UPDATE run_info SET image_dir = ?", (str(tmp_path),))
    connection.commit()
    save_image_transaction(
        connection,
        _image(1),
        _ground_truth(),
        [{
            "mode": "no_bbox",
            "category_id": 1,
            "category_name": "rice",
            "prompt": "rice",
            "confidence": 0.8,
            "xyxy": [0.0, 0.0, 2.0, 2.0],
            "source_xyxy": None,
            "mask": _mask_record(),
        }],
    )
    connection.close()
    current_settings = {
        "analysis_run_id": "run_00001",
        "output": {
            "results_dir": str(results),
            "visualization_samples": 1,
            "label_ja": {},
            "dpi": 60,
        },
    }
    monkeypatch.setattr(analyze, "load_settings", lambda: current_settings)

    analyze.run_analysis()

    assert (analysis_directory / "comparison_summary.png").is_file()
    assert (analysis_directory / "no_bbox" / "metrics.csv").is_file()
    assert (analysis_directory / "no_bbox" / "category_ap_mask_001.png").is_file()
    assert (analysis_directory / "no_bbox" / "category_ap_bbox_001.png").is_file()
    assert (analysis_directory / "no_bbox" / "category_heatmap_mask_001.png").is_file()
    assert (analysis_directory / "no_bbox" / "category_heatmap_bbox_001.png").is_file()
    assert not (stale_directory / "old_visualization.png").exists()
    assert len(list((analysis_directory / "no_bbox" / "overlays").glob("*.png"))) == 1


def test_main_is_compatibility_delegate(monkeypatch):
    called = []
    monkeypatch.setattr(main, "_main", lambda: called.append(True))
    main.main()
    assert called == [True]
    assert main.run_inference is main_infer.run_inference


def test_inference_reuses_segmenter_and_saves_only_completed_image_results(tmp_path, monkeypatch):
    image_directory = tmp_path / "input"
    image_directory.mkdir()
    import cv2

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_directory / "one.png"), image)
    assert cv2.imwrite(str(image_directory / "two.png"), image)
    annotation = {
        "images": [
            {"id": 1, "file_name": "one.png"},
            {"id": 2, "file_name": "two.png"},
            {"id": 3, "file_name": "missing.png"},
        ],
        "annotations": [
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [0, 0, 2, 2],
                "segmentation": [[0, 0, 2, 0, 2, 2, 0, 2]],
                "iscrowd": 0,
            }
            for image_id in (1, 2, 3)
        ],
        "categories": [{"id": 1, "name": "rice"}],
    }
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    settings = {
        "run_id": "auto",
        "dataset": {
            "annotation_path": str(annotation_path),
            "image_dir": str(image_directory),
            "include_crowd": False,
            "image_limit": None,
            "image_ids": None,
        },
        "evaluation": {
            "modes": ["no_bbox", "bbox"],
            "iou_thresholds": [0.5],
            "report_thresholds": [0.5],
            "pr_confidence_threshold": 0.5,
        },
        "output": {"results_dir": str(tmp_path / "results")},
    }
    monkeypatch.setattr(main_infer, "load_settings", lambda: settings)
    instances = []

    class FakeSegmenter:
        confidence_threshold = 0.5
        device = "cpu"

        def __init__(self):
            instances.append(self)

        def infer(self, image, category_ids, bboxes=None):
            return [{
                "category_id": category_ids[0],
                "category_name": "rice",
                "prompt": "rice",
                "confidence": 0.9,
                "xyxy": [0.0, 0.0, 2.0, 2.0],
                "source_xyxy": bboxes[0] if bboxes else None,
                "mask": np.ones((4, 4), dtype=bool),
            }]

    monkeypatch.setattr(main_infer, "Sam3Segmenter", FakeSegmenter)
    main_infer.run_inference()

    run_directory = tmp_path / "results" / "run_00001"
    connection = sqlite3.connect(run_directory / "inference.sqlite3")
    assert len(instances) == 1
    assert connection.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM ground_truths").fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 4
    run = connection.execute(
        "SELECT state, successful_image_count, skipped_image_count FROM run_info"
    ).fetchone()
    assert run == ("complete", 2, 1)
    connection.close()
    assert not (run_directory / "analysis").exists()
