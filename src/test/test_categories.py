"""Tests for the FoodInsSeg annotation summary script."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from categories import load_categories


def test_load_categories_displays_image_size_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Display counts grouped by image dimensions."""
    annotation_path = tmp_path / "Test.json"
    annotation_path.write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": "rice"}],
                "annotations": [
                    {"image_id": 1, "category_id": 1, "iscrowd": 0},
                    {"image_id": 2, "category_id": 1, "iscrowd": 1},
                ],
                "images": [
                    {"id": 1, "width": 640, "height": 480},
                    {"id": 2, "width": 640, "height": 480},
                    {"id": 3, "width": 800, "height": 600},
                ],
            }
        ),
        encoding="utf-8",
    )

    load_categories(annotation_path)

    output = capsys.readouterr().out
    assert "Total categories: 1" in output
    assert "1 / 2" in output
    assert "3 枚 / 2 組" in output
    assert "ユニークサイズ数: 2 / 総画像数: 3" in output
    assert "サイズ (640, 480): 2 枚" in output
    assert "サイズ (800, 600): 1 枚" in output
