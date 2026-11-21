"""settings_sam3.yamlの設定に従ってSAM3推論を実行する。"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sam3_inference.inference_categories import Sam3Segmenter

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


def run_inference() -> None:
    """YAML設定に従ってデータセットを推論し、予測情報をJSON保存する。"""
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

    results_dir_value = output_settings.get("results_dir", "results")
    if not isinstance(results_dir_value, str):
        raise ValueError("output.results_dirは文字列で指定してください。")

    run_directory = create_run_directory(resolve_project_path(results_dir_value))
    LOGGER.info("結果保存先: %s", run_directory)
    LOGGER.info("対象画像数: %d", len(images))
    LOGGER.info("推論モード: %s", ", ".join(modes))

    # モデルは一度だけ初期化し、全画像・全モードで共有する。
    segmenter = Sam3Segmenter()
    prediction_records: list[dict[str, Any]] = []
    processed_image_count = 0

    for image_index, image_info in enumerate(images, start=1):
        image_id = int(image_info["id"])
        image_path = image_dir / str(image_info["file_name"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            LOGGER.warning("画像を読み込めないためスキップします: %s", image_path)
            continue

        processed_image_count += 1
        image_annotations = annotations_by_image.get(image_id, [])

        LOGGER.info(
            "[%d/%d] image_id=%d file=%s",
            image_index,
            len(images),
            image_id,
            image_info["file_name"],
        )

        for mode in modes:
            if mode == "no_bbox":
                # 画像内に存在するGTカテゴリを重複なくプロンプトにする。
                category_ids = sorted({int(annotation["category_id"]) for annotation in image_annotations})
                if not category_ids:
                    continue

                predictions = segmenter.infer(
                    image=image,
                    category_ids=category_ids,
                )
            else:
                # COCOのXYWH形式を推論器が要求するXYXY形式に変換する。
                category_ids = []
                bboxes = []

                for annotation_index, annotation in enumerate(image_annotations):
                    x, y, bbox_width, bbox_height = annotation["bbox"]

                    bbox = [
                        float(x),
                        float(y),
                        float(x + bbox_width),
                        float(y + bbox_height),
                    ]

                    image_height, image_width = image.shape[:2]

                    x1, y1, x2, y2 = bbox

                    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height or x1 >= x2 or y1 >= y2:
                        LOGGER.error(
                            "Invalid bbox: image_id=%s, annotation_id=%s, index=%d, bbox=%s, image_size=(%d, %d)",
                            image_id,
                            annotation["id"],
                            annotation_index,
                            bbox,
                            image_width,
                            image_height,
                        )

                    category_ids.append(int(annotation["category_id"]))
                    bboxes.append(bbox)

                if not category_ids:
                    continue

                predictions = segmenter.infer(
                    image=image,
                    category_ids=category_ids,
                    bboxes=bboxes,
                )

            prediction_records.extend(prediction_to_record(image_id, mode, prediction) for prediction in predictions)

    predictions_path = run_directory / "predictions.json"
    with predictions_path.open("w", encoding="utf-8") as file:
        json.dump(prediction_records, file, ensure_ascii=False, indent=2)

    run_info = {
        "annotation_path": str(annotation_path),
        "image_dir": str(image_dir),
        "modes": modes,
        "selected_image_count": len(images),
        "processed_image_count": processed_image_count,
        "prediction_count": len(prediction_records),
        "confidence_threshold": segmenter.confidence_threshold,
        "device": segmenter.device,
    }
    run_info_path = run_directory / "run_info.json"
    with run_info_path.open("w", encoding="utf-8") as file:
        json.dump(run_info, file, ensure_ascii=False, indent=2)

    LOGGER.info("推論完了: %d件の予測", len(prediction_records))
    LOGGER.info("予測結果: %s", predictions_path)
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
