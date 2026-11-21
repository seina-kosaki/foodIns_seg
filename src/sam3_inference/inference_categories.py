"""SAM3画像セグメンテーション推論を提供する。"""

import json
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_PATH = _PROJECT_ROOT / "configs" / "settings_sam3.yaml"


def _validate_image(image: NDArray[np.uint8]) -> None:
    """BGR画像がSAM3推論の入力条件を満たすことを確認する。"""
    if not isinstance(image, np.ndarray):
        raise ValueError("入力画像はNumPy配列で指定してください。")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("入力画像の形状は H×W×3 にしてください。")
    if image.dtype != np.uint8:
        raise ValueError("入力画像のデータ型は uint8 にしてください。")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("入力画像が空です。画像の内容を確認してください。")


def _load_settings() -> tuple[float, str, dict[str, str], Path]:
    """SAM3設定とカテゴリ別プロンプト設定を読み込む。"""
    try:
        with _SETTINGS_PATH.open(encoding="utf-8") as settings_file:
            settings = yaml.safe_load(settings_file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"SAM3設定ファイルが見つかりません: {_SETTINGS_PATH}") from exc
    except (OSError, UnicodeError) as exc:
        raise OSError(f"SAM3設定ファイルを読み込めませんでした: {_SETTINGS_PATH}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"SAM3設定ファイルのYAML形式が正しくありません: {_SETTINGS_PATH}") from exc

    if not isinstance(settings, dict):
        raise ValueError("SAM3設定ファイルはキーと値の対応形式で記述してください。")

    sam3_settings = settings.get("sam3")
    dataset_settings = settings.get("dataset")
    prompt_settings = settings.get("prompt")
    if not isinstance(sam3_settings, dict):
        raise ValueError("設定ファイルに 'sam3' セクションが必要です。")
    if not isinstance(dataset_settings, dict):
        raise ValueError("設定ファイルに 'dataset' セクションが必要です。")
    if not isinstance(prompt_settings, dict):
        raise ValueError("設定ファイルに 'prompt' セクションが必要です。")

    threshold = sam3_settings.get("confidence_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("'sam3.confidence_threshold' は0から1の数値で指定してください。")

    template = prompt_settings.get("template")
    if not isinstance(template, str) or not template.strip():
        raise ValueError("'prompt.template' に空でない文字列を指定してください。")
    if "{name}" not in template:
        raise ValueError("'prompt.template' には '{name}' を含めてください。")

    overrides = prompt_settings.get("overrides", {})
    if not isinstance(overrides, dict) or not all(isinstance(key, str) and isinstance(value, str) and value.strip() for key, value in overrides.items()):
        raise ValueError("'prompt.overrides' はカテゴリ名とプロンプト文字列の対応にしてください。")

    annotation_value = dataset_settings.get("annotation_path")
    if not isinstance(annotation_value, str) or not annotation_value.strip():
        raise ValueError("'dataset.annotation_path' を指定してください。")

    annotation_path = Path(annotation_value)
    if not annotation_path.is_absolute():
        annotation_path = _PROJECT_ROOT / annotation_path

    return float(threshold), template, overrides, annotation_path


def _load_categories(annotation_path: Path) -> dict[int, str]:
    """COCOアノテーションJSONからカテゴリIDとカテゴリ名を読み込む。"""
    try:
        with annotation_path.open(encoding="utf-8") as annotation_file:
            data = json.load(annotation_file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"アノテーションファイルが見つかりません: {annotation_path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"アノテーションJSONを読み込めませんでした: {annotation_path}") from exc

    categories = data.get("categories") if isinstance(data, dict) else None
    if not isinstance(categories, list) or not categories:
        raise ValueError("アノテーションJSONに有効な 'categories' がありません。")

    category_map: dict[int, str] = {}
    for category in categories:
        if not isinstance(category, dict):
            raise ValueError("categoriesの各要素はオブジェクトである必要があります。")
        category_id = category.get("id")
        category_name = category.get("name")
        if isinstance(category_id, bool) or not isinstance(category_id, int) or not isinstance(category_name, str) or not category_name.strip():
            raise ValueError("カテゴリには整数のidと空でないnameが必要です。")
        if category_id in category_map:
            raise ValueError(f"カテゴリIDが重複しています: {category_id}")
        category_map[category_id] = category_name

    return category_map


def _load_runtime(confidence_threshold: float) -> tuple[Any, Any, Any, str]:
    """SAM3モデル、プロセッサ、実行デバイスを初期化する。"""
    try:
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam3_image_model(device=device).to(device=device)
        model.eval()
        processor = Sam3Processor(
            model,
            confidence_threshold=confidence_threshold,
            device=device,
        )
        return torch, model, processor, device
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("SAM3モデル読み込み失敗") from exc


def _to_numpy(value: Any) -> np.ndarray:
    """PyTorch tensorまたは配列相当の値をNumPy配列に変換する。"""
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _parse_candidates(
    output: dict[str, Any],
    height: int,
    width: int,
) -> list[dict[str, Any]]:
    """SAM3出力を画像寸法検証済みの検出結果へ変換する。"""
    masks = output.get("masks")
    boxes = output.get("boxes")
    scores = output.get("scores")
    if masks is None or boxes is None or scores is None:
        return []

    candidates: list[dict[str, Any]] = []
    for mask, box, score in zip(masks, boxes, scores, strict=True):
        binary_mask = np.squeeze(_to_numpy(mask)).astype(bool, copy=False)
        if binary_mask.shape != (height, width):
            raise RuntimeError("SAM3のマスク形状が入力画像と一致しません。")
        candidates.append(
            {
                "confidence": float(_to_numpy(score)),
                "xyxy": [float(coordinate) for coordinate in _to_numpy(box).tolist()],
                "mask": binary_mask,
            }
        )
    return candidates


def _validate_bboxes(
    bboxes: Iterable[ArrayLike] | None,
    height: int,
    width: int,
) -> list[list[float]]:
    """入力bboxを検証し、xyxy形式のfloatリストとして返す。"""
    if bboxes is None:
        return []

    try:
        supplied_boxes = list(bboxes)
    except TypeError as exc:
        raise ValueError("bboxはxyxy座標のリストで指定してください。") from exc

    validated: list[list[float]] = []
    for index, bbox in enumerate(supplied_boxes):
        values = np.asarray(bbox)
        if values.shape != (4,) or values.dtype.kind not in "iuf":
            raise ValueError(f"bbox {index} は有限な数値4個のxyxy形式で指定してください。")
        coordinates = values.astype(float).tolist()
        if not np.all(np.isfinite(coordinates)):
            raise ValueError(f"bbox {index} の座標は有限値で指定してください。")
        x1, y1, x2, y2 = coordinates
        tolerance = 1e-6
        if x1 < -tolerance or y1 < -tolerance or x2 > width + tolerance or y2 > height + tolerance:
            raise ValueError(f"bbox {index} は画像内の正の面積を持つxyxy座標で指定してください。")

        coordinates = [
            min(max(x1, 0.0), float(width)),
            min(max(y1, 0.0), float(height)),
            min(max(x2, 0.0), float(width)),
            min(max(y2, 0.0), float(height)),
        ]
        x1, y1, x2, y2 = coordinates
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"bbox {index} は画像内の正の面積を持つxyxy座標で指定してください。")
        validated.append(coordinates)

    return validated


class Sam3Segmenter:
    """COCOカテゴリをプロンプトとして画像セグメンテーションを実行する。"""

    def __init__(self) -> None:
        """設定、COCOカテゴリ、SAM3モデルを読み込む。"""
        (
            self.confidence_threshold,
            self.prompt_template,
            self.prompt_overrides,
            annotation_path,
        ) = _load_settings()
        self.categories = _load_categories(annotation_path)
        self._torch, self.model, self.processor, self.device = _load_runtime(self.confidence_threshold)

    def _autocast(self) -> AbstractContextManager[Any]:
        """CUDA利用時に自動混合精度を有効にするコンテキストを返す。"""
        if self.device == "cuda":
            return self._torch.autocast(
                device_type="cuda",
                dtype=self._torch.bfloat16,
            )
        return nullcontext()

    def _make_prompt(self, category_id: int) -> str:
        """カテゴリIDからSAM3に渡すテキストプロンプトを生成する。"""
        try:
            category_name = self.categories[category_id]
        except KeyError as exc:
            raise ValueError(f"未知のcategory_idです: {category_id}") from exc

        return self.prompt_overrides.get(
            category_name,
            self.prompt_template.format(name=category_name),
        )

    def infer(
        self,
        image: NDArray[np.uint8],
        category_ids: Iterable[int],
        bboxes: Iterable[ArrayLike] | None = None,
    ) -> list[dict[str, Any]]:
        """カテゴリプロンプトで推論し、confidence・bbox・maskを返す。

        Args:
            image: OpenCV形式のBGR uint8画像。
            category_ids: 推論対象のCOCO category_id。bboxなしでは画像内の
                カテゴリID一覧、bboxありでは各bboxに対応するID一覧を指定する。
            bboxes: 任意のxyxy座標。指定時はcategory_idsと同じ要素数にする。

        Returns:
            category_id、category_name、prompt、confidence、xyxy、mask、
            source_xyxyを含む検出結果のリスト。
        """
        _validate_image(image)
        height, width = image.shape[:2]

        try:
            supplied_category_ids = list(category_ids)
        except TypeError as exc:
            raise ValueError("category_idsは整数IDの反復可能オブジェクトで指定してください。") from exc

        if not supplied_category_ids:
            raise ValueError("category_idsが空です。推論対象カテゴリを指定してください。")
        if any(isinstance(category_id, bool) or not isinstance(category_id, int) for category_id in supplied_category_ids):
            raise ValueError("category_idsは整数で指定してください。")

        source_boxes = _validate_bboxes(bboxes, height, width)
        if source_boxes and len(source_boxes) != len(supplied_category_ids):
            raise ValueError("bboxありの場合、category_idsとbboxesの要素数を一致させてください。")

        if not source_boxes:
            # bboxなしでは同一カテゴリを重複して推論しない。
            supplied_category_ids = list(dict.fromkeys(supplied_category_ids))

        prompt_items = [(category_id, self._make_prompt(category_id)) for category_id in supplied_category_ids]
        rgb_image = Image.fromarray(image[:, :, ::-1].copy())

        try:
            detections: list[dict[str, Any]] = []
            with self._torch.inference_mode(), self._autocast():
                state = self.processor.set_image(rgb_image)

                for index, (category_id, prompt) in enumerate(prompt_items):
                    self.processor.reset_all_prompts(state)
                    source_box: list[float] | None = None

                    if source_boxes:
                        self.processor.set_text_prompt(
                            state=state,
                            prompt=prompt,
                        )
                        source_box = source_boxes[index]
                        x1, y1, x2, y2 = source_box
                        normalized_box = [
                            (x1 + x2) / 2 / width,
                            (y1 + y2) / 2 / height,
                            (x2 - x1) / width,
                            (y2 - y1) / height,
                        ]
                        output = self.processor.add_geometric_prompt(
                            box=normalized_box,
                            label=True,
                            state=state,
                        )
                    else:
                        output = self.processor.set_text_prompt(
                            state=state,
                            prompt=prompt,
                        )

                    candidates = _parse_candidates(output, height, width)
                    if source_box is not None:
                        if candidates:
                            best = max(
                                candidates,
                                key=lambda candidate: candidate["confidence"],
                            )
                            detections.append(
                                {
                                    **best,
                                    "category_id": category_id,
                                    "category_name": self.categories[category_id],
                                    "prompt": prompt,
                                    "source_xyxy": source_box,
                                }
                            )
                        continue

                    for candidate in candidates:
                        detections.append(
                            {
                                **candidate,
                                "category_id": category_id,
                                "category_name": self.categories[category_id],
                                "prompt": prompt,
                                "source_xyxy": None,
                            }
                        )

            return detections
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError("SAM3推論に失敗しました。入力画像とSAM3の実行環境を確認してください。") from exc
