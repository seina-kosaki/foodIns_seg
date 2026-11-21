"""SAM3画像セグメンテーション推論を提供する。"""

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


def _validate_image(image: NDArray[np.generic]) -> None:
    """BGR画像がSAM3推論の入力条件を満たすことを確認する。"""
    if not isinstance(image, np.ndarray):
        raise ValueError("入力画像はNumPy配列で指定してください。")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("入力画像の形状は H×W×3 にしてください。")
    if image.dtype != np.uint8:
        raise ValueError("入力画像のデータ型は uint8 にしてください。")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("入力画像が空です。画像の内容を確認してください。")


def _load_settings() -> tuple[str, float]:
    """SAM3のプロンプトと信頼度しきい値を設定ファイルから読み込む。"""
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
        raise ValueError(f"SAM3設定ファイルはキーと値の対応形式で記述してください: {_SETTINGS_PATH}")

    prompt = settings.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"SAM3設定ファイルのキー 'prompt' に空でない文字列を指定してください: {_SETTINGS_PATH}")

    threshold = settings.get("confidence_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
        raise ValueError(f"SAM3設定ファイルのキー 'confidence_threshold' は0から1の数値で指定してください: {_SETTINGS_PATH}")
    if not 0 <= threshold <= 1:
        raise ValueError(f"SAM3設定ファイルのキー 'confidence_threshold' は0から1の範囲で指定してください: {_SETTINGS_PATH}")

    return prompt, float(threshold)


def _load_runtime(confidence_threshold: float) -> tuple[Any, Any, Any, str]:
    """SAM3モデル、プロセッサ、実行デバイスを初期化する。"""
    try:
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam3_image_model(device=device).to(device=device)
        model.eval()
        processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
        return torch, model, processor, device
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("SAM3モデル読み込み失敗") from exc


def _to_numpy(value: Any) -> np.ndarray:
    """PyTorch tensorまたは配列相当の値をNumPy配列に変換する。"""
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _parse_candidates(output: dict[str, Any], height: int, width: int) -> list[dict[str, Any]]:
    """SAM3出力を画像寸法検証済みの検出結果へ変換する。"""
    masks, boxes, scores = output.get("masks"), output.get("boxes"), output.get("scores")
    if masks is None or boxes is None or scores is None:
        return []

    candidates = []
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


def _validate_bboxes(bboxes: Iterable[ArrayLike] | None, height: int, width: int) -> list[list[float]]:
    """入力bboxを検証し、xyxy形式のfloatリストとして返す。"""
    if bboxes is None:
        return []
    try:
        supplied_boxes = list(bboxes)
    except TypeError as exc:
        raise ValueError("bboxはxyxy座標のリストで指定してください。") from exc

    validated = []
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
    """設定に基づいて画像セグメンテーションを実行する。"""

    def __init__(self) -> None:
        """SAM3設定とモデルを読み込み、推論器を初期化する。"""
        self.prompt, self.confidence_threshold = _load_settings()
        self._torch, self.model, self.processor, self.device = _load_runtime(self.confidence_threshold)

    def _autocast(self) -> AbstractContextManager[Any]:
        """CUDA利用時に自動混合精度を有効にするコンテキストを返す。"""
        if self.device == "cuda":
            return self._torch.autocast(device_type="cuda", dtype=self._torch.bfloat16)
        return nullcontext()

    def infer(self, image: NDArray[np.uint8], bboxes: Iterable[ArrayLike] | None = None) -> list[dict[str, Any]]:
        """画像をセグメントし、confidence・xyxy・maskを返す。

        Args:
            image: OpenCV形式のBGR uint8画像。
            bboxes: 任意のxyxy座標4個からなるbboxの反復可能オブジェクト。
                Noneまたは空の場合は画像全体を推論する。

        Returns:
            セグメンテーション結果。各要素はconfidence、xyxy、bool mask、
            および入力bboxに対応するsource_xyxyを含む。

        Raises:
            ValueError: 画像またはbboxが入力条件を満たさない場合。
            RuntimeError: モデル推論に失敗した場合。
        """
        _validate_image(image)
        height, width = image.shape[:2]
        source_boxes = _validate_bboxes(bboxes, height, width)
        rgb_image = Image.fromarray(image[:, :, ::-1].copy())

        try:
            with self._torch.inference_mode(), self._autocast():
                state = self.processor.set_image(rgb_image)
                if not source_boxes:
                    output = self.processor.set_text_prompt(state=state, prompt=self.prompt)
                    return [{**candidate, "source_xyxy": None} for candidate in _parse_candidates(output, height, width)]

                detections = []
                for source_box in source_boxes:
                    self.processor.reset_all_prompts(state)
                    self.processor.set_text_prompt(state=state, prompt=self.prompt)
                    x1, y1, x2, y2 = source_box
                    normalized_box = [
                        (x1 + x2) / 2 / width,
                        (y1 + y2) / 2 / height,
                        (x2 - x1) / width,
                        (y2 - y1) / height,
                    ]
                    output = self.processor.add_geometric_prompt(box=normalized_box, label=True, state=state)
                    candidates = _parse_candidates(output, height, width)
                    if candidates:
                        best = max(candidates, key=lambda candidate: candidate["confidence"])
                        detections.append({**best, "source_xyxy": source_box})
                return detections
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError("SAM3推論に失敗しました。入力画像とSAM3の実行環境を確認してください。") from exc
