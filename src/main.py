"""従来の起動経路を推論専用エントリポイントへ委譲する互換ラッパー。"""

from src.analyze import _build_metric_rows
from src.main_infer import (
    SETTINGS_PATH,
    build_annotations_by_image,
    coco_bbox_to_clipped_xyxy,
    limit_predictions_per_category,
    load_coco_annotations,
    load_settings,
    main as _main,
    prediction_to_record,
    resolve_project_path,
    run_inference,
    select_images,
)


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
