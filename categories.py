"""Display statistics from FoodInsSeg COCO-format annotations."""

import json
from pathlib import Path


def load_categories(annotation_path: Path) -> None:
    """Display categories and summary statistics from an annotation JSON.

    Args:
        annotation_path: Path to the COCO-format annotation JSON.
    """
    with annotation_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    categories = data["categories"]
    annotations = data["annotations"]
    images = data["images"]

    print(f"Total categories: {len(categories)}")

    for category in categories:
        print(f"ID: {category['id']:3d} | Name: {category['name']}")

    print(sum(a.get("iscrowd", 0) for a in annotations), "/", len(annotations))

    pairs = {(a["image_id"], a["category_id"]) for a in annotations}
    print(len(images), "枚 /", len(pairs), "組")

    image_sizes = {}
    for image in images:
        size = (image["width"], image["height"])
        image_sizes[size] = image_sizes.get(size, 0) + 1

    print(f"ユニークサイズ数: {len(image_sizes)} / 総画像数: {len(images)}")
    for (width, height), count in sorted(image_sizes.items()):
        print(f"サイズ ({width}, {height}): {count} 枚")


if __name__ == "__main__":
    load_categories(Path("datasets/annotations/Test.json"))
