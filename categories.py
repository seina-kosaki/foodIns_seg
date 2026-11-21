import json
from pathlib import Path


def load_categories(annotation_path: Path) -> None:
    """Load and display FoodInsSeg categories."""
    with annotation_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    categories = data["categories"]

    print(f"Total categories: {len(categories)}")

    for category in categories:
        print(f"ID: {category['id']:3d} | Name: {category['name']}")


if __name__ == "__main__":
    load_categories(Path("datasets/annotations/Test.json"))
    data = json.load(open("datasets/annotations/Test.json", encoding="utf-8"))
    print(sum(a.get("iscrowd", 0) for a in data["annotations"]), "/", len(data["annotations"]))

    pairs = {(a["image_id"], a["category_id"]) for a in data["annotations"]}
    print(len(data["images"]), "枚 /", len(pairs), "組")
