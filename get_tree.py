import os
from pathlib import Path
from typing import List, Set


def print_directory_tree(
    root_path: Path,
    exclude_dirs: set[str],
    exclude_extensions: set[str],
    indent: str = "",
    is_last: bool = True
) -> None:
    """画像ファイルを除外してディレクトリ構成をツリー形式で表示する。

    Args:
        root_path (Path): 走査を開始するルートパス。
        exclude_dirs (Set[str]): 除外するディレクトリ名の集合。
        exclude_extensions (Set[str]): 除外する拡張子の集合（例: {'.png', '.jpg'}）。
        indent (str): 現在のインデント文字列。
        is_last (bool): 現在の要素がその階層の最後かどうか。
    """
    # 隠しファイル/ディレクトリはスキップ
    if root_path.name.startswith("."):
        return

    # 表示用の記号
    marker = "└── " if is_last else "├── "

    # 自身の名前を表示
    suffix = "/" if root_path.is_dir() else ""
    print(f"{indent}{marker}{root_path.name}{suffix}")

    # 次の階層のインデント
    new_indent = indent + ("    " if is_last else "│   ")

    if root_path.is_dir():
        # 子要素のフィルタリング
        items: list[Path] = []
        for item in sorted(root_path.iterdir()):
            # ディレクトリ除外判定
            if item.is_dir() and item.name in exclude_dirs:
                continue
            # ファイル拡張子除外判定
            if item.is_file() and item.suffix.lower() in exclude_extensions:
                continue
            # 隠し要素除外
            if item.name.startswith("."):
                continue

            items.append(item)

        for i, item in enumerate(items):
            print_directory_tree(
                item,
                exclude_dirs,
                exclude_extensions,
                new_indent,
                is_last=(i == len(items) - 1)
            )

if __name__ == "__main__":
    # プロジェクトルートの設定
    project_root = Path(__file__).parent

    # 除外ディレクトリ設定
    ignored_dirs: set[str] = {
        ".venv", ".mypy_cache", ".ruff_cache", "__pycache__",
        "build", "dist", "node_modules"
    }

    # 除外拡張子設定
    ignored_exts: set[str] = {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"
    }

    print(f"Project: {project_root.name} (Source Structure Only)")

    # ルート直下の子要素から開始
    top_items = [
        item for item in sorted(project_root.iterdir())
        if item.name not in ignored_dirs
        and not item.name.startswith(".")
        and not (item.is_file() and item.suffix.lower() in ignored_exts)
    ]

    for idx, top_item in enumerate(top_items):
        print_directory_tree(
            top_item,
            ignored_dirs,
            ignored_exts,
            is_last=(idx == len(top_items) - 1)
        )
