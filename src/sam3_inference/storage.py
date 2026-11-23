"""SQLite永続化を推論側と解析側で共有する。"""

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def create_database(
    database_path: Path,
    run_id: str,
    settings_snapshot: dict[str, Any],
    annotation_path: str,
    image_dir: str,
    modes: list[str],
    device: str,
    categories: list[dict[str, Any]],
) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE run_info (
            run_id TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            settings_json TEXT NOT NULL,
            annotation_path TEXT NOT NULL,
            image_dir TEXT NOT NULL,
            modes_json TEXT NOT NULL,
            state TEXT NOT NULL,
            device TEXT NOT NULL,
            successful_image_count INTEGER NOT NULL DEFAULT 0,
            skipped_image_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE images (
            image_id INTEGER PRIMARY KEY,
            image_path TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            status TEXT NOT NULL,
            processed_at TEXT NOT NULL
        );
        CREATE TABLE ground_truths (
            annotation_id INTEGER NOT NULL,
            image_id INTEGER NOT NULL REFERENCES images(image_id),
            category_id INTEGER NOT NULL REFERENCES categories(category_id),
            bbox_json TEXT NOT NULL,
            mask_json TEXT NOT NULL,
            iscrowd INTEGER NOT NULL,
            PRIMARY KEY (image_id, annotation_id)
        );
        CREATE TABLE predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL REFERENCES images(image_id),
            mode TEXT NOT NULL,
            category_id INTEGER NOT NULL REFERENCES categories(category_id),
            category_name TEXT NOT NULL,
            prompt TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox_json TEXT NOT NULL,
            source_bbox_json TEXT,
            mask_json TEXT NOT NULL
        );
        """
    )
    connection.execute(
        """INSERT INTO run_info
        (run_id, schema_version, started_at, settings_json, annotation_path,
         image_dir, modes_json, state, device)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?)""",
        (
            run_id,
            SCHEMA_VERSION,
            settings_snapshot["started_at"],
            json.dumps(settings_snapshot["settings"], ensure_ascii=False),
            annotation_path,
            image_dir,
            json.dumps(modes, ensure_ascii=False),
            device,
        ),
    )
    connection.executemany(
        "INSERT INTO categories(category_id, name) VALUES (?, ?)",
        [(int(item["id"]), str(item["name"])) for item in categories],
    )
    connection.commit()
    return connection


def save_image_transaction(
    connection: sqlite3.Connection,
    image: dict[str, Any],
    ground_truths: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> None:
    try:
        connection.execute("BEGIN")
        connection.execute(
            """INSERT INTO images(image_id, image_path, width, height, status, processed_at)
            VALUES (?, ?, ?, ?, 'processed', ?)""",
            (
                image["image_id"],
                image["image_path"],
                image["width"],
                image["height"],
                image["processed_at"],
            ),
        )
        connection.executemany(
            """INSERT INTO ground_truths
            (annotation_id, image_id, category_id, bbox_json, mask_json, iscrowd)
            VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (
                    item["annotation_id"],
                    image["image_id"],
                    item["category_id"],
                    json.dumps(item["xyxy"]),
                    json.dumps(item["mask"], ensure_ascii=True),
                    item["iscrowd"],
                )
                for item in ground_truths
            ],
        )
        connection.executemany(
            """INSERT INTO predictions
            (image_id, mode, category_id, category_name, prompt, confidence,
             bbox_json, source_bbox_json, mask_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    image["image_id"],
                    item["mode"],
                    item["category_id"],
                    item["category_name"],
                    item["prompt"],
                    item["confidence"],
                    json.dumps(item["xyxy"]),
                    json.dumps(item["source_xyxy"]) if item["source_xyxy"] is not None else None,
                    json.dumps(item["mask"], ensure_ascii=True),
                )
                for item in predictions
            ],
        )
        connection.execute(
            "UPDATE run_info SET successful_image_count = successful_image_count + 1"
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def increment_skipped(connection: sqlite3.Connection) -> None:
    connection.execute(
        "UPDATE run_info SET skipped_image_count = skipped_image_count + 1"
    )
    connection.commit()
