#!/usr/bin/env python3
"""Migrate ODVG labels to object/phrase relations. Default: read-only audit.

python3 migrate_annotations.py
python3 migrate_annotations.py --output datasets/labels_v2
python3 migrate_annotations.py --apply

--apply validates all files before writing and backs up original bytes first.
The new schema requires a corresponding DataLoader update before training.
Color completeness is inherited from annotation semantics, not image inspection.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from collections import Counter
from datetime import datetime


VERSION = "lightdet.relations.v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def semantic(key):
    kind, _, value = str(key).partition(":")
    colors = frozenset(value.split("|")) if value else frozenset()
    if kind not in {"colors_exact", "single_color", "contains_color"}:
        return None, frozenset()
    require(bool(colors) and "" not in colors, "empty semantic color")
    if kind in {"single_color", "contains_color"}:
        require(len(colors) == 1, "single/contains color must have one color")
    return kind, colors


def relation(kind, colors, known, exact):
    if kind is None:
        return -1
    if kind == "contains_color":
        if colors <= known:
            return 1
        return 0 if exact is not None else -1
    if exact is not None:
        return int(exact == colors)
    return 0 if known - colors else -1


def validate_v2(record):
    objects = record["objects"]
    ids = [o["id"] for o in objects]
    require(len(ids) == len(set(ids)), "duplicate object IDs")
    all_ids = set(ids)
    phrases = record["grounding"]["phrases"]
    for p in phrases:
        groups = [p[k] for k in ("positive_object_ids", "negative_object_ids", "ignore_object_ids")]
        flat = sum(groups, [])
        require(len(flat) == len(set(flat)) and set(flat) == all_ids,
                "relations must partition all objects")
    return record


def convert(record):
    if record.get("schema_version") == VERSION:
        return validate_v2(record), False
    require("schema_version" not in record and "objects" not in record,
            "unrecognized schema; refusing to overwrite")
    width, height = record["width"], record["height"]
    require(width > 0 and height > 0, "invalid image size")
    require(bool(record.get("filename")), "missing filename")
    grounding = record["grounding"]
    require("phrases" not in grounding, "phrases already exists")
    caption, regions = grounding["caption"], grounding["regions"]
    require(isinstance(caption, str) and bool(regions), "missing caption/regions")
    boxes, index, positives, semantics = [], {}, [], []
    for r in regions:
        require(isinstance(r["phrase"], str) and bool(r["phrase"]), "empty phrase")
        spans = r["tokens_positive"]
        require(bool(spans), "empty character spans")
        for start, end in spans:
            require(isinstance(start, int) and isinstance(end, int)
                    and 0 <= start < end <= len(caption), "invalid character span")
        if len(spans) == 1:
            start, end = spans[0]
            require(caption[start:end] == r["phrase"], "phrase/span mismatch")
        raw_boxes = r["bbox"]
        require(isinstance(raw_boxes, list) and bool(raw_boxes), "empty bbox")
        if len(raw_boxes) == 4 and all(isinstance(x, (int, float)) for x in raw_boxes):
            raw_boxes = [raw_boxes]
        selected = set()
        for box in raw_boxes:
            require(len(box) == 4 and all(isinstance(x, (int, float))
                    and not isinstance(x, bool) and math.isfinite(x) for x in box), "invalid bbox")
            x1, y1, x2, y2 = box
            require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, "bbox out of bounds")
            # Exact coordinates only: do not merge nearby distinct boats by IoU.
            key = tuple(box)
            if key not in index:
                index[key] = len(boxes)
                boxes.append(list(box))
            selected.add(index[key])
        positives.append(selected)
        semantics.append(semantic(r.get("semantic_key", "")))

    known = [set() for _ in boxes]
    exact = [None for _ in boxes]
    for selected, (kind, colors) in zip(positives, semantics):
        for i in selected:
            if kind is None:
                continue
            known[i].update(colors)
            if kind in {"colors_exact", "single_color"}:
                require(exact[i] is None or exact[i] == colors,
                        f"conflicting exact colors for object {i}")
                exact[i] = colors
    for i in range(len(boxes)):
        require(exact[i] is None or known[i] <= exact[i],
                f"contains/exact color conflict for object {i}")

    result = copy.deepcopy(record)
    result["schema_version"] = VERSION
    result["objects"] = [dict(id=f"object_{i:04d}", bbox=b,
        attributes=dict(colors=sorted(known[i]), colors_complete=exact[i] is not None),
        attribute_source="legacy_semantic_key") for i, b in enumerate(boxes)]
    phrases = []
    for n, (r, selected, (kind, colors)) in enumerate(zip(regions, positives, semantics)):
        p = {k: copy.deepcopy(v) for k, v in r.items() if k != "bbox"}
        p["id"] = f"phrase_{n:04d}"
        for name in ("positive", "negative", "ignore"):
            p[f"{name}_object_ids"] = []
        for i, obj in enumerate(result["objects"]):
            inferred = relation(kind, colors, known[i], exact[i])
            require(not (i in selected and inferred == 0), "explicit/inferred relation conflict")
            value = 1 if i in selected else inferred
            name = {1: "positive", 0: "negative", -1: "ignore"}[value]
            p[f"{name}_object_ids"].append(obj["id"])
        phrases.append(p)
    result["grounding"].pop("regions")
    result["grounding"]["phrases"] = phrases
    result["migration"] = dict(source_schema="odvg_regions",
        object_identity="exact_bbox_coordinates", relation_policy="explicit_positive_then_color_logic",
        source_object_count=record.get("metadata", {}).get("object_count"))
    return validate_v2(result), True


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".migrate-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=Path(__file__).resolve().parent / "datasets/labels")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="replace labels after a complete backup")
    mode.add_argument("--output", type=Path, help="write converted labels to a new directory")
    args = parser.parse_args()
    root = args.labels.resolve()
    files = sorted(root.rglob("*.json"))
    require(bool(files), f"no JSON labels found: {root}")
    if args.output:
        output = args.output.resolve()
        require(not output.exists() and root not in output.parents and output not in root.parents,
                "output must be a new directory outside labels")
    plans, errors, counts = [], [], Counter()
    sample = None
    for path in files:
        try:
            require(not path.is_symlink() and root in path.resolve().parents, "symlink labels unsupported")
            raw = path.read_bytes()
            converted, changed = convert(json.loads(raw))
            plans.append((path, raw, converted, changed))
            counts["files"] += 1
            counts["to_convert" if changed else "already_converted"] += 1
            counts["objects"] += len(converted["objects"])
            for p in converted["grounding"]["phrases"]:
                counts["phrases"] += 1
                for name in ("positive", "negative", "ignore"):
                    counts[name + "_pairs"] += len(p[name + "_object_ids"])
            if sample is None:
                sample = converted
        except (ValueError, KeyError, TypeError, OSError) as error:
            errors.append(dict(file=str(path), error=str(error)))
    print(json.dumps(dict(mode="apply" if args.apply else "output" if args.output else "dry-run",
                         summary=dict(counts), error_count=len(errors), errors=errors[:30]), ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit("Validation failed. No labels written.")
    if not args.apply and not args.output:
        print("Preview (first record):")
        print(json.dumps(sample, ensure_ascii=False, indent=2))
        return
    backup = None
    if args.apply:
        if not any(p[3] for p in plans):
            print("All labels already converted; nothing changed.")
            return
        backup = Path(tempfile.mkdtemp(prefix="labels_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_", dir=root.parent))
        manifest = {}
        for path, raw, _, _ in plans:
            destination = backup / path.relative_to(root)
            atomic_write(destination, raw)
            require(destination.read_bytes() == raw, "backup verification failed")
            manifest[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        atomic_write(backup / "migration_manifest.json", json.dumps(manifest, indent=2).encode())
        print(f"Verified backup: {backup}", flush=True)
        for path, raw, _, _ in plans:
            require(path.read_bytes() == raw, f"source changed during audit: {path}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    for path, raw, converted, changed in plans:
        if args.apply and not changed:
            continue
        target = path if args.apply else output / path.relative_to(root)
        require(not args.apply or path.read_bytes() == raw, f"source changed: {path}")
        data = (json.dumps(converted, ensure_ascii=False, indent=2) + "\n").encode()
        atomic_write(target, data)
        require(target.read_bytes() == data, f"write verification failed: {target}")
    print("Completed. Update the DataLoader for lightdet.relations.v1 before training.")


if __name__ == "__main__":
    main()
