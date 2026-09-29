"""Strict object-ID relations and deterministic, epoch-dependent text sampling."""

import copy
import math
import random


def normalize_record(raw):
    if raw.get("schema_version") != "lightdet.relations.v1":
        raise ValueError("Run migrate_annotations.py --apply before training")
    objects = raw["objects"]
    ids = [o["id"] for o in objects]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate object IDs")
    for o in objects:
        x1, y1, x2, y2 = o["bbox"]
        if not (all(math.isfinite(v) for v in o["bbox"])
                and 0 <= x1 < x2 <= raw["width"] and 0 <= y1 < y2 <= raw["height"]):
            raise ValueError("Invalid object bbox")
    phrases = copy.deepcopy(raw["grounding"]["phrases"])
    if len({p["id"] for p in phrases}) != len(phrases):
        raise ValueError("Duplicate phrase IDs")
    caption = raw["grounding"]["caption"]
    for p in phrases:
        flat = sum([p[k] for k in ("positive_object_ids", "negative_object_ids", "ignore_object_ids")], [])
        if len(flat) != len(set(flat)) or set(flat) != set(ids):
            raise ValueError("Relations must partition every object")
        if len(p["tokens_positive"]) != 1:
            raise ValueError("Relation captions require one contiguous span per phrase")
        a, b = p["tokens_positive"][0]
        if not 0 <= a < b <= len(caption) or caption[a:b] != p["phrase"]:
            raise ValueError("Phrase/character-span mismatch")
    result = dict(filename=raw["filename"], width=raw["width"], height=raw["height"],
                  caption=caption, objects=copy.deepcopy(objects), regions=phrases,
                  metadata=copy.deepcopy(raw.get("metadata", {})))
    return rebuild_targets(result)


def rebuild_targets(record):
    objects, phrases = record["objects"], record["regions"]
    index = {o["id"]: i for i, o in enumerate(objects)}
    targets = [dict(id=o["id"], bbox=o["bbox"], positive_char_spans=[],
                    phrases=[], semantic_keys=[], region_indices=[]) for o in objects]
    relations, mapping = [], []
    for j, p in enumerate(phrases):
        row = [-1] * len(objects)
        for name, value in (("positive_object_ids", 1), ("negative_object_ids", 0)):
            for oid in p[name]:
                row[index[oid]] = value
        indices = [index[oid] for oid in p["positive_object_ids"]]
        p["region_index"] = j
        # Runtime evaluation consumes region boxes; annotation identity stays ID-based.
        p["bbox"] = [objects[i]["bbox"] for i in indices]
        for i in indices:
            targets[i]["positive_char_spans"].extend(p["tokens_positive"])
            targets[i]["phrases"].append(p["phrase"])
            targets[i]["semantic_keys"].append(p["semantic_key"])
            targets[i]["region_indices"].append(j)
        relations.append(row)
        mapping.append(indices)
    if any(not t["positive_char_spans"] for t in targets):
        raise ValueError("Every localization object must retain a positive description")
    record.update(unique_targets=targets, region_to_target_indices=mapping,
                  phrase_relations=relations)
    return record


def sample_record(record, seed, epoch, index, training):
    if not training:
        return record
    sampled = copy.deepcopy(record)
    # Four reusable caption views limit frozen-text cache churn.
    view = int(epoch) % 4
    rng = random.Random(seed + index * 1_000_003 + view * 97_409)
    variants = record["metadata"].get("phrase_variants", {})
    for p in sampled["regions"]:
        kind, _, color = p["semantic_key"].partition(":")
        allowed = {p["phrase"]}
        if kind == "single_color":
            allowed.update((f"純{color}的船", f"船體僅為{color}的船", f"只有{color}的船"))
        elif kind == "contains_color":
            allowed.update((f"帶有{color}的船", f"船體包含{color}的船", f"有{color}部分的船"))
        elif kind == "colors_exact":
            allowed.update(("一艘" + p["phrase"], p["phrase"].replace("相間", "多色")))
        candidates = [s for s in variants.get(p["phrase"], [])
                      if isinstance(s, str) and s.replace("、", "") in allowed]
        choices = list(dict.fromkeys([p["phrase"]] + candidates))
        p["phrase"] = choices[(view + index) % len(choices)]
    rng.shuffle(sampled["regions"])
    parts, cursor = [], 0
    for p in sampled["regions"]:
        text = p["phrase"]
        p["tokens_positive"] = [[cursor, cursor + len(text)]]
        parts.append(text + "。")
        cursor += len(text) + 1
    sampled["caption"] = "".join(parts)
    return rebuild_targets(sampled)
