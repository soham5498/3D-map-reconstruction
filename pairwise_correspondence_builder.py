#!/usr/bin/env python3
"""Pipeline A, version 3: automatic frame discovery and reviewed object-ID merges.

Python >= 3.10, macOS/Linux. Dependencies: matplotlib, Pillow.
Run from your project directory:
    python3 pairwise_correspondence_builder.py
Use --help for direct frame selection, migration, and root overrides.
Nothing is committed until you explicitly type SAVE after reviewing a pair.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import signal
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# ======================== CONFIGURATION ========================
# Relative paths are resolved from your terminal's working directory.
LABEL_ROOT = Path("./dataset")
IMAGE_ROOT = Path("../../data_folders")
OUTPUT_DIR = Path("./DHBW_correspondences")
# Automatically imported on first use if present. Set None to disable.
LEGACY_DIR = Path("./dhbw_correspondences")
BUILDING = "DHBW"
CLASS_NAMES = {
    0: "glass", 1: "window", 2: "frame", 3: "casement",
    4: "window sill", 5: "roof", 6: "darkening", 7: "door",
}
CORNER_ORDER = ("TL", "TR", "BR", "BL")  # semantic order in your YOLO export
ONLY_FULLY_VISIBLE = True  # --all-visibility also permits v=0/1 for identity matching
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
THUMBNAILS_PER_PAGE = 9
# ===============================================================

FRAME_FIELDS = ("frame_id", "folder_id", "image_path", "label_path")
OBJECT_FIELDS = ("object_id", "building", "class_id", "class_name")
WIDE_OBJECT_FIELDS = OBJECT_FIELDS + ("merged_from_ids",)
OBS_FIELDS = ("frame_id", "local_object_id", "object_id", "corner", "x_px", "y_px", "visibility")
PAIR_FIELDS = ("frame_a", "frame_b", "status")
FRAME_SUFFIXES = ("local_object_id",) + tuple(f"{c}_{axis}" for c in CORNER_ORDER for axis in ("x", "y", "v"))
CSV_FILES = ("frames.csv", "correspondences.csv", "pairs.csv")
LEGACY_FILES = ("frames.csv", "objects.csv", "observations.csv")
TXN_NAME = ".building_correspondence_transaction"
LOCK_NAME = ".pairwise_builder.lock"  # also coordinates with the version-1 builder
ALLOWED_TARGETS = (set(CSV_FILES) | {f"legacy_backup/{name}" for name in LEGACY_FILES}
                   | {f"v2_backup/{name}" for name in CSV_FILES})
PAIR_STATUSES = {"partial", "reviewed", "no_overlap"}

class DataError(ValueError):
    """A source, identity, or CSV consistency problem; do not silently repair."""


class MergeRequired(DataError):
    """Both selected annotations have IDs; review their full tracks first."""


def integer(value, name):
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise DataError(f"{name} must be an integer, got {value!r}.")
    return int(number)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_text(value):
    return "" if value is None else f"{value:.9f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class Annotation:
    local_id: int
    class_id: int
    bbox: tuple
    keypoints: tuple  # (x_px or None, y_px or None, visibility), in CORNER_ORDER

    @property
    def fully_visible(self):
        return all(kp[2] == 2 for kp in self.keypoints)


def parse_labels(text, width, height):
    """Parse class + normalized box + four (normalized x, y, visibility)."""
    annotations = {}
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        if not raw_line.strip():
            continue
        parts = raw_line.split()
        if len(parts) != 17:
            raise DataError(f"Label line {line_number}: expected 17 values "
                            "(class, box, four x/y/v triplets).")
        values = [float(x) for x in parts]
        if not all(math.isfinite(x) for x in values):
            raise DataError(f"Non-finite value at label line {line_number}.")
        class_id = integer(values[0], "class ID")
        if class_id not in CLASS_NAMES:
            raise DataError(f"Unknown class {class_id}; add it to CLASS_NAMES.")
        cx, cy, w, h = values[1:5]
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise DataError(f"Label line {line_number}: invalid normalized box.")
        kps = []
        for offset, corner in zip(range(5, 17, 3), CORNER_ORDER):
            x, y, v_raw = values[offset:offset + 3]
            v = integer(v_raw, "visibility")
            if v not in (0, 1, 2):
                raise DataError(f"Label line {line_number}: {corner} has invalid v={v}.")
            if v > 0 and not (0 <= x <= 1 and 0 <= y <= 1):
                raise DataError(f"Label line {line_number}: {corner} is outside [0, 1].")
            # v=0 has no usable position: do not write the YOLO 0,0 placeholder.
            kps.append((x * width, y * height, v) if v else (None, None, 0))
        local_id = len(annotations) + 1  # 1-based NONBLANK label row, not class ID
        annotations[local_id] = Annotation(
            local_id, class_id,
            ((cx - w / 2) * width, (cy - h / 2) * height,
             (cx + w / 2) * width, (cy + h / 2) * height), tuple(kps),
        )
    if not annotations:
        raise DataError("The label file contains no objects.")
    return annotations


@dataclass
class Frame:
    frame_id: str
    image_path: Path
    label_path: Path
    image: object
    annotations: dict
    source_hashes: dict

    @classmethod
    def load(cls, spec):
        from PIL import Image
        frame_id, image_path, label_path = spec
        if not re.fullmatch(r"[A-Za-z0-9_-]+", frame_id):
            raise DataError("Frame IDs may contain letters, digits, underscores and hyphens.")
        image_path, label_path = Path(image_path).resolve(), Path(label_path).resolve()
        image_bytes, label_bytes = image_path.read_bytes(), label_path.read_bytes()
        with Image.open(io.BytesIO(image_bytes)) as raw:
            # No resizing, undistortion, EXIF rotation, or keypoint reordering.
            image = raw.convert("RGB")
        annotations = parse_labels(label_bytes.decode("utf-8-sig"), *image.size)
        hashes = {image_path: hashlib.sha256(image_bytes).hexdigest(),
                  label_path: hashlib.sha256(label_bytes).hexdigest()}
        return cls(frame_id, image_path, label_path, image, annotations, hashes)

    def check_unchanged(self):
        for path, old_hash in self.source_hashes.items():
            if digest_file(path) != old_hash:
                raise DataError(f"Source changed during selection: {path}. "
                                "Nothing was saved. Restart with the intended files.")


def validate_tracks(tables):
    """Enforce references, four named corners, and one-to-one object identities."""
    frames, objects, assignments, reverse, groups = {}, {}, {}, {}, {}
    for row in tables["frames.csv"]:
        fid = row["frame_id"]
        if not fid or fid in frames or not row["image_path"] or not row["label_path"]:
            raise DataError(f"Duplicate or incomplete frame: {fid!r}.")
        frames[fid] = row
    for row in tables["objects.csv"]:
        oid = row["object_id"]
        if not oid or oid in objects or not row["building"] or not row["class_name"]:
            raise DataError(f"Duplicate or incomplete object: {oid!r}.")
        if integer(row["class_id"], "class ID") < 0:
            raise DataError("Negative class ID in objects.csv.")
        objects[oid] = row
    for row in tables["observations.csv"]:
        fid, oid, corner = row["frame_id"], row["object_id"], row["corner"]
        lid = integer(row["local_object_id"], "local_object_id")
        if fid not in frames or oid not in objects or lid < 1 or corner not in CORNER_ORDER:
            raise DataError(f"Invalid observation reference: {row}.")
        local_key, object_key = (fid, lid), (fid, oid)
        if local_key in assignments and assignments[local_key] != oid:
            raise DataError(f"Annotation {local_key} has two global IDs.")
        if object_key in reverse and reverse[object_key] != lid:
            raise DataError(f"Object {oid} is assigned to two annotations in {fid}.")
        assignments[local_key], reverse[object_key] = oid, lid
        group = groups.setdefault(local_key, {})
        if corner in group:
            raise DataError(f"Duplicate observation: {fid}, {oid}, {corner}.")
        group[corner] = row
        v = integer(row["visibility"], "visibility")
        if v not in (0, 1, 2):
            raise DataError(f"Invalid visibility {v}.")
        if v == 0:
            if row["x_px"] != "" or row["y_px"] != "":
                raise DataError("v=0 coordinates must be blank in observations.csv.")
        elif not all(math.isfinite(float(row[k])) and float(row[k]) >= 0
                     for k in ("x_px", "y_px")):
            raise DataError("Non-finite or negative pixel coordinates.")
    for key, group in groups.items():
        if set(group) != set(CORNER_ORDER):
            raise DataError(f"Incomplete corner record for {key}; expected TL/TR/BR/BL.")
    return assignments, reverse


def natural_key(text):
    return tuple((1, int(x)) if x.isdigit() else (0, x.lower())
                 for x in re.split(r"(\d+)", str(text)))


def merged_ids(row):
    """Retired IDs are stored as a pipe-separated, flat list on the surviving row."""
    raw = row.get("merged_from_ids", "")
    ids = raw.split("|") if raw else []
    if (len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", oid) for oid in ids)):
        raise DataError(f"Invalid merged_from_ids for {row['object_id']}: {raw!r}.")
    return ids


def alias_map(tables):
    active = {row["object_id"] for row in tables["objects.csv"]}
    aliases = {}
    for row in tables["objects.csv"]:
        for old_id in merged_ids(row):
            if old_id in active or old_id in aliases:
                raise DataError(f"Retired ID {old_id} is active or assigned to more than one object.")
            aliases[old_id] = row["object_id"]
    return aliases


def resolve_object_id(tables, object_id):
    if any(row["object_id"] == object_id for row in tables["objects.csv"]):
        return object_id
    resolved = alias_map(tables).get(object_id)
    if resolved is None:
        raise DataError(f"Unknown object ID: {object_id}.")
    return resolved


def empty_tables():
    return {"frames.csv": [], "objects.csv": [], "observations.csv": [], "pairs.csv": []}


def canonical_pair(a, b):
    if a == b:
        raise DataError("Select two different frames.")
    return tuple(sorted((a, b)))


def validate_tables(tables):
    bindings, reverse = validate_tracks(tables)
    alias_map(tables)
    frames = {r["frame_id"]: r for r in tables["frames.csv"]}
    for row in frames.values():
        if not row.get("folder_id"):
            raise DataError(f"Missing folder_id for {row['frame_id']}.")
    seen = set()
    for row in tables["pairs.csv"]:
        a, b = row["frame_a"], row["frame_b"]
        pair = canonical_pair(a, b)
        if a not in frames or b not in frames or frames[a]["folder_id"] == frames[b]["folder_id"]:
            raise DataError(f"Pair {a}, {b} must refer to two registered, different folders.")
        if pair != (a, b) or pair in seen or row["status"] not in PAIR_STATUSES:
            raise DataError(f"Invalid, reversed, or duplicate pair record: {row}.")
        seen.add(pair)
        if row["status"] == "no_overlap" and any(
                fid == a and (b, oid) in reverse for (fid, _), oid in bindings.items()):
            raise DataError(f"{a} and {b} already share an object; no_overlap would conflict with it.")
    return bindings, reverse


@dataclass(frozen=True)
class FrameSpec:
    frame_id: str
    folder_id: str
    image_path: Path
    label_path: Path
    object_count: int
    visible_count: int
    label_hash: str

    def load(self):
        frame = Frame.load((self.frame_id, self.image_path, self.label_path))
        if frame.source_hashes[self.label_path] != self.label_hash:
            raise DataError(f"Labels changed since scanning {self.frame_id}. Refresh the inventory.")
        return frame


def discover_frames(label_root, image_root, known_frames=(), csv_directory=None):
    """Labels are authoritative; scan only direct folder/*.txt under label_root."""
    from PIL import Image
    label_root, image_root = Path(label_root).resolve(), Path(image_root).resolve()
    if not label_root.is_dir():
        raise DataError(f"Label root does not exist: {label_root}")
    base = Path(csv_directory or Path.cwd())
    by_source = {(base / r["label_path"]).resolve(): r for r in known_frames}
    catalogue, messages = {}, []
    for folder in sorted(label_root.iterdir(), key=lambda p: natural_key(p.name)):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        image_folder = image_root / folder.name
        images = {}
        if image_folder.is_dir():
            for path in image_folder.iterdir():
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    images.setdefault(path.stem, []).append(path.resolve())
        for label in sorted(folder.iterdir(), key=lambda p: natural_key(p.name)):
            if not label.is_file() or label.suffix.lower() != ".txt":
                continue
            matches = images.get(label.stem, [])
            if len(matches) != 1:
                reason = "missing image" if not matches else "ambiguous image extensions"
                messages.append(f"Excluded {folder.name}/{label.name}: {reason}.")
                continue
            label = label.resolve()
            try:
                data = label.read_bytes()
                with Image.open(matches[0]) as image:
                    annotations = parse_labels(data.decode("utf-8-sig"), *image.size)
            except (OSError, ValueError) as exc:
                messages.append(f"Excluded {folder.name}/{label.name}: {exc}")
                continue
            saved = by_source.get(label)
            fid = saved["frame_id"] if saved else f"{folder.name}_{label.stem}"
            if saved and ((base / saved["image_path"]).resolve() != matches[0]
                          or saved["folder_id"] != folder.name):
                raise DataError(f"Saved frame {fid} now refers to different image/folder paths.")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", fid) or fid in catalogue:
                raise DataError(f"Invalid or colliding frame ID: {fid}. Use unique folder/file names.")
            catalogue[fid] = FrameSpec(fid, folder.name, matches[0], label, len(annotations),
                                      sum(a.fully_visible for a in annotations.values()),
                                      hashlib.sha256(data).hexdigest())
    return catalogue, messages


def read_csv_bytes(data, expected=None, name="CSV"):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    fields = tuple(reader.fieldnames or ())
    if len(fields) != len(set(fields)) or not fields or (expected is not None and fields != tuple(expected)):
        raise DataError(f"Unexpected or duplicate columns in {name}: {fields}.")
    rows = list(reader)
    if any(set(row) != set(fields) or any(v is None for v in row.values()) for row in rows):
        raise DataError(f"Malformed row in {name}.")
    return fields, rows


def encode_csv(fields, rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def wide_columns(frames, include_merged_ids=True):
    # Existing frame order is retained; discovered frames are appended.
    metadata = WIDE_OBJECT_FIELDS if include_merged_ids else OBJECT_FIELDS
    return metadata + tuple(f"{r['frame_id']}_{s}" for r in frames for s in FRAME_SUFFIXES)


def serialize_project(tables):
    validate_tables(tables)
    objects = {r["object_id"]: {key: r.get(key, "") for key in WIDE_OBJECT_FIELDS}
               for r in tables["objects.csv"]}
    for observation in tables["observations.csv"]:
        row = objects[observation["object_id"]]
        prefix = observation["frame_id"] + "_"
        row[prefix + "local_object_id"] = observation["local_object_id"]
        corner = observation["corner"]
        for axis, source in (("x", "x_px"), ("y", "y_px"), ("v", "visibility")):
            row[prefix + corner + "_" + axis] = observation[source]
    return {
        "frames.csv": encode_csv(FRAME_FIELDS, tables["frames.csv"]),
        "correspondences.csv": encode_csv(wide_columns(tables["frames.csv"]),
                                          (objects[o] for o in sorted(objects, key=natural_key))),
        "pairs.csv": encode_csv(PAIR_FIELDS, sorted(tables["pairs.csv"],
                                                  key=lambda r: (r["frame_a"], r["frame_b"]))),
    }


def deserialize_project(raw):
    tables = empty_tables()
    _, tables["frames.csv"] = read_csv_bytes(raw["frames.csv"], FRAME_FIELDS, "frames.csv")
    fields, wide_rows = read_csv_bytes(raw["correspondences.csv"], name="correspondences.csv")
    if fields not in (wide_columns(tables["frames.csv"]), wide_columns(tables["frames.csv"], False)):
        raise DataError("Unexpected columns in correspondences.csv; expected the version-2 or version-3 frame blocks.")
    _, tables["pairs.csv"] = read_csv_bytes(raw["pairs.csv"], PAIR_FIELDS, "pairs.csv")
    for row in wide_rows:
        tables["objects.csv"].append({key: row.get(key, "") for key in WIDE_OBJECT_FIELDS})
        for frame in tables["frames.csv"]:
            fid, prefix = frame["frame_id"], frame["frame_id"] + "_"
            lid = row[prefix + "local_object_id"]
            if not lid:
                if any(row[prefix + suffix] for suffix in FRAME_SUFFIXES):
                    raise DataError(f"{row['object_id']}, {fid}: coordinates without a local object ID.")
                continue  # An entirely blank frame block is a missing observation.
            for corner in CORNER_ORDER:
                tables["observations.csv"].append({
                    "frame_id": fid, "local_object_id": lid, "object_id": row["object_id"],
                    "corner": corner, "x_px": row[prefix + corner + "_x"],
                    "y_px": row[prefix + corner + "_y"], "visibility": row[prefix + corner + "_v"],
                })
    validate_tables(tables)
    return tables


def fsync_dir(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_write(path, data):
    with Path(path).open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def replace_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        fsync_dir(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def store_lock(directory):
    import fcntl
    with (directory / LOCK_NAME).open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DataError("Another builder is saving this project; retry shortly.") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def recover_locked(directory):
    txn = directory / TXN_NAME
    if not txn.exists():
        return
    path = txn / "manifest.json"
    if not path.exists():
        shutil.rmtree(txn)  # Preparation failed before any target could be changed.
        return
    manifest = json.loads(path.read_text())
    entries = manifest.get("entries", [])
    if (not entries or len({e.get('name') for e in entries}) != len(entries)
            or any(e.get("name") not in ALLOWED_TARGETS for e in entries)):
        raise DataError(f"Invalid recovery journal: {txn}. Preserve it for recovery.")
    if manifest.get("state") == "prepared":
        backups = {}
        for i, entry in enumerate(entries):
            if entry["old_hash"] is not None:
                data = (txn / f"old_{i}").read_bytes()
                if hashlib.sha256(data).hexdigest() != entry["old_hash"]:
                    raise DataError("Recovery backup checksum mismatch; preserve recovery files.")
                backups[i] = data
        for i, entry in enumerate(entries):
            target = directory / entry["name"]
            if i in backups:
                replace_bytes(target, backups[i])
            else:
                target.unlink(missing_ok=True)
                if target.parent.exists():
                    fsync_dir(target.parent)
        print("Recovered an interrupted save: restored the previous complete CSV set.")
    elif manifest.get("state") != "committed":
        raise DataError(f"Unknown recovery state in {txn}.")
    shutil.rmtree(txn)
    fsync_dir(directory)


def atomic_commit(directory, payload, expected, guards=()):
    """Explicit-save transaction with per-file replacement and crash recovery."""
    if not set(payload).issubset(ALLOWED_TARGETS):
        raise DataError("Unexpected save target.")
    directory.mkdir(parents=True, exist_ok=True)
    with store_lock(directory):
        recover_locked(directory)
        for path, old in list((directory / n, data) for n, data in expected.items()) + list(guards):
            actual = path.read_bytes() if path.exists() else None
            if actual != old:
                raise DataError(f"File changed during this session: {path}. Reload; no session data was saved.")
        txn = directory / TXN_NAME
        txn.mkdir()
        try:
            entries = []
            for i, (name, data) in enumerate(payload.items()):
                target = directory / name
                old = target.read_bytes() if target.exists() else None
                entries.append({"name": name, "old_hash": hashlib.sha256(old).hexdigest() if old is not None else None})
                if old is not None:
                    durable_write(txn / f"old_{i}", old)
                durable_write(txn / f"new_{i}", data)
            manifest = {"state": "prepared", "entries": entries}
            replace_bytes(txn / "manifest.json", json.dumps(manifest).encode())
            fsync_dir(directory)
            for i, entry in enumerate(entries):
                target = directory / entry["name"]
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(txn / f"new_{i}", target)
                fsync_dir(target.parent)
            manifest["state"] = "committed"
            replace_bytes(txn / "manifest.json", json.dumps(manifest).encode())
        except BaseException:
            recover_locked(directory)
            raise
        recover_locked(directory)  # The commit marker is durable; clean backups.


def snapshot(directory, names):
    return {name: (directory / name).read_bytes() if (directory / name).exists() else None for name in names}


class Store:
    def __init__(self, directory, tables, original, legacy=None, guards=(), upgrade=None):
        self.directory, self.tables, self.original = directory, tables, original
        self.legacy, self.guards = legacy, list(guards)
        self.upgrade = upgrade

    @classmethod
    def load(cls, directory, legacy_directory=None):
        directory = Path(directory).resolve()
        if directory.exists():
            with store_lock(directory):
                recover_locked(directory)
                if (directory / ".pairwise_transaction").exists():
                    raise DataError("An interrupted version-1 save exists. Reopen the old builder once to recover it.")
                original = snapshot(directory, CSV_FILES)
        else:
            original = {n: None for n in CSV_FILES}
        if original["correspondences.csv"] is not None:
            if any(value is None for value in original.values()):
                raise DataError("Incomplete wide CSV project. Keep frames.csv, correspondences.csv and pairs.csv together.")
            tables = deserialize_project(original)
            header = next(csv.reader(io.StringIO(original["correspondences.csv"].decode("utf-8-sig"))))
            upgrade = original if "merged_from_ids" not in header else None
            return cls(directory, tables, original, upgrade=upgrade)
        if original["pairs.csv"] is not None:
            raise DataError("pairs.csv exists without correspondences.csv. No automatic repair was attempted.")
        source = directory if all((directory / n).is_file() for n in LEGACY_FILES) else (
            Path(legacy_directory).resolve() if legacy_directory is not None else None)
        if source is not None and all((source / n).is_file() for n in LEGACY_FILES):
            if (source / ".pairwise_transaction").exists():
                raise DataError("Recover the old builder's interrupted save before importing its CSVs.")
            with store_lock(source):
                raw = snapshot(source, LEGACY_FILES)
            fields, frames = read_csv_bytes(raw["frames.csv"], name="legacy frames.csv")
            if fields not in (("frame_id", "image_path", "label_path"), FRAME_FIELDS):
                raise DataError("Legacy frames.csv is not the earlier pairwise builder format.")
            tables = empty_tables()
            for row in frames:
                ip, lp = (source / row["image_path"]).resolve(), (source / row["label_path"]).resolve()
                tables["frames.csv"].append({"frame_id": row["frame_id"],
                    "folder_id": row.get("folder_id") or lp.parent.name,
                    "image_path": os.path.relpath(ip, directory), "label_path": os.path.relpath(lp, directory)})
            _, tables["objects.csv"] = read_csv_bytes(raw["objects.csv"], OBJECT_FIELDS, "legacy objects.csv")
            _, tables["observations.csv"] = read_csv_bytes(raw["observations.csv"], OBS_FIELDS, "legacy observations.csv")
            validate_tables(tables)
            if original["frames.csv"] is not None and source != directory:
                raise DataError("Output contains an unrelated frames.csv; choose an empty output directory.")
            print(f"Loaded existing tracks from {source}. Conversion remains in memory until SAVE.")
            return cls(directory, tables, original, legacy=raw, guards=[(source/n, data) for n, data in raw.items()])
        if any(value is not None for value in original.values()):
            raise DataError("Incomplete existing project; no files were overwritten.")
        if legacy_directory is not None:
            raise DataError(f"Legacy directory lacks the three required CSVs: {legacy_directory}")
        return cls(directory, empty_tables(), original)

    def verify_frame(self, frame, building):
        for row in self.tables["frames.csv"]:
            ip = (self.directory / row["image_path"]).resolve()
            lp = (self.directory / row["label_path"]).resolve()
            if row["frame_id"] == frame.frame_id and (ip != frame.image_path or lp != frame.label_path):
                raise DataError(f"Frame {frame.frame_id} is already assigned to different source paths.")
            if row["frame_id"] != frame.frame_id and (ip == frame.image_path or lp == frame.label_path):
                raise DataError(f"These sources already have frame ID {row['frame_id']}; reuse that ID.")
        objects = {r["object_id"]: r for r in self.tables["objects.csv"]}
        for row in self.tables["observations.csv"]:
            if row["frame_id"] != frame.frame_id:
                continue
            lid = integer(row["local_object_id"], "local object ID")
            a, obj = frame.annotations.get(lid), objects[row["object_id"]]
            if a is None or a.class_id != integer(obj["class_id"], "class ID") or obj["building"] != building:
                raise DataError(f"Saved identity changed for {frame.frame_id}:{lid}; reconcile the source labels.")
            x, y, v = a.keypoints[CORNER_ORDER.index(row["corner"])]
            if v != integer(row["visibility"], "visibility") or (
                    v and (abs(x-float(row["x_px"])) > 1e-6 or abs(y-float(row["y_px"])) > 1e-6)):
                raise DataError(f"Saved corners changed for {frame.frame_id}:{lid}. Restore/reconcile the original labels.")

    def commit(self, tables):
        payload = serialize_project(tables)
        for folder, source in (("legacy_backup", self.legacy), ("v2_backup", self.upgrade)):
            for name, data in (source or {}).items():
                target = self.directory / folder / name
                if target.exists() and target.read_bytes() != data:
                    raise DataError(f"A different migration backup already exists: {target}")
                payload[f"{folder}/{name}"] = data
        if self.legacy is None and self.upgrade is None and all(payload[n] == self.original[n] for n in CSV_FILES):
            return False
        atomic_commit(self.directory, payload, self.original, self.guards)
        self.tables, self.original = copy.deepcopy(tables), {n: payload[n] for n in CSV_FILES}
        self.legacy, self.guards = None, []
        self.upgrade = None
        return True


def register_catalogue(tables, catalogue, directory):
    existing = {r["frame_id"]: r for r in tables["frames.csv"]}
    paths = {(directory / r["label_path"]).resolve(): r["frame_id"] for r in existing.values()}
    for spec in catalogue.values():
        record = {"frame_id": spec.frame_id, "folder_id": spec.folder_id,
                  "image_path": os.path.relpath(spec.image_path, directory),
                  "label_path": os.path.relpath(spec.label_path, directory)}
        if spec.frame_id in existing:
            old = existing[spec.frame_id]
            if old["folder_id"] != spec.folder_id or any(
                (directory / old[key]).resolve() != (directory / record[key]).resolve()
                for key in ("image_path", "label_path")):
                raise DataError(f"Conflicting sources for frame {spec.frame_id}.")
        elif spec.label_path in paths:
            raise DataError(f"This label is already registered as {paths[spec.label_path]}.")
        else:
            tables["frames.csv"].append(record)
            existing[spec.frame_id], paths[spec.label_path] = record, spec.frame_id


@dataclass(frozen=True)
class MergePlan:
    keep_id: str
    retire_id: str
    left_local_id: int
    right_local_id: int
    views: tuple  # one saved observation block per (frame, object)
    conflicts: tuple  # (frame_id, local ID in keep track, local ID in retire track)
    retired_ids: tuple
    fingerprint: str

    @property
    def can_merge(self):
        return not self.conflicts


class MatchSession:
    """One pair's staged changes. Closing/cancelling never invokes Store.commit."""
    def __init__(self, store, left, right, building=BUILDING, visible_only=ONLY_FULLY_VISIBLE, catalogue=None):
        if left.frame_id == right.frame_id or left.image_path == right.image_path:
            raise DataError("Choose two different frames.")
        if left.label_path.parent.name == right.label_path.parent.name:
            raise DataError("Matching frames from the same folder is disabled.")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", building):
            raise DataError("Building ID must use letters, digits, underscores or hyphens.")
        if any(o["building"] != building for o in store.tables["objects.csv"]):
            raise DataError("This output project already contains another building. Choose its correct building ID.")
        self.store, self.left, self.right = store, left, right
        self.building, self.visible_only = building, visible_only
        self.tables, self.history, self.matches = copy.deepcopy(store.tables), [], []
        self.merges = []
        for frame in (left, right):
            store.verify_frame(frame, building)
        if catalogue is None:
            catalogue = {f.frame_id: FrameSpec(f.frame_id, f.label_path.parent.name,
                f.image_path, f.label_path, len(f.annotations), sum(a.fully_visible for a in f.annotations.values()),
                f.source_hashes[f.label_path]) for f in (left, right)}
        if left.frame_id not in catalogue or right.frame_id not in catalogue:
            raise DataError("Both selected frames must be present in the dataset catalogue.")
        self.catalogue = catalogue
        register_catalogue(self.tables, catalogue, store.directory)
        validate_tables(self.tables)

    def bindings(self):
        return validate_tables(self.tables)[0]

    def next_object_id(self):
        prefix = f"{self.building}_OBJ_"
        used = {r["object_id"] for r in self.tables["objects.csv"]}
        used.update(alias_map(self.tables))  # Retired numbers must never be reused.
        numbers = [int(oid[len(prefix):]) for oid in used
                   if oid.startswith(prefix) and oid[len(prefix):].isdigit()]
        return f"{prefix}{max(numbers, default=0) + 1:03d}"

    def eligible(self, annotation):
        return not self.visible_only or annotation.fully_visible

    def selected_annotations(self, left_id, right_id):
        if left_id not in self.left.annotations or right_id not in self.right.annotations:
            raise DataError("Selected local annotation is not present in its frame.")
        a, b = self.left.annotations[left_id], self.right.annotations[right_id]
        if a.class_id != b.class_id:
            raise DataError("Classes differ. Select the same physical object and class.")
        if not self.eligible(a) or not self.eligible(b):
            raise DataError("Visible-only mode requires four v=2 corners in both frames.")
        return a, b

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.tables, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def checkpoint(self):
        return copy.deepcopy(self.tables), copy.deepcopy(self.merges)

    def refresh_pair_statuses(self):
        pair = canonical_pair(self.left.frame_id, self.right.frame_id)
        assigned, inverse = validate_tracks(self.tables)
        for record in self.tables["pairs.csv"]:
            current = (record["frame_a"], record["frame_b"])
            if current == pair or (record["status"] == "no_overlap" and any(
                    fid == record["frame_a"] and (record["frame_b"], oid) in inverse
                    for (fid, _), oid in assigned.items())):
                record["status"] = "partial"

    def plan_merge(self, left_id, right_id):
        """Review the complete two tracks without changing their assignments."""
        self.selected_annotations(left_id, right_id)
        bindings, reverse = validate_tables(self.tables)
        ga, gb = bindings.get((self.left.frame_id, left_id)), bindings.get((self.right.frame_id, right_id))
        if not ga or not gb or ga == gb:
            raise DataError("These annotations do not need an ID merge. Use Confirm match.")
        keep, retire = sorted((ga, gb), key=natural_key)
        objects = {r["object_id"]: r for r in self.tables["objects.csv"]}
        if (objects[keep]["building"] != objects[retire]["building"] or
                integer(objects[keep]["class_id"], "class ID") != integer(objects[retire]["class_id"], "class ID")):
            raise DataError("Object records have different buildings or classes; reconcile them before merging.")
        frame_rows = {r["frame_id"]: r for r in self.tables["frames.csv"]}
        by_view = {}
        for row in self.tables["observations.csv"]:
            if row["object_id"] in (keep, retire):
                key = (row["frame_id"], row["object_id"])
                by_view.setdefault(key, {})[row["corner"]] = dict(row)
        views = []
        for (fid, oid), corners in sorted(by_view.items(), key=lambda item: (natural_key(item[0][0]), natural_key(item[0][1]))):
            views.append({"frame_id": fid, "object_id": oid,
                          "local_object_id": reverse[(fid, oid)],
                          "image_path": str((self.store.directory/frame_rows[fid]["image_path"]).resolve()),
                          "corners": tuple(corners[c] for c in CORNER_ORDER)})
        conflicts = tuple((fid, reverse[(fid, keep)], reverse[(fid, retire)])
                          for fid in sorted(frame_rows, key=natural_key)
                          if (fid, keep) in reverse and (fid, retire) in reverse
                          and reverse[(fid, keep)] != reverse[(fid, retire)])
        retired = tuple(sorted(set(merged_ids(objects[keep]) + merged_ids(objects[retire]) + [retire]), key=natural_key))
        return MergePlan(keep, retire, left_id, right_id, tuple(views), conflicts, retired, self.fingerprint())

    def merge_match(self, left_id, right_id, reviewed_plan):
        """Stage an explicitly reviewed merge; Undo and final SAVE still apply."""
        plan = self.plan_merge(left_id, right_id)
        if not isinstance(reviewed_plan, MergePlan) or (
                reviewed_plan.fingerprint, reviewed_plan.left_local_id, reviewed_plan.right_local_id,
                reviewed_plan.keep_id, reviewed_plan.retire_id) != (
                plan.fingerprint, plan.left_local_id, plan.right_local_id, plan.keep_id, plan.retire_id):
            raise DataError("The merge review is stale. Reopen it before staging this change.")
        if plan.conflicts:
            details = "; ".join(f"{fid}: local {a} vs {b}" for fid, a, b in plan.conflicts)
            raise DataError("Merge blocked: different annotations in the same frame. " + details)
        previous = self.checkpoint()
        try:
            kept_row = next(r for r in self.tables["objects.csv"] if r["object_id"] == plan.keep_id)
            kept_row["merged_from_ids"] = "|".join(plan.retired_ids)
            self.tables["objects.csv"] = [r for r in self.tables["objects.csv"] if r["object_id"] != plan.retire_id]
            for row in self.tables["observations.csv"]:
                if row["object_id"] == plan.retire_id:
                    row["object_id"] = plan.keep_id
            self.refresh_pair_statuses()
            validate_tables(self.tables)
            self.merges.append({"keep_id": plan.keep_id, "retire_id": plan.retire_id,
                                "frames": len({v["frame_id"] for v in plan.views})})
        except BaseException:
            self.tables, self.merges = previous
            raise
        self.history.append(previous)
        self.matches.append((left_id, right_id, plan.keep_id))
        return plan.keep_id, plan.retire_id

    def add_match(self, left_id, right_id):
        a, b = self.selected_annotations(left_id, right_id)
        bindings, reverse = validate_tables(self.tables)
        ga = bindings.get((self.left.frame_id, left_id))
        gb = bindings.get((self.right.frame_id, right_id))
        if ga and gb and ga != gb:
            raise MergeRequired(f"{ga} and {gb} are different identities. Review both tracks before merging.")
        oid = ga or gb or self.next_object_id()
        for frame, annotation in ((self.left, a), (self.right, b)):
            assigned = reverse.get((frame.frame_id, oid))
            if assigned is not None and assigned != annotation.local_id:
                raise DataError(f"{oid} already belongs to local {assigned} in {frame.frame_id}.")
        if ga and gb:
            return oid, False
        previous = self.checkpoint()
        try:
            if ga is None and gb is None:
                self.tables["objects.csv"].append({"object_id": oid, "building": self.building,
                    "class_id": str(a.class_id), "class_name": CLASS_NAMES[a.class_id], "merged_from_ids": ""})
            for frame, annotation, existing in ((self.left, a, ga), (self.right, b, gb)):
                if existing:
                    continue
                for corner, (x, y, v) in zip(CORNER_ORDER, annotation.keypoints):
                    self.tables["observations.csv"].append({
                        "frame_id": frame.frame_id, "local_object_id": str(annotation.local_id),
                        "object_id": oid, "corner": corner, "x_px": pixel_text(x),
                        "y_px": pixel_text(y), "visibility": str(v)})
            self.refresh_pair_statuses()
            validate_tables(self.tables)
        except BaseException:
            self.tables, self.merges = previous
            raise
        self.history.append(previous)
        self.matches.append((left_id, right_id, oid))
        return oid, True

    def undo(self):
        if not self.history:
            return None
        self.tables, self.merges = self.history.pop()
        return self.matches.pop()

    def print_review(self):
        print(f"\nReview: {self.left.frame_id} <-> {self.right.frame_id}")
        bindings, inverse = validate_tables(self.tables)
        staged = {resolve_object_id(self.tables, oid) for _, _, oid in self.matches}
        count = 0
        for (fid, lid), oid in sorted(bindings.items()):
            rid = inverse.get((self.right.frame_id, oid))
            if fid == self.left.frame_id and rid is not None:
                count += 1
                note = "STAGED" if oid in staged else "already saved"
                print(f"  {lid:3d} <-> {rid:3d}  {oid}  {note}")
        new_objects = len(self.tables["objects.csv"]) - len(self.store.tables["objects.csv"])
        new_obs = len(self.tables["observations.csv"]) - len(self.store.tables["observations.csv"])
        new_frames = len(self.tables["frames.csv"]) - len(self.store.tables["frames.csv"])
        print(f"{count} shared objects; {new_objects:+d} permanent object rows; {new_obs:+d} corner observations.")
        print(f"+{new_frames} registered frames/column blocks; missing observations remain blank.")
        for merge in self.merges:
            print(f"  STAGED MERGE: {merge['retire_id']} -> {merge['keep_id']} ({merge['frames']} frames combined).")
        if self.merges:
            print("All stored corners are retained. Retired IDs are reserved in merged_from_ids.")
        if self.store.legacy is not None:
            print("This SAVE also converts existing tracks and backs up the original CSVs in legacy_backup/.")
        if self.store.upgrade is not None:
            print("This SAVE adds merged_from_ids and backs up your version-2 CSVs in v2_backup/.")
        print(f"Output: {self.store.directory}")

    def save_if_approved(self, answer, status="partial"):
        if answer.strip().upper() != "SAVE":
            return False
        if status not in PAIR_STATUSES:
            raise DataError("Choose partial, reviewed or no_overlap.")
        for frame in (self.left, self.right):
            frame.check_unchanged()
        tables = copy.deepcopy(self.tables)
        a, b = canonical_pair(self.left.frame_id, self.right.frame_id)
        records = [r for r in tables["pairs.csv"] if (r["frame_a"], r["frame_b"]) != (a, b)]
        records.append({"frame_a": a, "frame_b": b, "status": status})
        tables["pairs.csv"] = records
        validate_tables(tables)
        result = self.store.commit(tables)
        self.tables = tables
        return result


class MergeReview:
    """Review all observations of two IDs before staging their combination."""
    PER_PAGE = 6

    def __init__(self, parent, plan):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        self.parent, self.plan, self.plt = parent, plan, plt
        self.page_index, self.seen_pages, self.axes = 0, set(), []
        self.page_count = max(1, math.ceil(len(plan.views) / self.PER_PAGE))
        self._closed, self.message = False, ""
        self.fig = plt.figure(figsize=(13, 9))
        self.fig.canvas.manager.set_window_title("Review object-ID merge - version 3")
        self.fig.suptitle(f"Keep {plan.keep_id}; retire {plan.retire_id}", fontsize=16, y=.975)
        self.subtitle = self.fig.text(.5, .932, "", ha="center", fontsize=10)
        self.status = self.fig.text(.05, .145, "", va="top", fontsize=10, wrap=True)
        self.fig.text(.05, .022, "Stage merge changes memory only. Undo last can revert it. "
                      "The pair's final SAVE writes the CSVs.", fontsize=9)
        self.buttons = []
        for label, x, callback in [
            ("Previous page", .05, lambda e: self.page(-1)),
            ("Next page", .27, lambda e: self.page(1)),
            ("Stage merge", .49, self.stage),
            ("Cancel merge", .71, self.cancel),
        ]:
            button = Button(self.fig.add_axes((x, .060, .19, .047)), label)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.fig.canvas.mpl_connect("close_event", self.closed)
        self.fig.canvas.mpl_connect("key_press_event", lambda e: self.cancel() if e.key == "escape" else None)
        self.draw()

    def page(self, delta):
        self.page_index = max(0, min(self.page_count - 1, self.page_index + delta))
        self.message = ""
        self.draw()

    def draw_view(self, ax, view):
        """Display a crop; saved pixel coordinates are never changed."""
        from PIL import Image
        corners = [r for r in view["corners"] if integer(r["visibility"], "visibility") > 0]
        conflict = next((c for c in self.plan.conflicts if c[0] == view["frame_id"]), None)
        try:
            with Image.open(view["image_path"]) as raw:
                image = raw.convert("RGB")
            width, height = image.size
            if corners:
                xs, ys = [float(r["x_px"]) for r in corners], [float(r["y_px"]) for r in corners]
                margin = max(max(xs) - min(xs), max(ys) - min(ys)) * .20 + 12
                x0 = max(0, min(width - 1, math.floor(min(xs) - margin)))
                y0 = max(0, min(height - 1, math.floor(min(ys) - margin)))
                x1 = max(x0 + 1, min(width, math.ceil(max(xs) + margin)))
                y1 = max(y0 + 1, min(height, math.ceil(max(ys) + margin)))
            else:
                x0, y0, x1, y1 = 0, 0, width, height
            image = image.crop((x0, y0, x1, y1))
            image.thumbnail((520, 320))
            # Extent preserves the original coordinates despite display downsampling.
            ax.imshow(image, origin="upper", extent=(x0-.5, x1-.5, y1-.5, y0-.5))
            for row in corners:
                color = "#25e4ff" if row["visibility"] == "2" else "#ffa628"
                x, y = float(row["x_px"]), float(row["y_px"])
                ax.plot(x, y, "+", color=color, markersize=9)
                ax.annotate(row["corner"], (x, y), xytext=(3, 3), textcoords="offset points",
                            fontsize=8, color=color,
                            bbox=dict(facecolor="black", alpha=.6, edgecolor="none", pad=1))
            ax.set_xlim(x0-.5, x1-.5)
            ax.set_ylim(y1-.5, y0-.5)
        except (OSError, ValueError):
            ax.text(.5, .5, "Image unavailable\nSaved corner data will be retained.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
        title = f"{view['frame_id']} | local {view['local_object_id']}\n{view['object_id']}"
        flags = " ".join(f"{r['corner']}:v{r['visibility']}" for r in view["corners"])
        if conflict:
            title += f"\nCONFLICT: locals {conflict[1]} and {conflict[2]} in this frame"
        else:
            title += f"\n{flags}"
        ax.set_title(title, fontsize=9, color="#ae1532" if conflict else "#18212b", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#ae1532" if conflict else "#b7c4cd")
            spine.set_linewidth(2 if conflict else .8)

    def draw(self):
        for ax in self.axes:
            ax.remove()
        self.axes = []
        self.seen_pages.add(self.page_index)
        start = self.page_index * self.PER_PAGE
        for index, view in enumerate(self.plan.views[start:start+self.PER_PAGE]):
            row, col = divmod(index, 3)
            ax = self.fig.add_axes((.04+col*.325, .575-row*.345, .29, .245))
            self.draw_view(ax, view)
            self.axes.append(ax)
        self.subtitle.set_text(f"Page {self.page_index+1}/{self.page_count} | "
                               f"{len(self.plan.views)} observations | "
                               "Check the same physical object in every view.")
        if self.plan.conflicts:
            first = self.plan.conflicts[0]
            default = (f"Merge blocked: {len(self.plan.conflicts)} frame(s) contain different local objects. "
                       f"First: {first[0]}, locals {first[1]} and {first[2]}.\n"
                       "Cancel and correct the earlier association; different frames in one folder are allowed.")
        elif len(self.seen_pages) < self.page_count:
            default = "Review every page before Stage merge. Cancel merge keeps both object IDs unchanged."
        else:
            default = ("If all views show the same physical object, Stage merge combines the records.\n"
                       f"{self.plan.retire_id} will remain recorded in merged_from_ids.")
        self.status.set_text(self.message or default)
        self.status.set_color("#ae1532" if self.plan.conflicts or self.message else "#18212b")
        ready = self.plan.can_merge and len(self.seen_pages) == self.page_count
        self.buttons[2].label.set_color("#18212b" if ready else "#888888")
        self.fig.canvas.draw_idle()

    def stage(self, _event=None):
        if self._closed:
            return
        if not self.plan.can_merge:
            self.message = "Merge blocked by conflicting annotations in the same frame. No records changed."
            self.draw()
            return
        if len(self.seen_pages) < self.page_count:
            self.message = "Visit all review pages before staging this merge."
            self.draw()
            return
        try:
            result = self.parent.session.merge_match(
                self.plan.left_local_id, self.plan.right_local_id, self.plan)
        except DataError as exc:
            self.message = str(exc)
            self.draw()
            return
        self.dismiss(result)

    def dismiss(self, result=None, notify_parent=True):
        if self._closed:
            return
        self._closed = True
        self.plt.close(self.fig)
        if notify_parent:
            self.parent.merge_review_done(self, result)

    def cancel(self, _event=None):
        self.dismiss()

    def closed(self, _event=None):
        self.dismiss()


class PairViewer:
    """Two image panels; clicks preview only. A separate button confirms."""
    def __init__(self, session):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, TextBox
        self.plt, self.session = plt, session
        self.left_id = self.right_id = None
        self.skipped = set()
        self.outcome = "cancel"  # Window close, Ctrl+C, and Cancel are never SAVE.
        self.message = "Select a source object, then its match. Check corners before Confirm match."
        self.error = False
        self.last_hit = None
        self.setting_text = False
        self.merge_dialog = None
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 9))
        self.fig.canvas.manager.set_window_title("Correspondence builder - version 3")
        self.fig.subplots_adjust(left=.035, right=.975, top=.87, bottom=.31, wspace=.10)
        self.fig.suptitle(f"{session.building} correspondence builder - version 3", fontsize=17, y=.98)
        self.header = self.fig.text(.5, .936, "", ha="center", fontsize=10)
        self.status = self.fig.text(.04, .247, "", fontsize=10, va="top", wrap=True)
        self.fig.text(.04, .028, "Clicks only preview. Review/finish asks for SAVE in the terminal. "
                      "Closing this window discards this session.", fontsize=10)
        self.source_box = TextBox(self.fig.add_axes((.11, .181, .08, .036)), "Source ID ")
        self.target_box = TextBox(self.fig.add_axes((.30, .181, .08, .036)), "Target ID ")
        self.source_box.on_submit(lambda value: self.submit_id("left", value))
        self.target_box.on_submit(lambda value: self.submit_id("right", value))
        # Keep references to every widget for the full figure lifetime.
        self.buttons = []
        for label, x, callback in (("Focus pair", .43, self.focus), ("Show all", .565, self.show_all)):
            button = Button(self.fig.add_axes((x, .178, .12, .041)), label)
            button.on_clicked(callback)
            self.buttons.append(button)
        controls = [("Confirm match", self.confirm), ("Undo last", self.undo),
                    ("Skip source", self.skip), ("Clear target", self.clear_target),
                    ("Review / finish", self.finish), ("Cancel", self.cancel)]
        for i, (label, callback) in enumerate(controls):
            button = Button(self.fig.add_axes((.04 + i * .156, .092, .145, .057)), label)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.fig.canvas.mpl_connect("button_press_event", self.click)
        self.fig.canvas.mpl_connect("close_event", self.closed)
        self.fig.canvas.mpl_connect("key_press_event", self.key)
        self.advance()
        self.draw(reset=True)

    def set_message(self, message, error=False):
        self.message, self.error = message, error

    def merge_review_open(self):
        if self.merge_dialog is None:
            return False
        self.set_message("Finish or cancel the open merge review before changing this pair.", True)
        self.draw()
        return True

    def merge_review_done(self, dialog, result):
        if self.merge_dialog is not dialog:
            return
        self.merge_dialog = None
        if result is None:
            self.set_message("Merge cancelled. Both IDs and their observations are unchanged.")
        else:
            kept, retired = result
            self.advance()
            self.set_message(f"Staged merge: {retired} -> {kept}. All frame observations retained. "
                             "Undo last restores the separate IDs; final SAVE writes the change.")
        self.draw()

    def advance(self):
        bindings = self.session.bindings()
        right_globals = {g for (f, _), g in bindings.items() if f == self.session.right.frame_id}
        available = [a.local_id for a in self.session.left.annotations.values()
                     if self.session.eligible(a) and a.local_id not in self.skipped and
                     bindings.get((self.session.left.frame_id, a.local_id)) not in right_globals]
        self.left_id = available[0] if available else None
        self.right_id = None
        if not available:
            self.set_message("No unreviewed sources remain. Use Review / finish, "
                             "or choose an ID to inspect an earlier object.")

    def candidates(self, side):
        frame = self.session.left if side == "left" else self.session.right
        selected_class = (self.session.left.annotations[self.left_id].class_id
                          if self.left_id is not None else None)
        return [a for a in frame.annotations.values() if self.session.eligible(a) and
                (side == "left" or selected_class is None or a.class_id == selected_class)]

    def choose(self, side, local_id):
        if self.merge_dialog is not None:
            raise DataError("Finish or cancel the open merge review first.")
        frame = self.session.left if side == "left" else self.session.right
        if local_id not in frame.annotations:
            raise DataError(f"No local object {local_id} in {frame.frame_id}.")
        annotation = frame.annotations[local_id]
        if not self.session.eligible(annotation):
            raise DataError("This object has a v=0/1 corner and visible-only mode is enabled.")
        if side == "left":
            self.left_id, self.right_id = local_id, None
            self.skipped.discard(local_id)
        else:
            if self.left_id is None:
                raise DataError("Select the source object first.")
            if annotation.class_id != self.session.left.annotations[self.left_id].class_id:
                raise DataError("The target has a different class from the source.")
            self.right_id = local_id
        self.set_message("Preview only. Check that TL/TR/BR/BL identify the same physical corners, "
                         "then click Confirm match.")

    def submit_id(self, side, value):
        if self.setting_text or not value.strip():
            return
        try:
            self.choose(side, integer(value, "local object ID"))
        except (ValueError, KeyError) as exc:
            self.set_message(str(exc), True)
        self.draw()

    def click(self, event):
        if self.merge_dialog is not None:
            return
        if event.inaxes not in self.axes or event.xdata is None or event.ydata is None:
            return
        toolbar = getattr(self.fig.canvas.manager, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return  # Navigation clicks must not select an object.
        if event.button == 3:
            self.clear_target()
            return
        if event.button != 1:
            return
        side = "left" if event.inaxes is self.axes[0] else "right"
        hits = [a.local_id for a in self.candidates(side)
                if a.bbox[0] <= event.xdata <= a.bbox[2]
                and a.bbox[1] <= event.ydata <= a.bbox[3]]
        if not hits:
            # In particular, NEVER snap a missed click to the nearest window.
            self.set_message("No eligible box at this click. Click inside a box or type its local ID.", True)
            self.draw()
            return
        signature = (side, tuple(hits))
        current = self.left_id if side == "left" else self.right_id
        selected = hits[(hits.index(current) + 1) % len(hits)] if (
            self.last_hit == signature and current in hits) else hits[0]
        self.last_hit = signature
        try:
            self.choose(side, selected)
            if len(hits) > 1:
                self.set_message(f"Overlapping boxes {hits}: previewing {selected}. "
                                 "Click again to cycle, or type the exact ID. Confirm only after checking.")
        except DataError as exc:
            self.set_message(str(exc), True)
        self.draw()

    def confirm(self, _event=None):
        if self.merge_review_open():
            return
        if self.left_id is None or self.right_id is None:
            self.set_message("Choose both objects before confirming.", True)
        else:
            try:
                left, right = self.left_id, self.right_id
                oid, added = self.session.add_match(left, right)
                self.advance()
                self.set_message(f"{left} <-> {right}: {oid}. " +
                                 ("Staged in memory; Undo last can revert it." if added else
                                  "Already saved/linked; no duplicate rows added."))
            except MergeRequired:
                try:
                    plan = self.session.plan_merge(self.left_id, self.right_id)
                    self.merge_dialog = MergeReview(self, plan)
                    self.merge_dialog.fig.show()
                    self.set_message("Review the two IDs in the merge window. No changes have been staged yet.")
                except DataError as exc:
                    self.set_message(str(exc), True)
            except DataError as exc:
                self.set_message(str(exc), True)
        self.draw()

    def undo(self, _event=None):
        if self.merge_review_open():
            return
        match = self.session.undo()
        if match is None:
            self.set_message("No staged match to undo. Existing saved matches are not changed.")
        else:
            self.left_id, self.right_id, oid = match
            self.skipped.discard(self.left_id)
            self.set_message(f"Undid {oid}. The previous selection is shown for correction; "
                             "choose a target and confirm again.")
        self.draw()

    def skip(self, _event=None):
        if self.merge_review_open():
            return
        if self.left_id is not None:
            self.skipped.add(self.left_id)
        self.advance()
        self.draw()

    def clear_target(self, _event=None):
        if self.merge_review_open():
            return
        self.right_id = None
        self.set_message("Target preview cleared. Confirmed matches are unchanged; use Undo last for those.")
        self.draw()

    def finish(self, _event=None):
        if self.merge_review_open():
            return
        self.outcome = "review"
        self.plt.close(self.fig)

    def cancel(self, _event=None):
        self.outcome = "cancel"
        if self.merge_dialog is not None:
            self.merge_dialog.dismiss(notify_parent=False)
            self.merge_dialog = None
        self.plt.close(self.fig)

    def closed(self, _event=None):
        if self.merge_dialog is not None:
            self.merge_dialog.dismiss(notify_parent=False)
            self.merge_dialog = None
        if self.outcome != "review":
            self.outcome = "cancel"

    def key(self, event):
        if event.key == "escape":
            self.cancel()

    def focus(self, _event=None):
        for ax, frame, lid in zip(self.axes, (self.session.left, self.session.right),
                                  (self.left_id, self.right_id)):
            if lid is None:
                continue
            x0, y0, x1, y1 = frame.annotations[lid].bbox
            padding = max(x1 - x0, y1 - y0) * .35 + 10
            ax.set_xlim(x0 - padding, x1 + padding)
            ax.set_ylim(y1 + padding, y0 - padding)
        self.fig.canvas.draw_idle()

    def show_all(self, _event=None):
        self.draw(reset=True)

    def draw(self, reset=False):
        from matplotlib.patches import Rectangle
        bindings = self.session.bindings()
        for side, ax, frame, selected in zip(
            ("left", "right"), self.axes, (self.session.left, self.session.right),
            (self.left_id, self.right_id),
        ):
            limits = (ax.get_xlim(), ax.get_ylim())
            ax.clear()
            # Pixel centers use imshow's default origin/extent; do not use extent=(0,W,H,0).
            ax.imshow(frame.image, origin="upper", interpolation="nearest")
            for annotation in self.candidates(side):
                lid = annotation.local_id
                active = lid == selected
                mapped = (frame.frame_id, lid) in bindings
                color = "#ff3a7d" if active else "#27e6ae" if mapped else "#ffc857"
                x0, y0, x1, y1 = annotation.bbox
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor=color, linewidth=2.5 if active else .8))
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, str(lid), color="white",
                        ha="center", va="center", fontsize=10 if active else 8,
                        bbox=dict(facecolor="#18212b", alpha=.85, edgecolor=color, pad=2))
                if active:
                    for corner, (x, y, v) in zip(CORNER_ORDER, annotation.keypoints):
                        if v == 0:
                            continue
                        kp_color = "#25e4ff" if v == 2 else "#ffa628"
                        ax.plot(x, y, marker="+" if v == 2 else "x", color=kp_color,
                                markersize=11, markeredgewidth=1.5)
                        ax.annotate(corner, (x, y), xytext=(5, 5), textcoords="offset points",
                                    color=kp_color, fontsize=9,
                                    bbox=dict(facecolor="black", alpha=.65, edgecolor="none", pad=1))
            suffix = "source" if side == "left" else "target (same class)"
            title = f"{frame.frame_id} | {suffix}"
            if selected is not None:
                a = frame.annotations[selected]
                gid = bindings.get((frame.frame_id, selected), "unassigned")
                title += f"\nLocal {selected}: {CLASS_NAMES[a.class_id]} | {gid}"
                title += "\n" + "  ".join(f"{c}:v{k[2]}" for c, k in zip(CORNER_ORDER, a.keypoints))
            ax.set_title(title, fontsize=9, pad=8)
            ax.set_axis_off()
            if not reset:
                ax.set_xlim(limits[0])
                ax.set_ylim(limits[1])
        count = len(self.session.matches)
        mode = "four v=2 corners required" if self.session.visible_only else "v=0/1/2 retained"
        self.header.set_text(f"{count} changes staged | {mode} | "
                             "pink: selected; green: assigned; yellow: unassigned")
        self.status.set_text(self.message)
        self.status.set_color("#a01427" if self.error else "#18212b")
        self.setting_text = True
        try:
            self.source_box.set_val("" if self.left_id is None else str(self.left_id))
            self.target_box.set_val("" if self.right_id is None else str(self.right_id))
        finally:
            self.setting_text = False
        self.fig.canvas.draw_idle()

    def run(self):
        self.plt.show(block=True)
        return self.outcome


@lru_cache(maxsize=96)
def thumbnail(path_text, size, mtime_ns):
    """Cache small display copies only; full-resolution coordinates are untouched."""
    from PIL import Image
    with Image.open(path_text) as image:
        thumb = image.convert("RGB")
        thumb.thumbnail((400, 260))
        return thumb


class FramePicker:
    """Paginated contact sheets for two folders; no correspondence mutations."""
    def __init__(self, catalogue, store, left_folder, right_folder):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        if left_folder == right_folder:
            raise DataError("Choose two different folders.")
        self.plt, self.catalogue, self.store = plt, catalogue, store
        self.folders = (left_folder, right_folder)
        self.specs = [sorted((s for s in catalogue.values() if s.folder_id == folder),
                             key=lambda s: natural_key(s.image_path.name)) for folder in self.folders]
        if not all(self.specs):
            raise DataError("Each folder must contain an available labeled frame.")
        self.statuses = {(r["frame_a"], r["frame_b"]): r["status"] for r in store.tables["pairs.csv"]}
        self.known = {r["frame_id"] for r in store.tables["frames.csv"]}
        self.include_reviewed = False
        self.pages, self.selected, self.result = [0, 0], [None, None], None
        self.fig = plt.figure(figsize=(15, 9))
        self.title = self.fig.suptitle("Choose one labeled frame from each folder", y=.98, fontsize=16)
        self.message = self.fig.text(.5, .913, "", ha="center", fontsize=10)
        self.axes, self.hits = [], {}
        self.buttons = []
        for label, rect, callback in [
            ("Left previous", (.03,.11,.14,.048), lambda e: self.page(0,-1)),
            ("Left next", (.19,.11,.14,.048), lambda e: self.page(0,1)),
            ("Right previous", (.53,.11,.14,.048), lambda e: self.page(1,-1)),
            ("Right next", (.69,.11,.14,.048), lambda e: self.page(1,1)),
            ("Show reviewed", (.05,.035,.19,.05), self.toggle_reviewed),
            ("Open selected pair", (.40,.035,.20,.05), self.open_pair),
            ("Back to folders", (.75,.035,.19,.05), self.cancel),
        ]:
            button = Button(self.fig.add_axes(rect), label)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.fig.canvas.mpl_connect("button_press_event", self.click)
        self.fig.canvas.mpl_connect("key_press_event", lambda e: self.cancel() if e.key == "escape" else None)
        self.draw()

    def pool(self, side):
        rows = self.specs[side]
        if side == 1 and self.selected[0] is not None and not self.include_reviewed:
            fid = self.selected[0]
            rows = [s for s in rows if self.statuses.get(canonical_pair(fid, s.frame_id))
                    not in ("reviewed", "no_overlap")]
        return rows

    def page(self, side, delta):
        last = max(0, (len(self.pool(side))-1)//THUMBNAILS_PER_PAGE)
        self.pages[side] = min(last, max(0, self.pages[side]+delta))
        self.draw()

    def choose(self, side, fid):
        if fid not in {s.frame_id for s in self.pool(side)}:
            raise DataError("That frame is not available in the current folder/filter.")
        self.selected[side] = fid
        if side == 0:
            self.selected[1], self.pages[1] = None, 0
        self.draw()

    def click(self, event):
        if event.button == 1 and event.inaxes in self.hits:
            side, fid = self.hits[event.inaxes]
            self.choose(side, fid)

    def toggle_reviewed(self, _event=None):
        self.include_reviewed = not self.include_reviewed
        self.selected[1], self.pages[1] = None, 0
        self.buttons[4].label.set_text("Hide reviewed" if self.include_reviewed else "Show reviewed")
        self.draw()

    def open_pair(self, _event=None):
        if all(self.selected):
            self.result = tuple(self.selected)
            self.plt.close(self.fig)
        else:
            self.message.set_text("Click one image on the left and one on the right before opening.")
            self.fig.canvas.draw_idle()

    def cancel(self, _event=None):
        self.result = None
        self.plt.close(self.fig)

    def draw(self):
        for ax in self.axes:
            ax.remove()
        self.axes, self.hits = [], {}
        for side in (0, 1):
            pool = self.pool(side)
            self.pages[side] = min(self.pages[side], max(0, (len(pool)-1)//THUMBNAILS_PER_PAGE))
            start = self.pages[side]*THUMBNAILS_PER_PAGE
            for index, spec in enumerate(pool[start:start+THUMBNAILS_PER_PAGE]):
                row, column = divmod(index, 3)
                ax = self.fig.add_axes((.025 + side*.50 + column*.155, .665-row*.235, .145, .18))
                try:
                    stat = spec.image_path.stat()
                    ax.imshow(thumbnail(str(spec.image_path), stat.st_size, stat.st_mtime_ns))
                except OSError:
                    ax.text(.5, .5, "Image unavailable", ha="center", va="center")
                status = ""
                if side == 1 and self.selected[0]:
                    status = self.statuses.get(canonical_pair(self.selected[0], spec.frame_id), "unreviewed")
                extra = "new frame" if spec.frame_id not in self.known else "registered"
                ax.set_title(f"{spec.image_path.name}\n{spec.visible_count}/{spec.object_count} fully v=2 | "
                             f"{status or extra}", fontsize=8, pad=3)
                ax.set_xticks([])
                ax.set_yticks([])
                selected = self.selected[side] == spec.frame_id
                for spine in ax.spines.values():
                    spine.set_edgecolor("#e64678" if selected else "#c1c6ca")
                    spine.set_linewidth(3 if selected else .7)
                self.axes.append(ax)
                self.hits[ax] = (side, spec.frame_id)
        totals = [max(1, math.ceil(len(self.pool(i))/THUMBNAILS_PER_PAGE)) for i in (0,1)]
        self.title.set_text(f"Folder {self.folders[0]} (page {self.pages[0]+1}/{totals[0]})"
                            f"                  Folder {self.folders[1]} (page {self.pages[1]+1}/{totals[1]})")
        message = f"Selected: {self.selected[0] or 'left image?'}  <->  {self.selected[1] or 'right image?'}"
        if self.selected[0] and not self.pool(1):
            message += " | No pending pairs for this left image; Show reviewed to reopen."
        elif not self.include_reviewed:
            message += " | Reviewed/no-overlap pairs are hidden after choosing the left image."
        self.message.set_text(message)
        self.fig.canvas.draw_idle()

    def run(self):
        self.plt.show(block=True)
        return self.result


def load_inventory(args):
    legacy = args.legacy_dir
    if legacy is None and not args.no_legacy_import and LEGACY_DIR is not None:
        if all((LEGACY_DIR / name).is_file() for name in LEGACY_FILES):
            legacy = LEGACY_DIR
    store = Store.load(args.output, legacy)
    if any(o["building"] != args.building for o in store.tables["objects.csv"]):
        raise DataError("Existing object records belong to another building; check --building and --output.")
    catalogue, messages = discover_frames(args.labels, args.images, store.tables["frames.csv"], store.directory)
    return store, catalogue, messages


def print_inventory(catalogue, store, messages, details=False):
    known = {r["frame_id"] for r in store.tables["frames.csv"]}
    folders = sorted({s.folder_id for s in catalogue.values()}, key=natural_key)
    print(f"\nBuilding inventory: {len(catalogue)} labeled frames in {len(folders)} folders.")
    print("No.  Folder      Frames   New frames")
    for i, folder in enumerate(folders, 1):
        rows = [s for s in catalogue.values() if s.folder_id == folder]
        print(f"{i:3d}  {folder:<12} {len(rows):5d}   {sum(s.frame_id not in known for s in rows):5d}")
    missing = known - set(catalogue)
    if missing:
        print(f"{len(missing)} saved frames are currently unavailable in the label/image roots; their records are retained.")
    if messages:
        print(f"{len(messages)} labels excluded:")
        for message in messages if details else messages[:8]:
            print("  " + message)
        if not details and len(messages) > 8:
            print("  Use --list-frames for the complete exclusion list.")
    if details:
        for spec in catalogue.values():
            print(f"{spec.frame_id}: {spec.object_count} objects, {spec.visible_count} with four v=2 corners")
            print(f"  label: {spec.label_path}\n  image: {spec.image_path}")
    return folders


def gui_backend(backend=None):
    import matplotlib
    if backend:
        matplotlib.use(backend)
    import matplotlib.pyplot as plt
    name = matplotlib.get_backend().lower()
    if name in {"agg", "pdf", "pgf", "ps", "svg", "cairo", "template"} or "inline" in name:
        raise DataError(f"Backend {name!r} cannot open a desktop window. On your Mac try --backend MacOSX.")
    return plt


def review_session(session):
    """Return True if approved, False if discarded; BACK retains staged changes."""
    while True:
        viewer = PairViewer(session)
        if viewer.run() != "review":
            print("Pair session discarded. Previously saved CSVs are unchanged.")
            return False
        session.print_review()
        while True:
            answer = input("\nType SAVE PARTIAL, SAVE REVIEWED, or SAVE NO_OVERLAP; "
                           "BACK to edit; anything else to discard (SAVE alone = PARTIAL): ").strip()
            if answer.upper() == "BACK":
                break
            words = answer.split(maxsplit=1)
            if not words or words[0].upper() != "SAVE":
                print("Pair session discarded. Previously saved CSVs are unchanged.")
                return False
            status = words[1].lower().replace(" ", "_") if len(words) == 2 else "partial"
            try:
                changed = session.save_if_approved("SAVE", status)
            except DataError as exc:
                print(f"Cannot save: {exc}\nUse BACK to edit, or cancel and refresh changed source/CSV files.")
                continue
            print(("Saved" if changed else "Already saved") + f": {status}. Output: {session.store.directory}")
            return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=LABEL_ROOT)
    parser.add_argument("--images", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--building", default=BUILDING)
    parser.add_argument("--legacy-dir", type=Path, help="Import frames/objects/observations CSVs on first use.")
    parser.add_argument("--no-legacy-import", action="store_true", help="Disable automatic import from LEGACY_DIR.")
    parser.add_argument("--pair", nargs=2, metavar=("FRAME_A", "FRAME_B"),
                        help="Open one pair directly using full IDs, e.g. 0051_POS_073 0052_POS_105.")
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--visible-only", dest="visible_only", action="store_true")
    visibility.add_argument("--all-visibility", dest="visible_only", action="store_false")
    parser.set_defaults(visible_only=ONLY_FULLY_VISIBLE)
    parser.add_argument("--list-frames", action="store_true", help="Read-only inventory; no GUI or CSV writes.")
    parser.add_argument("--backend", help="Optional desktop backend, e.g. MacOSX, TkAgg or QtAgg.")
    args = parser.parse_args(argv)
    if os.name != "posix":
        parser.error("This version supports macOS/Linux (fcntl file locking).")
    if args.legacy_dir is not None and args.no_legacy_import:
        parser.error("Choose either --legacy-dir or --no-legacy-import.")
    plt = None
    try:
        while True:
            store, catalogue, messages = load_inventory(args)
            folders = print_inventory(catalogue, store, messages, args.list_frames)
            if args.list_frames:
                return 0
            if len(folders) < 2:
                raise DataError("At least two dataset folders need valid labeled images for cross-folder matching.")
            if plt is None:
                plt = gui_backend(args.backend)
            if args.pair:
                pair = tuple(args.pair)
                if any(fid not in catalogue for fid in pair):
                    raise DataError("Selected frame ID is not in the dataset inventory. Run --list-frames to see valid IDs.")
            else:
                choice = input("\nChoose two folder numbers (e.g. 1 2), R to rescan, or Q to quit: ").strip()
                if choice.lower() in ("q", "quit", ""):
                    return 0
                if choice.lower() in ("r", "refresh"):
                    continue
                try:
                    selected = [int(n) for n in choice.replace(",", " ").split()]
                    if len(selected) != 2 or selected[0] == selected[1] or any(n < 1 or n > len(folders) for n in selected):
                        raise ValueError
                except ValueError:
                    print("Enter two different folder numbers from the table.")
                    continue
                picker = FramePicker(catalogue, store, folders[selected[0]-1], folders[selected[1]-1])
                pair = picker.run()
                if pair is None:
                    continue
            left, right = catalogue[pair[0]], catalogue[pair[1]]
            if left.folder_id == right.folder_id:
                raise DataError("Same-folder matching is disabled; choose frames from different folders.")
            try:
                session = MatchSession(store, left.load(), right.load(), args.building,
                                       args.visible_only, catalogue)
            except (DataError, OSError) as exc:
                if args.pair:
                    raise
                print(f"Cannot open pair: {exc}")
                continue
            print(f"\nMatching {left.frame_id} <-> {right.frame_id}. Return to the terminal after Review / finish.")
            review_session(session)
            if args.pair:
                return 0
            # Reload the committed project and rescan labels before selecting again.
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Unsaved pair changes were discarded; earlier approved sessions remain saved.")
        return 130
    except (DataError, OSError, ValueError, ImportError) as exc:
        print(f"\nStopped: {exc}", file=sys.stderr)
        print("If an approved save was interrupted, reopen this script to recover before reading the CSVs.", file=sys.stderr)
        return 1
    finally:
        if plt is not None:
            plt.close("all")


if __name__ == "__main__":
    def stop_requested(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_requested)
    raise SystemExit(main())
