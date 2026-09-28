"""Readable Stage 1 driver. Run: python main.py (or %run main.py in Jupyter).

Flow: inputs -> groups -> anchor candidates -> E/R/t -> triangulation -> BA
      -> all common objects -> final BA/screening -> select -> measure/plot/save.

Only accepted reconstructions get a group folder. CSV summaries record failures.
experiment.json is OUTPUT provenance, not a separate configuration to maintain.
"""

from pathlib import Path
import ast
import hashlib
import json
import re
import time

import cv2
import numpy as np
import pandas as pd

from geometry import (CORNERS, estimate_pairs, rotation_cycles, initial_models,
                      refine_and_screen, reconstruct_objects, evaluation, dimensions,
                      world_transform)
from plots import save_plots


# ============================================================================
# 1. ALL USER SETTINGS -- edit here. No input configuration file is needed.
# ============================================================================
# Relative paths are resolved from your current working directory.
CORRESPONDENCE_DIR = Path("../DHBW_correspondences")
GROUP_CATALOGUE = Path("./group_catalogue.csv")
CALIBRATION_NPZ = Path("../cameraParams.npz")
POSES_CSV = Path("../imageMatcher_POS_ALL/poses_6dof_with_pos.csv")
IMAGE_ROOT = Path("./../../data_folders")
OUTPUT_DIR = Path("stage1_results")

GROUP_IDS = None         # First inspect one group. None = all catalogue groups.
CANDIDATE_IDS = None              # None = every candidate; e.g. ["S01", "ALL"].

CFG = dict(
    building="DHBW",
    image_size=(4032, 3040),       # Width, height of the ORIGINAL annotated/calibrated images.
    world_crs="Use the CRS of your sensor pose CSV",  # Replace with the actual CRS.
    height_datum="Use the height datum of your sensor pose CSV",
    anchor_count=7,
    subset_count=4,               # Up to four DISTINCT spatial subsets, plus ALL common objects.
    random_seed=42,
    essential_threshold_px=2.0,
    min_pair_corners=8,
    min_pair_objects=2,
    cycle_warning_deg=2.0,        # Diagnostic ONLY: never rejects a group before BA.
    initial_error_px=50.0,        # Loose triangulation filter before joint pose refinement.
    pnp_threshold_px=8.0,
    max_seed_pairs=3,             # Fallbacks if reference-star reconstruction fails.
    pixel_sigma_px=1.0,
    baseline_sigma_m=0.02,        # Provisional prior weight, NOT established sensor accuracy.
    robust_scale_px=2.0,
    max_corner_error_px=3.0,      # Acceptance limits; not ground-truth accuracy claims.
    max_object_rmse_px=2.0,
    min_ray_angle_deg=0.5,
    min_group_objects=4,
    min_camera_objects=3,
    max_point_nfev=80,
    max_ba_nfev=300,
    ba_tolerance=1e-6,
    sparse_ba_above_variables=180,
    require_ba_convergence=True,
    screen_rounds=4,
    plot_frame="0052_POS_105",    # Otherwise choose the group's best-supported frame.
    plot_dpi=180,
    plot_elevation_deg=18.0,
    plot_azimuth_deg=-65.0,
    label_map_objects=False,     # Class legends always appear; this adds individual IDs.
    illustrative_reference_panel=False,  # Optional display panel, never a recovered wall.
    vp_trials=300,
    vp_min_edge_px=10.0,
    vp_inlier_angle_deg=1.0,
    vp_min_inlier_fraction=0.5,
    vp_max_orthogonality_error_deg=10.0,
    print_vps=True,
)

# Optional manual subsets. Frame tuples remain valid if catalogue group IDs change.
# Supply a list of subsets; automatic spatial sampling fills any remaining slots.
MANUAL_ANCHORS = {
    "0051_POS_073|0052_POS_105|0053_POS_105|0054_POS_113": [[
        "DHBW_OBJ_026", "DHBW_OBJ_034", "DHBW_OBJ_042", "DHBW_OBJ_046",
        "DHBW_OBJ_001", "DHBW_OBJ_024", "DHBW_OBJ_025",
    ]],
}


# ============================================================================
# 2. INPUTS -- your existing CSV/NPZ formats.
# ============================================================================
def load_inputs():
    paths = [CORRESPONDENCE_DIR / "correspondences.csv", CORRESPONDENCE_DIR / "frames.csv",
             CALIBRATION_NPZ, POSES_CSV, GROUP_CATALOGUE]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        print("Update the paths at the top of main.py. Missing inputs:\n" + "\n".join(missing))
        return None
    table = pd.read_csv(paths[0], dtype=str, keep_default_na=False)
    frames = pd.read_csv(paths[1], dtype=str, keep_default_na=False)
    groups = pd.read_csv(GROUP_CATALOGUE, dtype=str, keep_default_na=False)
    if table.object_id.duplicated().any() or frames.frame_id.duplicated().any() or groups.group_id.duplicated().any():
        print("Input IDs must be unique: check object_id, frame_id and group_id.")
        return None
    if not all(re.fullmatch(r"[A-Za-z0-9_-]+", v) for v in [*frames.frame_id, *groups.group_id]):
        print("Use only letters, digits, underscores and hyphens in frame/group IDs.")
        return None
    groups["frame_ids"] = groups.frame_ids.map(ast.literal_eval)
    with np.load(CALIBRATION_NPZ, allow_pickle=False) as archive:
        K, dist = np.asarray(archive["camera_matrix"], float), np.asarray(archive["dist"], float).ravel()
    if (K.shape != (3, 3) or not np.isfinite(K).all() or not np.isfinite(dist).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1])):
        print("Check camera_matrix and dist in the calibration NPZ.")
        return None
    meta = [c for c in ("building", "class_id", "class_name", "plane_id") if c in table]
    data = dict(K=K, dist=dist, groups=groups, frames=frames.set_index("frame_id").to_dict("index"),
                objects=table.set_index("object_id")[meta].to_dict("index"), raw={}, uv={}, visibility={},
                local_ids={}, sensor_C={}, sensor_Q={}, image_paths={}, input_paths=paths)
    width, height = CFG["image_size"]
    invalid_pixels = 0
    for fid, frame in data["frames"].items():
        columns = [f"{fid}_{c}_{a}" for c in CORNERS for a in ("x", "y", "v")]
        values = table[columns].apply(pd.to_numeric, errors="coerce").to_numpy().reshape(-1, 4, 3)
        for oid, lid, v in zip(table.object_id, table[f"{fid}_local_object_id"], values):
            if not lid:
                continue
            data["raw"][fid, oid], data["visibility"][fid, oid], data["local_ids"][fid, oid] = v[:, :2], v[:, 2], lid
            if not (np.all(v[:, 2] == 2) and np.isfinite(v).all()):
                continue
            if np.any(v[:, :2] < 0) or np.any(v[:, 0] >= width) or np.any(v[:, 1] >= height):
                invalid_pixels += 1
                continue
            data["uv"][fid, oid] = cv2.undistortPoints(v[:, :2].copy().reshape(-1, 1, 2), K, dist, P=K).reshape(4, 2)
        folder, stem = fid.split("_", 1)
        image_value = frame.get("image_path", "")
        candidates = [Path(image_value), CORRESPONDENCE_DIR / image_value,
                      IMAGE_ROOT / folder / (stem + ".bmp")]
        data["image_paths"][fid] = next((p.resolve() for p in candidates if p.is_file()), None)

    # CSV rotations are camera-to-world Q, not world-to-camera R.
    poses = pd.read_csv(POSES_CSV, sep=";")
    poses.columns = poses.columns.str.strip()
    paths_text = "/" + poses.log_path.astype(str).str.replace("\\", "/", regex=False).str.strip("/") + "/"
    names = poses.POS_ref_image.astype(str).str.strip().str.replace("\\", "/", regex=False).str.split("/").str[-1]
    for fid in data["frames"]:
        folder, stem = fid.split("_", 1)
        matched = poses[paths_text.str.contains(f"/{folder}/", regex=False) & names.eq(stem + ".bmp")]
        if len(matched) != 1:
            continue
        row = matched.iloc[0]
        C = row[["x_east_m", "y_north_m", "z_up_m"]].to_numpy(float)
        Q = row[[f"r{i}{j}" for i in range(3) for j in range(3)]].to_numpy(float).reshape(3, 3)
        if (np.isfinite(C).all() and np.isfinite(Q).all() and np.allclose(Q.T @ Q, np.eye(3), atol=1e-4)
                and np.isclose(np.linalg.det(Q), 1, atol=1e-4)):
            data["sensor_C"][fid], data["sensor_Q"][fid] = C, Q
    print(f"Loaded {len(data['objects'])} objects, {len(data['frames'])} frames, {len(groups)} groups; "
          f"{invalid_pixels} out-of-bounds observations excluded.")
    return data


def common_objects(data, frames):
    return sorted(o for o, meta in data["objects"].items()
                  if meta.get("building", CFG["building"]) == CFG["building"]
                  and all((f, o) in data["uv"] for f in frames))


# ============================================================================
# 3. CANDIDATES -- spread anchors over every image; also evaluate ALL objects.
# ============================================================================
def select_candidates(data, frames, common):
    count = min(CFG["anchor_count"], len(common))
    subsets, seen = [], set()

    def add(ids):
        key = tuple(sorted(set(ids)))
        if len(key) >= CFG["min_group_objects"] and set(key).issubset(common) and key not in seen:
            subsets.append(list(ids))
            seen.add(key)

    for ids in MANUAL_ANCHORS.get("|".join(frames), []):
        add(list(dict.fromkeys(ids)))
    centres = np.array([[data["uv"][f, o].mean(axis=0) / CFG["image_size"] for f in frames] for o in common])
    # Distances are deliberately evaluated in all views, not just one image.
    distance = np.linalg.norm(centres[:, None] - centres[None, :], axis=-1).min(axis=2)
    rng = np.random.default_rng(CFG["random_seed"])
    first_order = np.argsort(-np.mean(np.linalg.norm(centres-centres.mean(axis=0), axis=-1), axis=1))
    for trial in range(max(20, len(common) * 2)):
        if len(subsets) >= CFG["subset_count"]:
            break
        first = first_order[trial % len(common)]
        picked = [int(first)]
        while len(picked) < count:
            score = distance[:, picked].min(axis=1)
            # Known plane labels gently encourage depth diversity; no planes are invented.
            known = {data["objects"][common[i]].get("plane_id", "") for i in picked}
            bonus = np.array([1.15 if data["objects"][o].get("plane_id", "") not in known
                              and data["objects"][o].get("plane_id", "") else 1.0 for o in common])
            score = score * bonus * rng.uniform(0.9, 1.1, len(common))
            score[picked] = -1
            picked.append(int(np.argmax(score)))
        add([common[i] for i in picked])
    result = [(f"S{i+1:02d}", ids) for i, ids in enumerate(subsets[:CFG["subset_count"]])
              if set(ids) != set(common)]
    result.append(("ALL", list(common)))
    return result if CANDIDATE_IDS is None else [(name, ids) for name, ids in result if name in CANDIDATE_IDS]


# ============================================================================
# 4. ONE CANDIDATE -- the reconstruction methodology, in execution order.
# ============================================================================
def reconstruct_candidate(data, frames, common, candidate_id, anchors):
    # A. Independent pair geometry and cycle diagnostics. Cycles NEVER stop BA.
    pairs, pair_reports = estimate_pairs(data, frames, anchors, CFG)
    cycles = rotation_cycles(pairs, frames)
    max_cycle = max((c["error_deg"] for c in cycles), default=None)
    report = dict(candidate_id=candidate_id, anchor_ids="|".join(anchors), n_anchors=len(anchors),
                  n_pairs=len(pairs), initial_cycle_max_deg=max_cycle,
                  cycle_warning=max_cycle is not None and max_cycle > CFG["cycle_warning_deg"],
                  status="no_initialization", n_objects=0, n_corner_observations=0, rmse_px=None)
    attempts = []
    if not pairs:
        report["status"] = "no_supported_pair"
        return None, report

    # B. Try the reference star, then seed-pair/PnP recovery if necessary.
    for model in initial_models(data, frames, anchors, pairs, CFG):
        attempt = dict(initialization=model.info["initialization"], seed_pair=model.info.get("seed_pair"))
        # C. Joint BA of anchor points and cameras, then observation screening.
        ok, status = refine_and_screen(data, model, CFG)
        attempt["status"] = "anchor_" + status
        if ok:
            # D. Revisit ALL common objects, including initially rejected anchors.
            reconstruct_objects(data, model, common, CFG)
            # E. Final joint BA uses every accepted object's original observations.
            ok, status = refine_and_screen(data, model, CFG)
            attempt["status"] = "final_" + status
        attempts.append(attempt)
        report["status"] = attempt["status"]
        if not ok:
            continue
        model.info.update(candidate_id=candidate_id, anchor_ids=anchors, common_object_ids=common,
                          initial_pair_reports=pair_reports, initial_rotation_cycles=cycles,
                          initialization_attempts=attempts, metrics=evaluation(data, model))
        metrics = model.info["metrics"]
        report.update(status="accepted", initialization=model.info["initialization"],
                      n_initializations=len(attempts), n_objects=metrics["n_objects"],
                      n_corner_observations=metrics["n_corner_observations"], rmse_px=metrics["rmse_px"])
        return model, report
    report["n_initializations"] = len(attempts)
    return None, report


# ============================================================================
# 5. OUTPUTS -- flat summaries and two JSON files per accepted group.
# ============================================================================
def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, payload):
    path.write_text(json.dumps(json_ready(payload), indent=2, allow_nan=False), encoding="utf-8")


def prepare_output(data):
    # Prevent accidental mixing of different scientific settings in one result folder.
    settings = dict(CFG, manual_anchors=MANUAL_ANCHORS, candidate_ids=CANDIDATE_IDS)
    input_hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in data["input_paths"]}
    fingerprint = hashlib.sha256(json.dumps(json_ready([settings, list(input_hashes.values())]), sort_keys=True).encode()).hexdigest()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = OUTPUT_DIR / "experiment.json"
    if metadata_path.is_file() and json.loads(metadata_path.read_text())["fingerprint"] != fingerprint:
        print("These results use different inputs/settings. Choose a new OUTPUT_DIR at the top of main.py.")
        return None
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ("main.py", "geometry.py", "plots.py")}
    data["implementation_sha256"] = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    save_json(metadata_path, dict(schema_version=1, fingerprint=fingerprint, settings=settings,
                                  input_sha256=input_hashes,
                                  latest_source_sha256=source_hashes,
                                  calibration=dict(camera_matrix=data["K"], dist=data["dist"], image_size=CFG["image_size"]),
                                  conventions=dict(corner_order=CORNERS, rotation="X_camera = R @ (X_local - C)",
                                                   pixels="raw distorted pixels in map.json; undistorted pixels for geometry",
                                                   local_units="metres, scale constrained by sensor camera baselines",
                                                   world_positions="provisional: tied to the reference sensor pose",
                                                   metric_rmse="sqrt(mean(du^2 + dv^2)) over used corner observations",
                                                   uncertainty="internal consistency metrics, not absolute accuracy")))
    summary_path = OUTPUT_DIR / "group_summary.csv"
    if summary_path.is_file():
        summary = pd.read_csv(summary_path, dtype={"group_id": str}, keep_default_na=False).to_dict("records")
    else:
        summary = [dict(group_id=r.group_id, status="not_processed", n_frames=len(r.frame_ids))
                   for r in data["groups"].itertuples()]
    candidate_path = OUTPUT_DIR / "candidate_summary.csv"
    candidates = pd.read_csv(candidate_path, keep_default_na=False).to_dict("records") if candidate_path.is_file() else []
    return {r["group_id"]: r for r in summary}, candidates


def save_summaries(summary, candidates):
    pd.DataFrame(summary.values()).to_csv(OUTPUT_DIR / "group_summary.csv", index=False)
    pd.DataFrame(candidates, columns=list(dict.fromkeys(["group_id", "candidate_id", "status", "selected"] +
                                                       [k for r in candidates for k in r]))).to_csv(OUTPUT_DIR / "candidate_summary.csv", index=False)


def remove_previous_group_outputs(group_dir):
    # Remove only this script's five owned files; never delete input data or other files.
    for name in ("map.json", "geometry.json", "map_3d.png", "map_front.png", "image_overlay.png"):
        (group_dir / name).unlink(missing_ok=True)
    if group_dir.is_dir() and not any(group_dir.iterdir()):
        group_dir.rmdir()


def save_group(data, model, group_id, group_dir, plot_info):
    Q, origin = world_transform(data, model)
    objects = []
    for oid, X in model.points.items():
        world = X @ Q.T + origin
        observations = [dict(frame_id=f, local_object_id=data["local_ids"][f, oid],
                             raw_corners_px=data["raw"][f, oid], visibility=data["visibility"][f, oid],
                             used_in_geometry=f in model.views[oid])
                        for f in model.R if (f, oid) in data["raw"]]
        objects.append(dict(object_id=oid, **data["objects"][oid], corners_local_m=X, corners_world_m=world,
                            centre_local_m=X.mean(axis=0), centre_world_m=world.mean(axis=0),
                            measurements=dimensions(X), observations=observations))
    save_json(group_dir / "map.json", dict(group_id=group_id, coordinate_reference="geometry.json",
                                           calibration_reference="../experiment.json", objects=objects))
    metrics = model.info["metrics"]
    save_json(group_dir / "geometry.json", dict(
        group_id=group_id, implementation_sha256=data["implementation_sha256"],
        reference_frame=model.reference, selected_candidate=model.info["candidate_id"],
        selected_anchor_ids=model.info["anchor_ids"], initialization=model.info["initialization"],
        initialization_attempts=model.info["initialization_attempts"],
        cameras=[dict(frame_id=f, R_local_to_camera=model.R[f], C_local_m=model.C[f],
                      C_world_m=Q @ model.C[f] + origin, sensor_C_world_m=data["sensor_C"][f]) for f in model.R],
        local_to_world=dict(rotation=Q, origin_world_m=origin, status="provisional_reference_sensor_pose"),
        initial_pair_estimates=model.info["initial_pair_reports"],
        initial_rotation_cycles=model.info["initial_rotation_cycles"],
        metrics=metrics,
        excluded_common_object_ids=sorted(set(model.info["common_object_ids"]) - set(model.points)),
        display_alignment=plot_info))


# ============================================================================
# 6. MAIN LOOP -- one catalogue group is processed independently at a time.
# ============================================================================
def main():
    data = load_inputs()
    if data is None:
        return
    selected = data["groups"] if GROUP_IDS is None else data["groups"][data["groups"].group_id.isin(GROUP_IDS)]
    if GROUP_IDS is not None and set(GROUP_IDS) - set(selected.group_id):
        print("Unknown GROUP_IDS:", sorted(set(GROUP_IDS) - set(selected.group_id)))
        return
    prepared = prepare_output(data)
    if prepared is None:
        return
    summary, candidate_rows = prepared

    for group in selected.itertuples():
        started = time.perf_counter()
        gid, frames = group.group_id, list(group.frame_ids)
        directory = OUTPUT_DIR / gid
        candidate_rows = [r for r in candidate_rows if r["group_id"] != gid]
        remove_previous_group_outputs(directory)
        summary[gid] = dict(group_id=gid, status="processing", n_frames=len(frames))
        save_summaries(summary, candidate_rows)
        valid_frames = (2 <= len(frames) <= 4 and len(set(f.split("_", 1)[0] for f in frames)) == len(frames)
                        and all(f in data["sensor_C"] for f in frames))
        common = common_objects(data, frames) if valid_frames else []
        print(f"\n{gid}: {len(frames)} frames, {len(common)} fully visible common objects")
        candidates = select_candidates(data, frames, common) if len(common) >= CFG["min_group_objects"] else []
        winner, best_score = None, None

        # Reconstruct each anchor hypothesis, then compare its FINAL accepted geometry.
        for candidate_id, anchors in candidates:
            try:
                model, report = reconstruct_candidate(data, frames, common, candidate_id, anchors)
            except (np.linalg.LinAlgError, cv2.error, FloatingPointError) as error:
                # One numerical degeneracy should not stop the remaining 358 groups.
                model = None
                report = dict(candidate_id=candidate_id, anchor_ids="|".join(anchors), status="numerical_failure")
                print(f"  {candidate_id}: {type(error).__name__}: {error}")
            report.update(group_id=gid, selected=False)
            candidate_rows.append(report)
            print(f"  {candidate_id}: {report['status']}; accepted objects={report.get('n_objects', 0)}; "
                  f"initial cycle max={report.get('initial_cycle_max_deg')} deg")
            if model is None:
                continue
            m = model.info["metrics"]
            score = (m["n_objects"], m["n_corner_observations"], -m["rmse_px"])
            if best_score is None or score > best_score:
                winner, best_score = model, score

        if winner is None:
            summary[gid].update(status="failed", failure_stage="input_support" if not candidates else "no_valid_candidate",
                                n_common_objects=len(common), seconds=round(time.perf_counter()-started, 2))
            print("  No accepted reconstruction; saved a summary row only.")
        else:
            # One camera pose and one 3D estimate per object, not one map per pair.
            directory.mkdir(parents=True, exist_ok=True)
            plot_info = save_plots(data, winner, gid, directory, CFG)
            save_group(data, winner, gid, directory, plot_info)
            m = winner.info["metrics"]
            summary[gid].update(status="accepted", selected_candidate=winner.info["candidate_id"],
                                n_common_objects=len(common), n_objects=m["n_objects"],
                                n_corner_observations=m["n_corner_observations"], rmse_px=m["rmse_px"],
                                median_error_px=m["median_error_px"], p95_error_px=m["p95_error_px"],
                                max_error_px=m["max_error_px"], baseline_rmse_m=m["baseline_rmse_m"],
                                seconds=round(time.perf_counter()-started, 2))
            for row in candidate_rows:
                if row["group_id"] == gid:
                    row["selected"] = row["candidate_id"] == winner.info["candidate_id"]
            print(f"  Saved {gid}: {m['n_objects']}/{len(common)} objects; RMSE={m['rmse_px']:.3f} px.")
        save_summaries(summary, candidate_rows)
    print(f"\nFinished. Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
