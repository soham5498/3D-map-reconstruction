"""Stage 2: combine accepted Stage 1 groups into consistent component maps.

Run from the same working directory used for Stage 1: python stage2.py
This file is self-contained; it does not import main.py, geometry.py or plots.py.
Dependencies: numpy, scipy, pandas, opencv-python, matplotlib.

Inputs are the Stage 1 results and the SAME original correspondence CSV.
Calibration and sensor camera centres are read from Stage 1. Metric scale is
retained: group alignment is rigid, with no estimated scale factor.

Flow: load -> overlap eligibility -> gated alignment -> local BA/screen ->
selective retries -> periodic global BA -> recover -> final BA/export.
Progress snapshots preserve geometry but are not automatic resume checkpoints.
Retry caps and alignment gates trade some possible coverage for bounded work.
All quality decisions below are part of the reconstruction methodology.
"""

from pathlib import Path
from dataclasses import dataclass, field
from itertools import combinations
from copy import deepcopy
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================================================
# 1. SETTINGS -- relative to your current working directory.
# ============================================================================
STAGE1_DIR = Path("stage1_results")
CORRESPONDENCE_CSV = Path("../DHBW_correspondences/correspondences.csv")
OUTPUT_DIR = Path("stage2_results")
GROUP_IDS = None                         # None = all accepted Stage 1 groups.

CFG = dict(
    global_ba_every=10,
    max_group_attempts=3,
    retry_point_change_m=0.05,
    retry_camera_change_m=0.05,
    retry_rotation_change_deg=0.5,
    alignment_max_median_m=0.5,
    min_alignment_inlier_fraction=0.5,
    min_shared_objects=3,
    min_shared_objects_with_camera=2,
    alignment_inlier_m=0.15,             # Ranks proposals; not an immediate merge veto.
    alignment_trials=40,
    max_point_alignment_proposals=3,
    max_camera_alignment_proposals=3,
    random_seed=42,
    pixel_sigma_px=1.0,
    baseline_sigma_m=0.02,               # Provisional weight, not known sensor accuracy.
    robust_scale_px=2.0,
    max_corner_error_px=3.0,
    max_object_rmse_px=2.0,
    min_ray_angle_deg=0.5,
    min_component_objects=4,
    min_camera_objects=3,
    max_ba_nfev=300,
    ba_tolerance=1e-6,
    sparse_ba_above_variables=180,
    screen_rounds=4,
    min_old_observation_fraction=0.95,   # Every previously accepted object must survive.
    min_incoming_object_fraction=0.80,
    min_incoming_observation_fraction=0.75,
    old_rmse_ratio=1.25,
    old_rmse_allowance_px=0.15,
    baseline_rmse_limit_m=0.10,          # Baseline deterioration is checked against both inputs.
    baseline_rmse_allowance_m=0.02,
    recover_objects=True,
    recovery_initial_error_px=4.0,
    recovery_seed_pairs=20,
    max_point_nfev=80,
    plot_frame="0052_POS_105",           # Otherwise use the best-supported frame.
    plot_dpi=180,
    plot_elevation_deg=18.0,
    plot_azimuth_deg=-65.0,
    label_map_objects=False,
    illustrative_reference_panel=False,
    vp_trials=300,
    vp_min_edge_px=10.0,
    vp_inlier_angle_deg=1.0,
    vp_min_inlier_fraction=0.5,
    vp_max_orthogonality_error_deg=10.0,
)

CORNERS = ("TL", "TR", "BR", "BL")
COLORS = {"frame": "#E69F00", "glass": "#56B4E9", "darkening": "#9146B6",
          "windows": "#0072B2", "doors": "#009E73", "other": "#666666"}


@dataclass
class Component:
    seed: str
    groups: set
    reference: str
    R: dict                            # X_camera = R @ (X_local - C).
    C: dict
    points: dict                       # object_id -> four corners, TL/TR/BR/BL.
    views: dict                        # object_id -> SET of unique frame IDs.
    baselines: set                     # SET of sorted frame pairs.
    world_Q: np.ndarray                # X_world = Q @ X_local + origin.
    world_origin: np.ndarray
    sources: dict                      # Original accepted group IDs per object.
    recovered: set = field(default_factory=set)
    info: dict = field(default_factory=dict)


# ============================================================================
# 2. READ THE STAGE 1 RESULTS AND ORIGINAL ANNOTATIONS.
# ============================================================================
def load_inputs():
    experiment = json.loads((STAGE1_DIR / "experiment.json").read_text())
    calibration = experiment["calibration"]
    data = dict(K=np.asarray(calibration["camera_matrix"], float),
                dist=np.asarray(calibration["dist"], float),
                image_size=tuple(calibration["image_size"]), sensor_C={}, objects={},
                raw={}, visibility={}, local_ids={}, uv={}, experiment=experiment)
    summary = pd.read_csv(STAGE1_DIR / "group_summary.csv", keep_default_na=False)
    accepted = summary[summary.status.eq("accepted")]
    if GROUP_IDS is not None:
        accepted = accepted[accepted.group_id.isin(GROUP_IDS)]
    models = {}
    for gid in sorted(accepted.group_id):
        geometry = json.loads((STAGE1_DIR / gid / "geometry.json").read_text())
        saved_map = json.loads((STAGE1_DIR / gid / "map.json").read_text())
        R = {c["frame_id"]: np.asarray(c["R_local_to_camera"], float) for c in geometry["cameras"]}
        C = {c["frame_id"]: np.asarray(c["C_local_m"], float) for c in geometry["cameras"]}
        data["sensor_C"].update({c["frame_id"]: np.asarray(c["sensor_C_world_m"], float)
                                  for c in geometry["cameras"]})
        points, views = {}, {}
        for obj in saved_map["objects"]:
            oid = obj["object_id"]
            points[oid] = np.asarray(obj["corners_local_m"], float)
            views[oid] = {obs["frame_id"] for obs in obj["observations"] if obs["used_in_geometry"]}
        edges = {tuple(sorted((e["frame_a"], e["frame_b"]))) for e in geometry["metrics"]["baselines"]}
        transform = geometry["local_to_world"]
        models[gid] = Component(gid, {gid}, geometry["reference_frame"], R, C, points, views, edges,
                                np.asarray(transform["rotation"], float), np.asarray(transform["origin_world_m"], float),
                                {oid: {gid} for oid in points}, info={"final_refinement": "stage1_geometry_retained"})

    # Read the original universe of objects, including ones omitted from all maps.
    table = pd.read_csv(CORRESPONDENCE_CSV, dtype=str, keep_default_na=False)
    building = experiment["settings"].get("building", "DHBW")
    if "building" in table:
        table = table[table.building.eq(building)]
    meta_columns = [c for c in ("building", "class_id", "class_name", "plane_id") if c in table]
    data["objects"] = table.set_index("object_id")[meta_columns].to_dict("index")
    width, height = data["image_size"]
    for f in sorted(data["sensor_C"]):
        columns = [f"{f}_{corner}_{a}" for corner in CORNERS for a in ("x", "y", "v")]
        values = table[columns].apply(pd.to_numeric, errors="coerce").to_numpy().reshape(-1, 4, 3)
        for oid, lid, v in zip(table.object_id, table[f"{f}_local_object_id"], values):
            if not lid:
                continue
            data["raw"][f, oid], data["visibility"][f, oid] = v[:, :2], v[:, 2]
            data["local_ids"][f, oid] = lid
            if (np.all(v[:, 2] == 2) and np.isfinite(v).all() and np.all(v[:, :2] >= 0)
                    and np.all(v[:, 0] < width) and np.all(v[:, 1] < height)):
                data["uv"][f, oid] = cv2.undistortPoints(
                    v[:, :2].copy().reshape(-1, 1, 2), data["K"], data["dist"], P=data["K"]).reshape(4, 2)
    print(f"Loaded {len(models)} accepted groups and {len(data['objects'])} original object IDs.")
    return data, models


# ============================================================================
# 3. PROJECTION, SUPPORT AND QUALITY METRICS.
# ============================================================================
def project(X, R, C, K):
    camera = (X - C) @ R.T
    image = camera @ K.T
    z = image[:, 2:3]
    z = np.where(np.abs(z) < 1e-9, np.where(z < 0, -1e-9, 1e-9), z)
    return image[:, :2] / z, camera[:, 2]


def dimensions(X):
    top, bottom = np.linalg.norm(X[1]-X[0]), np.linalg.norm(X[2]-X[3])
    left, right = np.linalg.norm(X[3]-X[0]), np.linalg.norm(X[2]-X[1])
    return dict(width_m=float((top+bottom)/2), height_m=float((left+right)/2),
                top_m=float(top), bottom_m=float(bottom), left_m=float(left), right_m=float(right))


def observation_keys(model):
    return {(f, o, k) for o in model.points for f in model.views[o] for k in range(4)}


def errors_by_observation(data, model):
    errors = {}
    for o, X in model.points.items():
        for f in model.views[o]:
            pixels, _ = project(X, model.R[f], model.C[f], data["K"])
            errors.update({(f, o, k): float(e) for k, e in enumerate(np.linalg.norm(pixels-data["uv"][f, o], axis=1))})
    return errors


def rmse(values):
    return float(np.sqrt(np.mean(np.square(list(values)))))


def ray_angle(X, centres):
    best = np.zeros(4)
    for a, b in combinations(centres, 2):
        ra, rb = X-a, X-b
        cosine = np.sum(ra*rb, axis=1) / np.maximum(np.linalg.norm(ra, axis=1)*np.linalg.norm(rb, axis=1), 1e-12)
        best = np.maximum(best, np.degrees(np.arccos(np.clip(np.abs(cosine), 0, 1))))
    return float(best.min())


def object_quality(data, model, oid, X=None, views=None):
    X = model.points[oid] if X is None else X
    views = model.views[oid] if views is None else views
    errors, depths = [], []
    for f in sorted(views):
        p, z = project(X, model.R[f], model.C[f], data["K"])
        errors.extend(np.linalg.norm(p-data["uv"][f, oid], axis=1))
        depths.extend(z)
    return dict(n_views=len(views), rmse_px=rmse(errors), max_error_px=float(np.max(errors)),
                positive_depth=bool(np.all(np.asarray(depths) > 1e-6)),
                min_ray_angle_deg=ray_angle(X, [model.C[f] for f in sorted(views)]))


def baseline_records(data, model):
    records = []
    for a, b in sorted(model.baselines):
        target = float(np.linalg.norm(data["sensor_C"][a]-data["sensor_C"][b]))
        fitted = float(np.linalg.norm(model.C[a]-model.C[b]))
        records.append(dict(frame_a=a, frame_b=b, sensor_m=target, fitted_m=fitted, residual_m=fitted-target))
    return records


def enough_support(model):
    if len(model.points) < CFG["min_component_objects"]:
        return False
    if any(sum(f in views for views in model.views.values()) < CFG["min_camera_objects"] for f in model.R):
        return False
    # Connectivity must come from shared visual observations, not only sensor priors.
    reached = {model.reference}
    for _ in model.R:
        before = len(reached)
        for views in model.views.values():
            if reached.intersection(views):
                reached.update(views)
        if len(reached) == before:
            break
    return reached == set(model.R)


def coverage(model):
    X = np.concatenate(list(model.points.values()))
    _, _, V = np.linalg.svd(X-X.mean(axis=0), full_matrices=False)
    flat = (X-X.mean(axis=0)) @ V[:2].T
    return float(np.prod(np.percentile(flat, 95, axis=0)-np.percentile(flat, 5, axis=0)))


def quality(data, model):
    errors = errors_by_observation(data, model)
    object_metrics = {o: object_quality(data, model, o) for o in model.points}
    baselines = baseline_records(data, model)
    world_centres = {f: model.world_Q @ model.C[f]+model.world_origin for f in model.C}
    sensor_residuals = [dict(frame_id=f, delta_xyz_m=world_centres[f]-data["sensor_C"][f],
                             distance_m=float(np.linalg.norm(world_centres[f]-data["sensor_C"][f]))) for f in sorted(model.C)]
    return dict(n_groups=len(model.groups), n_frames=len(model.R), n_objects=len(model.points),
                n_corners=4*len(model.points), n_corner_observations=len(errors),
                rmse_px=rmse(errors.values()), median_error_px=float(np.median(list(errors.values()))),
                p95_error_px=float(np.percentile(list(errors.values()), 95)), max_error_px=max(errors.values()),
                min_ray_angle_deg=min(q["min_ray_angle_deg"] for q in object_metrics.values()),
                positive_depth=all(q["positive_depth"] for q in object_metrics.values()),
                coverage_m2=coverage(model), baseline_rmse_m=rmse(b["residual_m"] for b in baselines),
                sensor_position_rmse_m=rmse(e["distance_m"] for e in sensor_residuals),
                baselines=baselines, sensor_position_disagreement=sensor_residuals, objects=object_metrics)


def component_rank(data, model):
    m = quality(data, model)
    return (m["n_objects"], m["coverage_m2"], -m["rmse_px"], m["min_ray_angle_deg"])


# ============================================================================
# 4. ALIGNMENT PROPOSALS -- CORRESPONDING OBJECT CORNERS OR A SHARED CAMERA.
# ============================================================================
def overlap(a, b):
    return sorted(set(a.points) & set(b.points)), sorted(set(a.R) & set(b.R))


def can_join(a, b):
    objects, cameras = overlap(a, b)
    return (len(objects) >= CFG["min_shared_objects"] or
            (len(objects) >= CFG["min_shared_objects_with_camera"] and bool(cameras)))


def rigid_fit(source, target):
    """Find Q,d for target ~= Q @ source + d; reflection and scale are excluded."""
    p, q = source.mean(axis=0), target.mean(axis=0)
    U, singular, Vt = np.linalg.svd((source-p).T @ (target-q))
    if singular[0] <= 1e-12 or singular[1] < 1e-8*singular[0]:
        return None                     # Collinear support cannot constrain this fit.
    sign = np.diag([1., 1., np.linalg.det(Vt.T @ U.T)])
    Q = Vt.T @ sign @ U.T
    return Q, q-Q @ p


def alignment_proposals(a, b):
    shared, cameras = overlap(a, b)
    A, B = np.array([a.points[o] for o in shared]), np.array([b.points[o] for o in shared])
    proposals = []

    def add(transform, name):
        if transform is None:
            return
        Q, d = transform
        if any(np.allclose(Q, old["Q"], atol=1e-6) and np.allclose(d, old["d"], atol=1e-6) for old in proposals):
            return
        residuals = np.linalg.norm(B @ Q.T+d-A, axis=2)
        object_rms = np.sqrt(np.mean(residuals**2, axis=1))
        proposals.append(dict(Q=Q, d=d, method=name, n_inlier_objects=int(np.sum(object_rms <= CFG["alignment_inlier_m"])),
                              shared_corner_rmse_m=rmse(residuals.ravel()), median_object_disagreement_m=float(np.median(object_rms))))

    add(rigid_fit(B.reshape(-1, 3), A.reshape(-1, 3)), "all_shared_corners")
    rng = np.random.default_rng(CFG["random_seed"])
    for _ in range(CFG["alignment_trials"]):
        subset = rng.choice(len(shared), min(3, len(shared)), replace=False)
        fit = rigid_fit(B[subset].reshape(-1, 3), A[subset].reshape(-1, 3))
        if fit is None:
            continue
        Q, d = fit
        inliers = np.sqrt(np.mean(np.sum((B @ Q.T+d-A)**2, axis=2), axis=1)) <= CFG["alignment_inlier_m"]
        if inliers.sum() >= 2:
            fit = rigid_fit(B[inliers].reshape(-1, 3), A[inliers].reshape(-1, 3))
        add(fit, "shared_object_subset")
    proposals.sort(key=lambda p: (-p["n_inlier_objects"], p["median_object_disagreement_m"]))
    proposals = proposals[:CFG["max_point_alignment_proposals"]]
    for f in sorted(cameras, key=lambda f: -sum(f in a.views[o] and f in b.views[o] for o in shared))[:CFG["max_camera_alignment_proposals"]]:
        Q = a.R[f].T @ b.R[f]
        add((Q, a.C[f]-Q @ b.C[f]), "shared_camera:"+f)
    return proposals


def combine(a, b, proposal):
    """Build a copy. The accepted map is never changed by an unsuccessful trial."""
    Q, d = proposal["Q"], proposal["d"]
    trial = deepcopy(a)
    for f in b.R:
        if f not in trial.R:
            trial.R[f], trial.C[f] = b.R[f] @ Q.T, Q @ b.C[f]+d
    for o, X in b.points.items():
        if o not in trial.points:
            trial.points[o], trial.views[o] = X @ Q.T+d, set(b.views[o])
        else:
            trial.views[o].update(b.views[o])  # Existing estimate initializes the shared point.
        trial.sources.setdefault(o, set()).update(b.sources[o])
    trial.groups.update(b.groups)
    trial.baselines.update(b.baselines)
    trial.recovered.update(b.recovered)
    return trial


# ============================================================================
# 5. JOINT BUNDLE ADJUSTMENT -- UNIQUE OBSERVATIONS AND ONE FIXED REFERENCE.
# ============================================================================
def bundle_adjust(data, model, active_frames=None, active_objects=None):
    frames = [model.reference] + sorted(set(model.R)-{model.reference})
    ids = sorted(model.points)
    camera_index, object_index = {f: i for i, f in enumerate(frames)}, {o: i for i, o in enumerate(ids)}
    records = [(f, o, k) for o in ids for f in sorted(model.views[o]) for k in range(4)]
    camera_ids = np.array([camera_index[f] for f, o, k in records])
    point_ids = np.array([4*object_index[o]+k for f, o, k in records])
    observed = np.array([data["uv"][f, o][k] for f, o, k in records])
    edges = sorted(model.baselines)
    edge_a = np.array([camera_index[a] for a, b in edges])
    edge_b = np.array([camera_index[b] for a, b in edges])
    target_lengths = np.array([np.linalg.norm(data["sensor_C"][a]-data["sensor_C"][b]) for a, b in edges])
    camera_end = 6*(len(frames)-1)
    x0 = np.r_[np.concatenate([np.r_[Rotation.from_matrix(model.R[f]).as_rotvec(), model.C[f]] for f in frames[1:]]),
               np.concatenate([model.points[o].ravel() for o in ids])]
    n_pixels = 2*len(records)

    def unpack(x):
        poses = x[:camera_end].reshape(-1, 6)
        R = np.concatenate([model.R[model.reference][None], Rotation.from_rotvec(poses[:, :3]).as_matrix()])
        C = np.vstack([model.C[model.reference], poses[:, 3:]])
        return R, C, x[camera_end:].reshape(-1, 3)

    def residual(x):
        R, C, X = unpack(x)
        camera_points = np.einsum("nij,nj->ni", R[camera_ids], X[point_ids]-C[camera_ids])
        image = camera_points @ data["K"].T
        z = image[:, 2:3]
        z = np.where(np.abs(z) < 1e-9, np.where(z < 0, -1e-9, 1e-9), z)
        pixels = ((image[:, :2]/z-observed)/CFG["pixel_sigma_px"]).ravel()
        baselines = (np.linalg.norm(C[edge_a]-C[edge_b], axis=1)-target_lengths)/CFG["baseline_sigma_m"]
        return np.r_[pixels, baselines]

    def loss(z):
        rho = np.vstack([z.copy(), np.ones_like(z), np.zeros_like(z)])
        q = 1+z[:n_pixels]
        rho[:, :n_pixels] = [2*(np.sqrt(q)-1), q**-0.5, -0.5*q**-1.5]
        return rho                       # Robust pixels, quadratic baseline priors.

    sparsity = lil_matrix((n_pixels+len(edges), len(x0)), dtype=int)
    for i, (c, p) in enumerate(zip(camera_ids, point_ids)):
        sparsity[2*i:2*i+2, camera_end+3*p:camera_end+3*p+3] = 1
        if c:
            sparsity[2*i:2*i+2, 6*(c-1):6*c] = 1
    for i, (a, b) in enumerate(zip(edge_a, edge_b)):
        for c in (a, b):
            if c:
                sparsity[n_pixels+i, 6*(c-1)+3:6*c] = 1
    # Optimize only affected variables; every observation still participates.
    active = np.ones(len(x0), dtype=bool)
    if active_frames is not None:
        active[:] = False
        for i, f in enumerate(frames[1:]):
            active[6*i:6*i+6] = f in active_frames
        for i, o in enumerate(ids):
            active[camera_end+12*i:camera_end+12*i+12] = o in active_objects
    indices = np.flatnonzero(active)
    def expand(values):
        full = x0.copy()
        full[indices] = values
        return full
    if not len(indices):
        model.info["last_optimizer"] = dict(converged=True, nfev=0, optimality=0.)
        return True
    result = least_squares(lambda values: residual(expand(values)), x0[indices], loss=loss, f_scale=CFG["robust_scale_px"]/CFG["pixel_sigma_px"],
                           x_scale="jac", jac_sparsity=sparsity.tocsr()[:, indices] if len(indices) > CFG["sparse_ba_above_variables"] else None,
                           max_nfev=CFG["max_ba_nfev"], ftol=CFG["ba_tolerance"], xtol=CFG["ba_tolerance"], gtol=CFG["ba_tolerance"])
    R, C, X = unpack(expand(result.x))
    model.R = {f: R[i] for i, f in enumerate(frames)}
    model.C = {f: C[i] for i, f in enumerate(frames)}
    model.points = {o: X[4*i:4*i+4] for i, o in enumerate(ids)}
    model.info["last_optimizer"] = dict(converged=bool(result.success), nfev=int(result.nfev), optimality=float(result.optimality))
    return bool(result.success and np.isfinite(result.x).all())


def refine(data, model, active_frames=None, active_objects=None):
    before = errors_by_observation(data, model)
    for _ in range(CFG["screen_rounds"]):
        if not enough_support(model):
            return False, "insufficient_connected_support"
        if not bundle_adjust(data, model, active_frames, active_objects):
            return False, "ba_not_converged"
        changed = False
        for o in list(model.points):
            kept = set()
            for f in model.views[o]:
                p, z = project(model.points[o], model.R[f], model.C[f], data["K"])
                if np.all(z > 1e-6) and np.max(np.linalg.norm(p-data["uv"][f, o], axis=1)) <= CFG["max_corner_error_px"]:
                    kept.add(f)
            changed |= kept != model.views[o]
            model.views[o] = kept
            q = object_quality(data, model, o) if len({f.split("_", 1)[0] for f in kept}) >= 2 else None
            if q is None or q["rmse_px"] > CFG["max_object_rmse_px"] or q["min_ray_angle_deg"] < CFG["min_ray_angle_deg"]:
                del model.points[o], model.views[o]
                changed = True
        if not changed and enough_support(model):
            after = errors_by_observation(data, model)
            model.info["last_refinement"] = dict(before_rmse_px=rmse(before[k] for k in after),
                                                   after_rmse_px=rmse(after.values()),
                                                   n_corner_observations=len(after), same_observations=True)
            return True, "accepted"
    return False, "screening_not_stable"


def protect_existing(data, old, trial):
    if not set(old.points).issubset(trial.points):
        return False, "would_remove_existing_object"
    old_errors, new_errors = errors_by_observation(data, old), errors_by_observation(data, trial)
    retained = set(old_errors) & set(new_errors)
    if len(retained) < CFG["min_old_observation_fraction"]*len(old_errors):
        return False, "would_remove_too_many_existing_observations"
    before, after = rmse(old_errors[k] for k in retained), rmse(new_errors[k] for k in retained)
    limit = max(before*CFG["old_rmse_ratio"], before+CFG["old_rmse_allowance_px"])
    if after > limit:
        return False, "existing_reprojection_quality_degraded"
    return True, "accepted"


def baseline_check(data, trial, inputs):
    previous = max(rmse(e["residual_m"] for e in baseline_records(data, m)) for m in inputs)
    limit = max(CFG["baseline_rmse_limit_m"], previous+CFG["baseline_rmse_allowance_m"])
    return rmse(e["residual_m"] for e in baseline_records(data, trial)) <= limit


def try_join(data, current, incoming, log, local=True):
    shared, cameras = overlap(current, incoming)
    print(f"  Trying {incoming.seed}: {len(shared)} shared objects, {len(cameras)} shared frames")
    for proposal in alignment_proposals(current, incoming):
        started = time.perf_counter()
        # A cheap geometric gate before any nonlinear optimization.
        if (proposal["median_object_disagreement_m"] > CFG["alignment_max_median_m"] or
                proposal["n_inlier_objects"] < CFG["min_alignment_inlier_fraction"] * len(shared)):
            log.append(dict(target_seed=current.seed, incoming_group_ids="|".join(sorted(incoming.groups)),
                            method=proposal["method"], status="alignment_gate_failed", seconds=0.))
            print("    Alignment gate failed; BA skipped", flush=True)
            continue
        trial = combine(current, incoming, proposal)
        # Existing cameras anchor the merge; incoming objects may move, constrained
        # by ALL their old and new observations. Global BA later releases cameras.
        active_frames = set(incoming.R) - set(current.R) if local else None
        active_objects = set(incoming.points) if local else None
        ok, status = refine(data, trial, active_frames, active_objects)
        if ok:
            ok, status = protect_existing(data, current, trial)
        if ok and len(set(incoming.points) & set(trial.points)) < CFG["min_incoming_object_fraction"]*len(incoming.points):
            ok, status = False, "insufficient_incoming_objects_retained"
        if ok and len(observation_keys(incoming) & observation_keys(trial)) < CFG["min_incoming_observation_fraction"]*len(observation_keys(incoming)):
            ok, status = False, "insufficient_incoming_observations_retained"
        if ok and not baseline_check(data, trial, [current, incoming]):
            ok, status = False, "baseline_quality_degraded"
        row = dict(target_seed=current.seed, incoming_group_ids="|".join(sorted(incoming.groups)),
                   shared_objects=len(shared), shared_frames=len(cameras), method=proposal["method"],
                   initial_shared_corner_rmse_m=proposal["shared_corner_rmse_m"],
                   initial_inlier_objects=proposal["n_inlier_objects"],
                   ba_converged=trial.info.get("last_optimizer", {}).get("converged", False),
                   status=status, seconds=round(time.perf_counter()-started, 3))
        log.append(row)
        print(f"    {proposal['method']}: {status} ({row['seconds']:.1f} s)", flush=True)
        if ok:
            row["final_rmse_px"] = rmse(errors_by_observation(data, trial).values())
            row["before_ba_rmse_px"] = trial.info["last_refinement"]["before_rmse_px"]
            row["final_objects"] = len(trial.points)
            trial.info["final_refinement"] = "local_ba_accepted" if local else "joint_ba_accepted"
            return trial
    print(f"  Deferred {incoming.seed}; previous component retained.")
    return None


# ============================================================================
# 6. RETRY EXCLUDED OBJECTS / OBSERVATIONS USING THE REGISTERED CAMERAS.
# ============================================================================
def triangulate(data, model, oid, views):
    matrices = {f: data["K"] @ np.c_[model.R[f], -model.R[f] @ model.C[f]] for f in views}
    points = []
    for k in range(4):
        rows = []
        for f in views:
            u, v = data["uv"][f, oid][k]
            P = matrices[f]
            rows.extend([u*P[2]-P[0], v*P[2]-P[1]])
        _, _, V = np.linalg.svd(rows)
        if abs(V[-1, 3]) < 1e-12:
            return None
        points.append(V[-1, :3]/V[-1, 3])
    return np.asarray(points)


def supported_views(data, model, oid, X, available, threshold):
    supported = set()
    for f in available:
        pixels, z = project(X, model.R[f], model.C[f], data["K"])
        if np.all(z > 1e-6) and np.max(np.linalg.norm(pixels-data["uv"][f, oid], axis=1)) <= threshold:
            supported.add(f)
    return supported


def reconstruct_missing_object(data, model, oid, available):
    pairs = sorted(combinations(available, 2), key=lambda fs: -np.linalg.norm(model.C[fs[0]]-model.C[fs[1]]))
    seeds = [tuple(available)] + pairs[:CFG["recovery_seed_pairs"]]
    best, best_score = None, None
    for seed in seeds:
        X = triangulate(data, model, oid, seed)
        if X is None or not np.isfinite(X).all():
            continue
        used = supported_views(data, model, oid, X, available, CFG["recovery_initial_error_px"])
        if len({f.split("_", 1)[0] for f in used}) < 2:
            continue
        ordered = sorted(used)
        def residual(x):
            return np.concatenate([(project(x.reshape(4, 3), model.R[f], model.C[f], data["K"])[0]-data["uv"][f, oid]).ravel()
                                   for f in ordered])
        fit = least_squares(residual, X.ravel(), loss="soft_l1", f_scale=CFG["robust_scale_px"], max_nfev=CFG["max_point_nfev"])
        X = fit.x.reshape(4, 3)
        used = supported_views(data, model, oid, X, available, CFG["max_corner_error_px"])
        if len({f.split("_", 1)[0] for f in used}) < 2:
            continue
        q = object_quality(data, model, oid, X, used)
        score = (len(used), -q["rmse_px"])
        if q["rmse_px"] <= CFG["max_object_rmse_px"] and q["min_ray_angle_deg"] >= CFG["min_ray_angle_deg"]:
            if best_score is None or score > best_score:
                best, best_score = (X, used), score
            if len(used) == len(available):
                break
    return best


def recover(data, current):
    if not CFG["recover_objects"]:
        return current, False
    trial = deepcopy(current)
    for oid in sorted(data["objects"]):
        available = sorted(f for f in trial.R if (f, oid) in data["uv"])
        if len(available) < 2:
            continue
        if oid in trial.points:
            trial.views[oid].update(supported_views(data, trial, oid, trial.points[oid], available, CFG["recovery_initial_error_px"]))
        else:
            fit = reconstruct_missing_object(data, trial, oid, available)
            if fit is not None:
                trial.points[oid], trial.views[oid] = fit
                trial.sources[oid] = set()
                trial.recovered.add(oid)
    old_keys = observation_keys(current)
    if observation_keys(trial) == old_keys:
        return current, False
    ok, _ = refine(data, trial)
    if ok:
        ok, _ = protect_existing(data, current, trial)
    # Recovery is monotonic: retain old observations and add at least one new one.
    new_keys = observation_keys(trial)
    if ok and old_keys < new_keys and baseline_check(data, trial, [current]):
        trial.info["final_refinement"] = "recovery_ba_accepted"
        print(f"  Recovery: +{len(trial.points)-len(current.points)} objects, +{len(new_keys-old_keys)} corner observations.")
        return trial, True
    return current, False


# ============================================================================
# 7. COMPONENT CONSTRUCTION -- RETRY DEFERRED GROUPS AFTER EACH IMPROVEMENT.
# ============================================================================
def retry_state(current, incoming):
    objects, frames = overlap(current, incoming)
    return dict(objects=objects, frames=frames,
                points=np.array([current.points[o] for o in objects]),
                centres=np.array([current.C[f] for f in frames]),
                rotations=np.array([current.R[f] for f in frames]),
                views={o: frozenset(current.views[o]) for o in objects})


def changed_support(old, new):
    if old["objects"] != new["objects"] or old["frames"] != new["frames"] or old["views"] != new["views"]:
        return True
    if np.max(np.linalg.norm(new["points"]-old["points"], axis=-1), initial=0) >= CFG["retry_point_change_m"]:
        return True
    if np.max(np.linalg.norm(new["centres"]-old["centres"], axis=-1), initial=0) >= CFG["retry_camera_change_m"]:
        return True
    return any(np.degrees(Rotation.from_matrix(a @ b.T).magnitude()) >= CFG["retry_rotation_change_deg"]
               for a, b in zip(new["rotations"], old["rotations"]))


def global_refine(data, current):
    print(f"  Global BA: {len(current.groups)} groups", flush=True)
    trial = deepcopy(current)
    ok, status = refine(data, trial)
    if ok:
        ok, status = protect_existing(data, current, trial)
    if ok and baseline_check(data, trial, [current]):
        trial.info["final_refinement"] = "global_ba_accepted"
        return trial
    print(f"  Global update not accepted ({status}); retaining previous map", flush=True)
    return current


def build_components(data, groups):
    pending, components, log = dict(groups), [], []
    while pending:
        seed = max(pending, key=lambda g: component_rank(data, pending[g]))
        current = deepcopy(pending.pop(seed))
        started = time.perf_counter()
        print(f"\nSeed {seed}: {len(current.points)} objects")
        attempts, states = {}, {}
        since_global = 0
        pass_number = 0
        save_progress(current, log)
        while True:
            pass_number += 1
            candidates = [g for g in pending if can_join(current, pending[g])]
            candidates = [g for g in candidates if attempts.get(g, 0) < CFG["max_group_attempts"]
                          and (g not in states or changed_support(states[g], retry_state(current, pending[g])))]
            candidates.sort(key=lambda g: (len(set(pending[g].points)-set(current.points)),
                                            len(set(pending[g].R)-set(current.R)),
                                            len(overlap(current, pending[g])[0])), reverse=True)
            print(f"  Pass {pass_number}: {len(candidates)} eligible groups; {len(pending)} pending", flush=True)
            joined = False
            for gid in candidates:
                states[gid] = retry_state(current, pending[gid])
                attempts[gid] = attempts.get(gid, 0) + 1
                trial = try_join(data, current, pending[gid], log)
                if trial is not None:
                    current = trial
                    pending.pop(gid)
                    joined = True
                    print(f"  Joined {gid}: {len(current.groups)} groups, {len(current.points)} unique objects")
                    since_global += 1
                    if since_global >= CFG["global_ba_every"]:
                        current = global_refine(data, current)
                        since_global = 0
                    save_progress(current, log)
                # Finish this pass before retrying any deferred candidate.
            if joined:
                continue
            if since_global:
                current = global_refine(data, current)
                since_global = 0
                save_progress(current, log)
                continue  # Reconsider candidates only if their shared geometry changed.
            print("  Checking recoverable objects and observations...", flush=True)
            current, recovered = recover(data, current)
            save_progress(current, log)
            if not recovered:
                break
        current.info["seconds"] = time.perf_counter()-started
        components.append(current)
        save_progress(current, log)

    component_attempts = set()
    # Later components can create a new bridge to an earlier component.
    while True:
        components.sort(key=lambda m: component_rank(data, m), reverse=True)
        joined = False
        for i, j in combinations(range(len(components)), 2):
            if not can_join(components[i], components[j]):
                continue
            pair_key = (tuple(sorted(components[i].groups)), tuple(sorted(components[j].groups)))
            if pair_key in component_attempts:
                continue
            component_attempts.add(pair_key)
            started = time.perf_counter()
            trial = try_join(data, components[i], components[j], log, local=False)
            if trial is not None:
                trial.info["seconds"] = components[i].info["seconds"]+components[j].info["seconds"]+time.perf_counter()-started
                components[i] = trial
                components.pop(j)
                while True:
                    components[i], recovered = recover(data, components[i])
                    if not recovered:
                        break
                save_progress(components[i], log)
                joined = True
                break
        if not joined:
            break

    for i, current in enumerate(components):
        print(f"Final refinement: seed {current.seed}", flush=True)
        trial = deepcopy(current)
        ok, _ = refine(data, trial)
        if ok:
            ok, _ = protect_existing(data, current, trial)
        if ok and baseline_check(data, trial, [current]):
            trial.info["final_refinement"] = "final_ba_accepted"
            components[i] = trial
        save_progress(components[i], log)
        # Otherwise keep the previously accepted geometry in current.
    components.sort(key=lambda m: component_rank(data, m), reverse=True)
    return components, log


# ============================================================================
# 8. VANISHING-POINT DISPLAY ALIGNMENT AND PNG PLOTS.
# ============================================================================
def unit(v):
    return v/max(np.linalg.norm(v), 1e-12)


def estimate_vp(lines, K):
    """Intersect bearing planes robustly; the VP may lie at image infinity."""
    if len(lines) < 3:
        return None
    lines = np.asarray(lines)
    delta = lines[:, 1]-lines[:, 0]
    keep = np.linalg.norm(delta, axis=1) >= CFG["vp_min_edge_px"]
    lines, delta = lines[keep], delta[keep]
    if len(lines) < 3:
        return None
    direction_2d = delta/np.linalg.norm(delta, axis=1)[:, None]
    centres = lines.mean(axis=1)
    rays = (np.c_[lines.reshape(-1, 2), np.ones(2*len(lines))] @ np.linalg.inv(K).T).reshape(-1, 2, 3)
    normals = np.cross(rays[:, 0], rays[:, 1])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)

    def angular_error(d):
        v = K @ d
        projected = v[:2]-centres*v[2]
        projected /= np.maximum(np.linalg.norm(projected, axis=1)[:, None], 1e-12)
        return np.degrees(np.arccos(np.clip(np.abs(np.sum(projected*direction_2d, axis=1)), 0, 1)))

    rng, best, score = np.random.default_rng(CFG["random_seed"]), None, (-1, -np.inf)
    for _ in range(CFG["vp_trials"]):
        i, j = rng.choice(len(normals), 2, replace=False)
        d = np.cross(normals[i], normals[j])
        if np.linalg.norm(d) < 1e-10:
            continue
        d = unit(d)
        errors = angular_error(d)
        inliers = errors <= CFG["vp_inlier_angle_deg"]
        key = (int(inliers.sum()), -float(np.median(errors[inliers])) if inliers.any() else -np.inf)
        if key > score:
            best, score = d, key
    if best is None:
        return None
    for _ in range(3):
        inliers = angular_error(best) <= CFG["vp_inlier_angle_deg"]
        if inliers.sum() < 3:
            return None
        _, _, V = np.linalg.svd(normals[inliers])
        best = unit(V[-1])
    errors = angular_error(best)
    inliers = errors <= CFG["vp_inlier_angle_deg"]
    if inliers.sum() < 3 or inliers.mean() < CFG["vp_min_inlier_fraction"]:
        return None
    v = K @ best
    return dict(direction_camera=best, image_homogeneous=v,
                image_xy=v[:2]/v[2] if abs(best[2]) > 1e-9 else None,
                n_edges=len(lines), n_inliers=int(inliers.sum()), median_angular_error_deg=float(np.median(errors[inliers])))


def display_alignment(data, model):
    preferred = CFG["plot_frame"]
    frame = preferred if preferred in model.R else max(model.R, key=lambda f: sum(f in fs for fs in model.views.values()))
    h = unit(np.mean([X[1]-X[0]+X[2]-X[3] for X in model.points.values()], axis=0))
    v = unit(np.mean([X[0]-X[3]+X[1]-X[2] for X in model.points.values()], axis=0))
    lines = {"horizontal": [], "vertical": []}
    for o in model.points:
        if frame in model.views[o]:
            p = data["uv"][frame, o]
            lines["horizontal"].extend([p[[0, 1]], p[[3, 2]]])
            lines["vertical"].extend([p[[3, 0]], p[[2, 1]]])
    vps = {name: estimate_vp(segments, data["K"]) for name, segments in lines.items()}
    for name, vp in vps.items():
        if vp is not None:
            d = model.R[frame].T @ vp["direction_camera"]
            sign = 1 if d @ (h if name == "horizontal" else v) >= 0 else -1
            vp["direction_camera"] *= sign
            vp["image_homogeneous"] *= sign
            vp["direction_local"] = d*sign
            vp["direction_world"] = model.world_Q @ vp["direction_local"]
            vp["point_at_infinity_world"] = np.r_[vp["direction_world"], 0.]
    method, deviation = "reconstructed_edge_directions", None
    if all(vp is not None for vp in vps.values()):
        a, b = vps["horizontal"]["direction_local"], vps["vertical"]["direction_local"]
        deviation = abs(90-np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
        if deviation <= CFG["vp_max_orthogonality_error_deg"]:
            h, v, method = a, b, "measured_vp_directions_orthonormalized"
    up = unit(v-h*(h @ v))
    if np.linalg.norm(h) < 0.5 or np.linalg.norm(up) < 0.5:
        h, up, method = np.array([1., 0, 0]), np.array([0., -1, 0]), "local_coordinate_axes"
    depth = unit(np.cross(up, h))
    origin = np.concatenate(list(model.points.values())).mean(axis=0)
    return dict(frame_id=frame, method=method, origin_local_m=origin,
                basis_columns_local=np.column_stack([h, depth, up]), measured_vps=vps,
                measured_orthogonality_error_deg=deviation,
                derived_depth_direction_world=model.world_Q @ depth,
                note="Display rotation/translation only; depth direction derived, wall position not measured from VPs.")


def class_name(data, oid):
    name = data["objects"][oid].get("class_name", "other").strip().lower()
    return {"window": "windows", "door": "doors"}.get(name, name or "other")


def save_plots(data, model, cid, directory):
    alignment = display_alignment(data, model)
    B, origin = alignment["basis_columns_local"], alignment["origin_local_m"]
    points = {o: (X-origin) @ B for o, X in model.points.items()}
    all_points = np.concatenate(list(points.values()))
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    margin = np.maximum((high-low)*0.05, 0.15)
    low, high = low-margin, high+margin
    classes = sorted({class_name(data, o) for o in points})
    handles = [Line2D([0], [0], color=COLORS.get(c, COLORS["other"]), lw=2,
                      label=f"{c.title()} ({sum(class_name(data, o) == c for o in points)})") for c in classes]
    label = "VP-aligned" if alignment["method"].startswith("measured_vp") else "Geometry-aligned (VP fallback)"
    fig = plt.figure(figsize=(14, 7), dpi=CFG["plot_dpi"])
    ax = fig.add_subplot(111, projection="3d", proj_type="ortho")
    for oid, X in points.items():
        color = COLORS.get(class_name(data, oid), COLORS["other"])
        ax.add_collection3d(Poly3DCollection([X], facecolors=color, edgecolors=color, alpha=0.3, linewidths=1.2))
        if CFG["label_map_objects"]:
            ax.text(*X.mean(axis=0), oid.rsplit("_", 1)[-1], fontsize=6)
    if CFG["illustrative_reference_panel"]:
        d = float(np.median(all_points[:, 1]))
        panel = [[low[0], d, low[2]], [high[0], d, low[2]], [high[0], d, high[2]], [low[0], d, high[2]]]
        ax.add_collection3d(Poly3DCollection([panel], facecolors="#E8D3AF", edgecolors="#B28B49", alpha=0.08))
        handles.append(Line2D([0], [0], color="#B28B49", label="Illustrative reference panel"))
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]),
           xlabel="Along facade [m]", ylabel="Depth offset [m]", zlabel="Up facade [m]",
           title=f"{cid} · {len(points)} objects · {len(model.groups)} groups · {label}\nOrthographic 3D · measured depths retained")
    ax.set_box_aspect(high-low)
    ax.view_init(elev=CFG["plot_elevation_deg"], azim=CFG["plot_azimuth_deg"])
    ax.yaxis.set_major_locator(LinearLocator(3))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(axis="y", labelsize=8, pad=1)
    ax.yaxis.labelpad = 12
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False)
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.12, top=0.90)
    fig.savefig(directory / "map_3d.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=CFG["plot_dpi"])
    for oid, X in points.items():
        color = COLORS.get(class_name(data, oid), COLORS["other"])
        ax.add_patch(Polygon(X[:, [0, 2]], closed=True, facecolor=color, edgecolor=color, alpha=0.35))
        if CFG["label_map_objects"]:
            ax.text(*X[:, [0, 2]].mean(axis=0), oid.rsplit("_", 1)[-1], fontsize=6, ha="center")
    ax.set(xlim=(low[0], high[0]), ylim=(low[2], high[2]), xlabel="Along facade [m]", ylabel="Up facade [m]",
           title=f"{cid} · {label}\nFront view · depth omitted for display")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(directory / "map_front.png", bbox_inches="tight")
    plt.close(fig)
    return alignment


# ============================================================================
# 9. SIMPLE OUTPUT FOLDERS -- ONE MAP AND GEOMETRY FILE PER COMPONENT.
# ============================================================================
def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, set):
        return [json_ready(v) for v in sorted(value)]
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


def save_json(path, value):
    path.write_text(json.dumps(json_ready(value), indent=2, allow_nan=False), encoding="utf-8")


def save_progress(model, log):
    """One atomic accepted-geometry snapshot per seed; inspection, not resume state."""
    directory = OUTPUT_DIR / "progress"
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(
        status="intermediate_accepted_geometry", seed=model.seed,
        source_group_ids=model.groups, reference_frame=model.reference,
        note="Separate snapshots may overlap after component merges. Do not concatenate them. Not a resume checkpoint.",
        local_to_world=dict(rotation=model.world_Q, origin_world_m=model.world_origin),
        cameras=[dict(frame_id=f, R_local_to_camera=model.R[f], C_local_m=model.C[f])
                 for f in sorted(model.R)],
        objects=[dict(object_id=oid, corners_local_m=X,
                      corners_world_m=X @ model.world_Q.T + model.world_origin,
                      measurements=dimensions(X), used_frame_ids=model.views[oid])
                 for oid, X in sorted(model.points.items())])
    target = directory / f"{model.seed}.json"
    temporary = target.with_suffix(".tmp")
    save_json(temporary, payload)
    temporary.replace(target)
    temporary_log = directory / "merge_summary.tmp"
    pd.DataFrame(log).to_csv(temporary_log, index=False)
    temporary_log.replace(directory / "merge_summary.csv")


def prepare_output():
    # A fresh calculation replaces only this script's known result files.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for directory in OUTPUT_DIR.glob("C[0-9][0-9][0-9]*"):
        if directory.is_dir() and directory.name[1:].isdigit():
            for name in ("map.json", "geometry.json", "map_3d.png", "map_front.png"):
                (directory / name).unlink(missing_ok=True)
            if not any(directory.iterdir()):
                directory.rmdir()
    for name in ("summary.json", "component_summary.csv", "object_summary.csv", "merge_summary.csv"):
        (OUTPUT_DIR / name).unlink(missing_ok=True)


def save_component(data, model, cid, is_main):
    directory = OUTPUT_DIR / cid
    directory.mkdir(parents=True, exist_ok=True)
    alignment = save_plots(data, model, cid, directory)
    metrics = quality(data, model)
    calibration_reference = os.path.relpath(STAGE1_DIR / "experiment.json", directory)
    objects = []
    for oid, X in sorted(model.points.items()):
        world = X @ model.world_Q.T+model.world_origin
        observations = [dict(frame_id=f, local_object_id=data["local_ids"][f, oid],
                             raw_corners_px=data["raw"][f, oid], visibility=data["visibility"][f, oid],
                             used_in_geometry=f in model.views[oid]) for f in sorted(model.R) if (f, oid) in data["raw"]]
        objects.append(dict(object_id=oid, **data["objects"][oid], corners_local_m=X, corners_world_m=world,
                            centre_local_m=X.mean(axis=0), centre_world_m=world.mean(axis=0),
                            measurements=dimensions(X), observations=observations,
                            source_group_ids=model.sources[oid], recovered_in_stage2=oid in model.recovered))
    save_json(directory / "map.json", dict(component_id=cid, is_main=is_main, reference_frame=model.reference,
                                           source_group_ids=model.groups, coordinate_reference="geometry.json",
                                           calibration_reference=calibration_reference, corner_order=CORNERS, objects=objects))
    cameras = [dict(frame_id=f, R_local_to_camera=model.R[f], C_local_m=model.C[f],
                    R_world_to_camera=model.R[f] @ model.world_Q.T,
                    C_world_m=model.world_Q @ model.C[f]+model.world_origin,
                    sensor_C_world_m=data["sensor_C"][f]) for f in sorted(model.R)]
    save_json(directory / "geometry.json", dict(
        component_id=cid, is_main=is_main, reference_frame=model.reference, source_group_ids=model.groups,
        cameras=cameras, local_to_world=dict(rotation=model.world_Q, origin_world_m=model.world_origin,
                                             status="provisional_reference_sensor_pose"),
        metrics=metrics, last_refinement=model.info.get("last_refinement"),
        last_optimizer=model.info.get("last_optimizer"), final_refinement_status=model.info["final_refinement"],
        display_alignment=alignment,
        conventions=dict(projection="X_camera = R_local_to_camera @ (X_local - C_local)",
                         world="X_world = rotation @ X_local + origin_world_m", units="metres",
                         rmse="sqrt(mean(du^2 + dv^2)) over used corners",
                         interpretation="Internal consistency; absolute accuracy is not independently validated.")))
    row = {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}
    row.update(component_id=cid, is_main=is_main, group_ids="|".join(sorted(model.groups)),
               reference_frame=model.reference, construction_seconds=round(model.info["seconds"], 3),
               final_refinement_status=model.info["final_refinement"])
    print(f"{cid}: {len(model.points)} objects, {len(model.R)} frames, RMSE={metrics['rmse_px']:.3f} px" + (" [MAIN MAP]" if is_main else ""))
    return row, metrics


def save_results(data, components, merge_log, n_input_groups, started):
    component_rows, object_rows, quality_by_component = [], [], {}
    labelled = {f"C{i+1:03d}": model for i, model in enumerate(components)}
    main_id = next(iter(labelled))
    for cid, model in labelled.items():
        row, metrics = save_component(data, model, cid, cid == main_id)
        component_rows.append(row)
        quality_by_component[cid] = metrics
    for oid, meta in sorted(data["objects"].items()):
        memberships = [cid for cid, model in labelled.items() if oid in model.points]
        chosen = main_id if main_id in memberships else memberships[0] if memberships else None
        status = ("multiple_estimates" if len(memberships) > 1 else "main_map" if main_id in memberships
                  else "separate_component" if memberships else "not_reconstructed")
        row = dict(object_id=oid, class_name=meta.get("class_name", ""), status=status,
                   component_ids="|".join(memberships), selected_component=chosen,
                   in_main_map=main_id in memberships)
        if chosen is not None:
            model = labelled[chosen]
            row.update(dimensions(model.points[oid]))
            row.update(quality_by_component[chosen]["objects"][oid])
            row["source_group_ids"] = "|".join(sorted(model.sources[oid]))
            row["recovered_in_stage2"] = oid in model.recovered
        object_rows.append(row)
    group_component = {g: cid for cid, model in labelled.items() for g in model.groups}
    for row in merge_log:
        row["target_final_component"] = group_component[row["target_seed"]]
    pd.DataFrame(component_rows).to_csv(OUTPUT_DIR / "component_summary.csv", index=False)
    pd.DataFrame(object_rows).to_csv(OUTPUT_DIR / "object_summary.csv", index=False)
    columns = list(dict.fromkeys(["target_seed", "incoming_group_ids", "status"]+[k for row in merge_log for k in row]))
    pd.DataFrame(merge_log, columns=columns).to_csv(OUTPUT_DIR / "merge_summary.csv", index=False)
    represented = set().union(*(set(m.points) for m in components))
    save_json(OUTPUT_DIR / "summary.json", dict(
        main_component_id=main_id, main_map=f"{main_id}/map.json",
        n_input_accepted_groups=n_input_groups, n_components=len(components),
        n_original_objects=len(data["objects"]), n_unique_reconstructed_objects=len(represented),
        n_main_map_objects=len(labelled[main_id].points), n_not_reconstructed=len(set(data["objects"])-represented),
        selected_group_ids=GROUP_IDS, settings=CFG,
        experiment_reference=os.path.relpath(STAGE1_DIR / "experiment.json", OUTPUT_DIR),
        correspondence_csv=str(CORRESPONDENCE_CSV.resolve()), total_seconds=round(time.perf_counter()-started, 3),
        main_selection="Most unique accepted objects; coverage then reprojection quality break ties.",
        baseline_constraints="Union of unique camera pairs from contributing Stage 1 groups.",
        calibration="Fixed; stored once in the Stage 1 experiment metadata.",
        localization="Use one component per pose estimate; separate component coordinates are not interchangeable."))


# ============================================================================
# 10. MAIN -- THE COMPLETE STAGE 2 FLOW.
# ============================================================================
def main():
    started = time.perf_counter()

    # A. Read only the selected, accepted Stage 1 group reconstructions.
    data, groups = load_inputs()
    if not groups:
        print("No accepted Stage 1 groups selected. Nothing to reconstruct.")
        return

    # B. Align overlapping groups, refine trial merges and preserve failed joins.
    # C. Retry deferred groups and recover objects from their original annotations.
    # D. Keep unresolved groups as separate components and run final refinement.
    components, merge_log = build_components(data, groups)

    # E. Sort by accepted unique objects, then coverage and reprojection quality.
    #    The strongest component is C001 and is the default localization map.
    # F. Recalculate dimensions, place each component in world coordinates, export.
    prepare_output()
    save_results(data, components, merge_log, len(groups), started)
    print(f"\nFinished. Main localization map: {OUTPUT_DIR / 'C001' / 'map.json'}")


if __name__ == "__main__":
    main()
