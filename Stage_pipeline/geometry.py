"""Stage 1 geometry. Pixels are undistorted; X_camera = R @ (X_local - C).

E gives a rotation and a translation DIRECTION. Sensor-centre distances supply
metric scale. The reference camera fixes the remaining coordinate gauge.
No wall, rectangle, or vanishing-point constraint changes the measured points.
"""

from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

CORNERS = ("TL", "TR", "BR", "BL")


@dataclass
class Model:
    reference: str
    R: dict
    C: dict
    points: dict = field(default_factory=dict)
    views: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


def rotation_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def project(X, R, C, K):
    camera = (X - C) @ R.T
    pixels = camera @ K.T
    z = pixels[:, 2:3]
    safe_z = np.where(np.abs(z) < 1e-9, np.where(z < 0, -1e-9, 1e-9), z)
    return pixels[:, :2] / safe_z, camera[:, 2]


def dimensions(X):
    top, bottom = np.linalg.norm(X[1] - X[0]), np.linalg.norm(X[2] - X[3])
    left, right = np.linalg.norm(X[3] - X[0]), np.linalg.norm(X[2] - X[1])
    return dict(width_m=float((top + bottom) / 2), height_m=float((left + right) / 2),
                top_m=float(top), bottom_m=float(bottom), left_m=float(left), right_m=float(right))


def estimate_pairs(data, frames, anchors, cfg):
    """Estimate independent E/R/t for every pair; masks are initialization evidence."""
    pairs, reports = {}, []
    for a, b in combinations(frames, 2):
        pa = np.concatenate([data["uv"][a, o] for o in anchors])
        pb = np.concatenate([data["uv"][b, o] for o in anchors])
        report = dict(frame_a=a, frame_b=b, input_corners=len(pa), status="no_essential_matrix")
        reports.append(report)
        cv2.setRNGSeed(cfg["random_seed"])
        try:
            E, mask = cv2.findEssentialMat(pa, pb, data["K"], method=cv2.RANSAC,
                                          prob=0.999, threshold=cfg["essential_threshold_px"],
                                          maxIters=5000)
            if E is None or mask is None:
                continue
            report["essential_inliers"] = int(mask.sum())
            best = None
            for block in np.asarray(E).reshape(-1, 3, 3):
                count, R, t, good, _ = cv2.recoverPose(
                    block, pa, pb, data["K"], distanceThresh=1e9, mask=mask.copy())
                if best is None or count > best[0]:
                    best = count, block, R, t.ravel(), good.ravel().astype(bool)
        except cv2.error:
            report["status"] = "pose_estimation_failed"
            continue
        count, E, R, t, good = best
        complete = [o for o, keep in zip(anchors, good.reshape(-1, 4).all(axis=1)) if keep]
        report.update(pose_inliers=int(count), complete_objects=complete)
        if count < cfg["min_pair_corners"] or len(complete) < cfg["min_pair_objects"]:
            report["status"] = "insufficient_pair_support"
            continue
        invK = np.linalg.inv(data["K"])
        ra = np.c_[pa, np.ones(len(pa))] @ invK.T
        rb = (np.c_[pb, np.ones(len(pb))] @ invK.T) @ R
        cosine = np.sum(ra * rb, axis=1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1))
        angles = np.degrees(np.arccos(np.clip(np.abs(cosine), 0, 1)))
        report.update(status="estimated", median_ray_angle_deg=float(np.median(angles[good])))
        pairs[a, b] = dict(E=E, R=R, t=t, report=report)
    return pairs, reports


def relative_pose(pairs, a, b):
    if (a, b) in pairs:
        return pairs[a, b]["R"], pairs[a, b]["t"]
    if (b, a) in pairs:
        R, t = pairs[b, a]["R"], pairs[b, a]["t"]
        return R.T, -R.T @ t
    return None


def rotation_cycles(pairs, frames):
    cycles = []
    for a, b, c in combinations(frames, 3):
        ab, bc, ac = relative_pose(pairs, a, b), relative_pose(pairs, b, c), relative_pose(pairs, a, c)
        if ab is not None and bc is not None and ac is not None:
            error = rotation_angle(ac[0] @ (bc[0] @ ab[0]).T)
            cycles.append(dict(frames=[a, b, c], error_deg=error))
    return cycles


def triangulate(data, model, oid, views):
    """Multi-view DLT initializes the four independently reconstructed corners."""
    projections = {f: data["K"] @ np.c_[model.R[f], -model.R[f] @ model.C[f]] for f in views}
    points = []
    for corner in range(4):
        rows = []
        for f in views:
            u, v = data["uv"][f, oid][corner]
            P = projections[f]
            rows.extend([u * P[2] - P[0], v * P[2] - P[1]])
        _, _, V = np.linalg.svd(rows)
        h = V[-1]
        if abs(h[3]) < 1e-12:
            return None
        points.append(h[:3] / h[3])
    return np.asarray(points)


def ray_angle(X, centres):
    """For each corner take its best baseline; report the weakest of four corners."""
    best = np.zeros(4)
    for a, b in combinations(centres, 2):
        ra, rb = X - a, X - b
        cosine = np.sum(ra * rb, axis=1) / np.maximum(np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1), 1e-12)
        best = np.maximum(best, np.degrees(np.arccos(np.clip(np.abs(cosine), 0, 1))))
    return float(best.min())


def object_quality(data, model, oid, X=None, views=None):
    X = model.points[oid] if X is None else X
    views = model.views[oid] if views is None else views
    errors, depths = [], []
    for f in views:
        pixels, z = project(X, model.R[f], model.C[f], data["K"])
        errors.extend(np.linalg.norm(pixels - data["uv"][f, oid], axis=1))
        depths.extend(z)
    e = np.asarray(errors)
    return dict(rmse_px=float(np.sqrt(np.mean(e**2))), max_error_px=float(e.max()),
                min_ray_angle_deg=ray_angle(X, [model.C[f] for f in views]),
                positive_depth=bool(np.all(np.asarray(depths) > 1e-6)), n_views=len(views))


def reconstruct_objects(data, model, object_ids, cfg, initial=False):
    """Try largest view subsets first. A bad observation need not remove its object."""
    points, used_views = {}, {}
    for oid in object_ids:
        available = [f for f in model.R if (f, oid) in data["uv"]]
        for n in range(len(available), 1, -1):
            valid = []
            for subset in combinations(available, n):
                X = triangulate(data, model, oid, subset)
                if X is None or not np.isfinite(X).all():
                    continue
                def residual(x):
                    return np.concatenate([(project(x.reshape(4, 3), model.R[f], model.C[f], data["K"])[0]
                                            - data["uv"][f, oid]).ravel() for f in subset])
                fit = least_squares(residual, X.ravel(), loss="soft_l1", f_scale=cfg["robust_scale_px"],
                                    max_nfev=cfg["max_point_nfev"])
                X = fit.x.reshape(4, 3)
                q = object_quality(data, model, oid, X, subset)
                limit = cfg["initial_error_px"] if initial else cfg["max_corner_error_px"]
                rmse_limit = limit if initial else cfg["max_object_rmse_px"]
                if (q["positive_depth"] and q["max_error_px"] <= limit and q["rmse_px"] <= rmse_limit
                        and q["min_ray_angle_deg"] >= cfg["min_ray_angle_deg"]):
                    valid.append((q["rmse_px"], X, subset))
            if valid:
                _, points[oid], used_views[oid] = min(valid, key=lambda item: item[0])
                break
    model.points, model.views = points, used_views
    return model


def initial_models(data, frames, anchors, pairs, cfg):
    """First a reference-camera star; on failure the caller requests seed/PnP starts."""
    ref = frames[0]
    if all(relative_pose(pairs, ref, f) is not None for f in frames[1:]):
        model = Model(ref, {ref: np.eye(3)}, {ref: np.zeros(3)}, info={"initialization": "reference_star"})
        for f in frames[1:]:
            R, t = relative_pose(pairs, ref, f)
            baseline = np.linalg.norm(data["sensor_C"][f] - data["sensor_C"][ref])
            model.R[f], model.C[f] = R, -R.T @ (t * baseline)
        yield reconstruct_objects(data, model, anchors, cfg, initial=True)

    seeds = sorted(pairs, key=lambda p: (pairs[p]["report"]["median_ray_angle_deg"],
                                         pairs[p]["report"]["pose_inliers"]), reverse=True)
    for a, b in seeds[:cfg["max_seed_pairs"]]:
        R, t = relative_pose(pairs, a, b)
        baseline = np.linalg.norm(data["sensor_C"][b] - data["sensor_C"][a])
        model = Model(a, {a: np.eye(3), b: R}, {a: np.zeros(3), b: -R.T @ (t * baseline)},
                      info={"initialization": "seed_pair_pnp", "seed_pair": [a, b]})
        reconstruct_objects(data, model, anchors, cfg, initial=True)
        pending = [f for f in frames if f not in model.R]
        while pending and model.points:
            added = False
            for f in list(pending):
                ids = [o for o in model.points if (f, o) in data["uv"]]
                if len(ids) < cfg["min_pair_objects"]:
                    continue
                X = np.concatenate([model.points[o] for o in ids])
                uv = np.concatenate([data["uv"][f, o] for o in ids])
                cv2.setRNGSeed(cfg["random_seed"])
                try:
                    ok, r, t, inliers = cv2.solvePnPRansac(
                        X, uv, data["K"], None, iterationsCount=1000,
                        reprojectionError=cfg["pnp_threshold_px"], confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
                    if not ok or inliers is None or len(inliers) < cfg["min_pair_corners"]:
                        continue
                    keep = inliers.ravel()
                    r, t = cv2.solvePnPRefineLM(X[keep], uv[keep], data["K"], None, r, t)
                except cv2.error:
                    continue
                R = cv2.Rodrigues(r)[0]
                model.R[f], model.C[f] = R, -R.T @ t.ravel()
                pending.remove(f)
                added = True
                reconstruct_objects(data, model, anchors, cfg, initial=True)
            if not added:
                break
        if not pending:
            yield model


def reprojection_records(data, model):
    records = []
    for o in model.points:
        for f in model.views[o]:
            uv, depth = project(model.points[o], model.R[f], model.C[f], data["K"])
            for k, error in enumerate(np.linalg.norm(uv - data["uv"][f, o], axis=1)):
                records.append((o, f, k, float(error), float(depth[k])))
    return records


def bundle_adjust(data, model, cfg):
    """Joint camera/point BA, robust pixels and quadratic metric-baseline priors."""
    movable = [f for f in model.R if f != model.reference]
    ids = list(model.points)
    observations = [(o, f) for o in ids for f in model.views[o]]
    edges = list(combinations(model.R, 2))
    targets = np.array([np.linalg.norm(data["sensor_C"][a] - data["sensor_C"][b]) for a, b in edges])
    camera_slice = {f: slice(6*i, 6*i+6) for i, f in enumerate(movable)}
    start = 6 * len(movable)
    point_slice = {o: slice(start+12*i, start+12*i+12) for i, o in enumerate(ids)}
    x0 = np.r_[np.concatenate([np.r_[Rotation.from_matrix(model.R[f]).as_rotvec(), model.C[f]] for f in movable]),
               np.concatenate([model.points[o].ravel() for o in ids])]
    pixel_count = 8 * len(observations)

    def unpack(x):
        Rs, Cs = {model.reference: model.R[model.reference]}, {model.reference: model.C[model.reference]}
        for f, s in camera_slice.items():
            v = x[s]
            Rs[f], Cs[f] = Rotation.from_rotvec(v[:3]).as_matrix(), v[3:]
        return Rs, Cs, {o: x[s].reshape(4, 3) for o, s in point_slice.items()}

    def residual(x):
        Rs, Cs, Xs = unpack(x)
        pixels = np.concatenate([(project(Xs[o], Rs[f], Cs[f], data["K"])[0] - data["uv"][f, o]).ravel()
                                 for o, f in observations]) / cfg["pixel_sigma_px"]
        baselines = (np.array([np.linalg.norm(Cs[a] - Cs[b]) for a, b in edges]) - targets) / cfg["baseline_sigma_m"]
        return np.r_[pixels, baselines]

    def mixed_loss(z):
        # scipy passes squared, scaled residuals; leave baseline priors quadratic.
        rho = np.vstack([z.copy(), np.ones_like(z), np.zeros_like(z)])
        q = 1 + z[:pixel_count]
        rho[:, :pixel_count] = [2 * (np.sqrt(q) - 1), q**-0.5, -0.5 * q**-1.5]
        return rho

    sparsity = lil_matrix((pixel_count + len(edges), len(x0)), dtype=int)
    for i, (o, f) in enumerate(observations):
        sparsity[8*i:8*i+8, point_slice[o]] = 1
        if f in camera_slice:
            sparsity[8*i:8*i+8, camera_slice[f]] = 1
    for i, (a, b) in enumerate(edges):
        for f in (a, b):
            if f in camera_slice:
                sparsity[pixel_count+i, camera_slice[f]] = 1
    before = reprojection_records(data, model)
    result = least_squares(residual, x0, loss=mixed_loss,
                           f_scale=cfg["robust_scale_px"] / cfg["pixel_sigma_px"], x_scale="jac",
                           jac_sparsity=sparsity.tocsr() if len(x0) > cfg["sparse_ba_above_variables"] else None,
                           max_nfev=cfg["max_ba_nfev"], ftol=cfg["ba_tolerance"],
                           xtol=cfg["ba_tolerance"], gtol=cfg["ba_tolerance"])
    model.R, model.C, model.points = unpack(result.x)
    model.info["last_ba"] = dict(converged=bool(result.success), nfev=int(result.nfev),
                                  optimality=float(result.optimality), before_records=before)
    return bool(np.isfinite(result.x).all() and (result.success or not cfg["require_ba_convergence"]))


def enough_support(model, cfg):
    if len(model.points) < cfg["min_group_objects"]:
        return False
    if any(sum(f in fs for fs in model.views.values()) < cfg["min_camera_objects"] for f in model.R):
        return False
    reached = {model.reference}
    for _ in model.R:
        for views in model.views.values():
            if reached.intersection(views):
                reached.update(views)
    return reached == set(model.R)


def refine_and_screen(data, model, cfg):
    """BA -> remove unreliable observations -> BA again on exactly retained data."""
    for _ in range(cfg["screen_rounds"]):
        if not enough_support(model, cfg):
            return False, "insufficient_connected_support"
        if not bundle_adjust(data, model, cfg):
            return False, "ba_not_converged"
        changed = False
        for o in list(model.points):
            views = []
            for f in model.views[o]:
                uv, z = project(model.points[o], model.R[f], model.C[f], data["K"])
                if np.all(z > 1e-6) and np.max(np.linalg.norm(uv - data["uv"][f, o], axis=1)) <= cfg["max_corner_error_px"]:
                    views.append(f)
            changed |= set(views) != set(model.views[o])
            model.views[o] = views
            q = object_quality(data, model, o) if len(views) >= 2 else None
            if q is None or q["rmse_px"] > cfg["max_object_rmse_px"] or q["min_ray_angle_deg"] < cfg["min_ray_angle_deg"]:
                del model.points[o], model.views[o]
                changed = True
        if not changed:
            return enough_support(model, cfg), "accepted" if enough_support(model, cfg) else "insufficient_connected_support"
    return False, "screening_not_stable"


def evaluation(data, model):
    records = reprojection_records(data, model)
    errors = np.array([r[3] for r in records])
    keys = {r[:3] for r in records}
    before = [r[3] for r in model.info["last_ba"]["before_records"] if r[:3] in keys]
    baselines = [dict(frame_a=a, frame_b=b,
                      sensor_m=float(np.linalg.norm(data["sensor_C"][a] - data["sensor_C"][b])),
                      fitted_m=float(np.linalg.norm(model.C[a] - model.C[b]))) for a, b in combinations(model.R, 2)]
    for edge in baselines:
        edge["residual_m"] = edge["fitted_m"] - edge["sensor_m"]
    return dict(n_objects=len(model.points), n_cameras=len(model.R), n_corner_observations=len(records),
                rmse_px=float(np.sqrt(np.mean(errors**2))), median_error_px=float(np.median(errors)),
                p95_error_px=float(np.percentile(errors, 95)), max_error_px=float(errors.max()),
                positive_depth_fraction=float(np.mean([r[4] > 0 for r in records])),
                before_last_ba_rmse_px=float(np.sqrt(np.mean(np.square(before)))),
                baseline_rmse_m=float(np.sqrt(np.mean([b["residual_m"]**2 for b in baselines]))),
                ba_converged=model.info["last_ba"]["converged"], baselines=baselines,
                objects={o: object_quality(data, model, o) for o in model.points})


def world_transform(data, model):
    """Provisional georeference; sensor rotations do not enter the image-only R fit."""
    Q = data["sensor_Q"][model.reference] @ model.R[model.reference]
    origin = data["sensor_C"][model.reference] - Q @ model.C[model.reference]
    return Q, origin
