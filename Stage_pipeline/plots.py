"""PNG-only plots. Vanishing points define display axes, never reconstruction depth.

The front view omits depth. The 3D view retains depth at the same metric scale as
the other axes. Any optional background panel is illustrative, not a fitted wall.
"""

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from geometry import dimensions, world_transform

COLORS = {"frame": "#E69F00", "glass": "#56B4E9", "darkening": "#9146B6",
          "windows": "#0072B2", "doors": "#009E73", "other": "#666666"}


def object_class(data, oid):
    value = data["objects"][oid].get("class_name", "other").strip().lower()
    return {"window": "windows", "door": "doors"}.get(value, value or "other")


def color(data, oid):
    return COLORS.get(object_class(data, oid), COLORS["other"])


def unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def fit_vanishing_direction(segments, K, cfg):
    """RANSAC on image line directions, followed by a bearing-plane SVD fit."""
    if len(segments) < 3:
        return None
    segments = np.asarray(segments)
    delta = segments[:, 1] - segments[:, 0]
    lengths = np.linalg.norm(delta, axis=1)
    segments, delta = segments[lengths >= cfg["vp_min_edge_px"]], delta[lengths >= cfg["vp_min_edge_px"]]
    if len(segments) < 3:
        return None
    directions = delta / np.linalg.norm(delta, axis=1)[:, None]
    centres = segments.mean(axis=1)
    rays = np.c_[segments.reshape(-1, 2), np.ones(2 * len(segments))] @ np.linalg.inv(K).T
    rays = rays.reshape(-1, 2, 3)
    normals = np.cross(rays[:, 0], rays[:, 1])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)

    def errors(d):
        v = K @ d
        lines = v[:2] - centres * v[2]
        lines /= np.maximum(np.linalg.norm(lines, axis=1)[:, None], 1e-12)
        return np.degrees(np.arccos(np.clip(np.abs(np.sum(lines * directions, axis=1)), 0, 1)))

    rng = np.random.default_rng(cfg["random_seed"])
    best, score = None, (-1, -np.inf)
    for _ in range(cfg["vp_trials"]):
        i, j = rng.choice(len(normals), 2, replace=False)
        d = np.cross(normals[i], normals[j])
        if np.linalg.norm(d) < 1e-10:
            continue
        d = unit(d)
        e = errors(d)
        keep = e <= cfg["vp_inlier_angle_deg"]
        key = (int(keep.sum()), -float(np.median(e[keep])) if keep.any() else -np.inf)
        if key > score:
            best, score = d, key
    if best is None:
        return None
    for _ in range(3):
        keep = errors(best) <= cfg["vp_inlier_angle_deg"]
        if keep.sum() < 3:
            return None
        _, _, V = np.linalg.svd(normals[keep])
        best = unit(V[-1])
    e = errors(best)
    keep = e <= cfg["vp_inlier_angle_deg"]
    if keep.sum() < 3 or keep.mean() < cfg["vp_min_inlier_fraction"]:
        return None
    v = K @ best
    return dict(direction_camera=best, image_homogeneous=v,
                image_xy=(v[:2] / v[2]) if abs(best[2]) > 1e-9 else None,
                n_edges=len(segments), n_inliers=int(keep.sum()),
                median_angular_error_deg=float(np.median(e[keep])))


def display_alignment(data, model, frame, cfg):
    points = np.concatenate(list(model.points.values()))
    # Annotation order provides the sign: TL->TR is right, BL->TL is up.
    horizontal = unit(np.mean([X[1] - X[0] + X[2] - X[3] for X in model.points.values()], axis=0))
    vertical = unit(np.mean([X[0] - X[3] + X[1] - X[2] for X in model.points.values()], axis=0))
    segments = {"horizontal": [], "vertical": []}
    for o in model.points:
        if frame not in model.views[o]:
            continue
        p = data["uv"][frame, o]
        segments["horizontal"].extend([p[[0, 1]], p[[3, 2]]])
        segments["vertical"].extend([p[[3, 0]], p[[2, 1]]])
    measured = {name: fit_vanishing_direction(lines, data["K"], cfg) for name, lines in segments.items()}
    Q, origin_world = world_transform(data, model)
    for name, result in measured.items():
        if result is None:
            continue
        d = model.R[frame].T @ result["direction_camera"]
        sign = 1 if d @ (horizontal if name == "horizontal" else vertical) >= 0 else -1
        for key in ("direction_camera", "image_homogeneous"):
            result[key] *= sign
        result["direction_local"] = d * sign
        result["direction_world"] = Q @ result["direction_local"]
        result["point_at_infinity_world"] = np.r_[result["direction_world"], 0.0]
    method = "reconstructed_edge_directions"
    orthogonality_error = None
    if all(v is not None for v in measured.values()):
        h, v = measured["horizontal"]["direction_local"], measured["vertical"]["direction_local"]
        orthogonality_error = abs(90 - np.degrees(np.arccos(np.clip(h @ v, -1, 1))))
        if orthogonality_error <= cfg["vp_max_orthogonality_error_deg"]:
            horizontal, vertical, method = h, v, "measured_vp_directions_orthonormalized"
    up = unit(vertical - horizontal * (horizontal @ vertical))
    if np.linalg.norm(up) < 0.5:
        # A pathological set of edges has no usable facade axes.
        horizontal, up, method = np.array([1., 0, 0]), np.array([0., -1, 0]), "reference_camera_axes"
    depth = unit(np.cross(up, horizontal))
    basis = np.column_stack([horizontal, depth, up])
    centre = points.mean(axis=0)
    return dict(frame_id=frame, method=method, origin_local_m=centre, basis_columns_local=basis,
                origin_world_m=Q @ centre + origin_world, measured_vps=measured,
                measured_orthogonality_error_deg=orthogonality_error,
                depth_direction_world=Q @ depth,
                depth_direction_note="derived cross product, not an independently measured VP")


def image_overlay(data, model, frame, path, cfg):
    source = data["image_paths"].get(frame)
    image = cv2.imread(str(source)) if source is not None else None
    if image is None:
        return "source_image_unavailable"
    if (image.shape[1], image.shape[0]) != tuple(cfg["image_size"]):
        return "source_image_size_differs_from_calibration_pixels"
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(15, 15 * height / width), dpi=cfg["plot_dpi"])
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set(xlim=(0, width), ylim=(height, 0), title=f"{frame} | solid: fitted projection; dashed: annotation | dimensions in m")
    ax.axis("off")
    fig.tight_layout(pad=0.7)
    labels = []
    for o, X in model.points.items():
        if (frame, o) not in data["raw"]:
            continue
        R, C = model.R[frame], model.C[frame]
        if not np.all(((X - C) @ R.T)[:, 2] > 0):
            continue
        pixels = cv2.projectPoints(X, cv2.Rodrigues(R)[0], -R @ C, data["K"], data["dist"])[0].reshape(4, 2)
        if not np.all((pixels >= 0) & (pixels < [width, height])):
            continue
        c = color(data, o)
        ax.add_patch(Polygon(pixels, closed=True, fill=False, edgecolor=c, linewidth=1.2))
        raw = data["raw"][frame, o]
        if np.isfinite(raw).all():
            ax.add_patch(Polygon(raw, closed=True, fill=False, edgecolor="yellow", linewidth=0.6, linestyle="--", alpha=0.7))
        size = dimensions(X)
        suffix = "" if frame in model.views[o] else " [unused view]"
        label = f"{o.rsplit('_', 1)[-1]} {size['width_m']:.2f} × {size['height_m']:.2f}{suffix}"
        labels.append((pixels.mean(axis=0), label, c))
    # Greedy label placement in screen coordinates; small leaders identify offsets.
    fig.canvas.draw()
    renderer, occupied = fig.canvas.get_renderer(), []
    bounds = ax.get_window_extent(renderer)
    for centre, text, c in sorted(labels, key=lambda item: item[0][1]):
        annotation = ax.annotate(text, centre, xytext=(0, 0), textcoords="offset points",
                                 ha="center", va="center", fontsize=6.5, color="white",
                                 bbox=dict(facecolor="black", alpha=0.72, edgecolor=c, pad=1.8))
        best = None
        for dy in (0, -12, 12, -24, 24, -36, 36, -48, 48):
            for dx in (0, -45, 45, -90, 90):
                annotation.set_position((dx, dy))
                box = annotation.get_window_extent(renderer).expanded(1.04, 1.15)
                overlaps = sum(max(0, min(box.x1, b.x1)-max(box.x0, b.x0)) *
                               max(0, min(box.y1, b.y1)-max(box.y0, b.y0)) for b in occupied)
                outside = max(0, bounds.x0-box.x0) + max(0, box.x1-bounds.x1) + max(0, bounds.y0-box.y0) + max(0, box.y1-bounds.y1)
                score = overlaps + 1000 * outside + 0.01 * (dx*dx + dy*dy)
                if best is None or score < best[0]:
                    best = score, (dx, dy), box.frozen()
        annotation.set_position(best[1])
        occupied.append(best[2])
        if best[1] != (0, 0):
            end = ax.transData.inverted().transform(ax.transData.transform(centre) + np.asarray(best[1]) * fig.dpi / 72)
            ax.plot([centre[0], end[0]], [centre[1], end[1]], color=c, linewidth=0.5)
    fig.savefig(path, dpi=cfg["plot_dpi"], bbox_inches="tight")
    plt.close(fig)
    return "saved"


def save_plots(data, model, group_id, output_dir, cfg):
    output_dir = Path(output_dir)
    preferred = cfg["plot_frame"]
    frame = preferred if preferred in model.R else max(model.R, key=lambda f: sum(f in fs for fs in model.views.values()))
    alignment = display_alignment(data, model, frame, cfg)
    basis, origin = alignment["basis_columns_local"], alignment["origin_local_m"]
    points = {o: (X-origin) @ basis for o, X in model.points.items()}
    all_points = np.concatenate(list(points.values()))
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    margin = np.maximum((high-low) * 0.05, 0.15)
    low, high = low-margin, high+margin
    classes = sorted({object_class(data, o) for o in points})
    handles = [Line2D([0], [0], color=COLORS.get(c, COLORS["other"]), lw=2,
                      label=f"{c.title()} ({sum(object_class(data, o) == c for o in points)})") for c in classes]
    label = "VP-aligned" if alignment["method"].startswith("measured_vp") else "Geometry-aligned (VP fallback)"

    fig = plt.figure(figsize=(14, 7), dpi=cfg["plot_dpi"])
    ax = fig.add_subplot(111, projection="3d", proj_type="ortho")
    for o, X in points.items():
        ax.add_collection3d(Poly3DCollection([X], facecolors=color(data, o), edgecolors=color(data, o), alpha=0.3, linewidths=1.2))
        if cfg["label_map_objects"]:
            ax.text(*X.mean(axis=0), o.rsplit("_", 1)[-1], fontsize=6)
    if cfg["illustrative_reference_panel"]:
        d = float(np.median(all_points[:, 1]))
        panel = [[low[0], d, low[2]], [high[0], d, low[2]], [high[0], d, high[2]], [low[0], d, high[2]]]
        ax.add_collection3d(Poly3DCollection([panel], facecolors="#E8D3AF", edgecolors="#B28B49", alpha=0.08))
        handles.append(Line2D([0], [0], color="#B28B49", label="Illustrative reference panel"))
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]),
           xlabel="Along facade [m]", ylabel="Depth offset [m]", zlabel="Up facade [m]")
    ax.set_box_aspect(high-low)
    ax.view_init(elev=cfg["plot_elevation_deg"], azim=cfg["plot_azimuth_deg"])
    ax.yaxis.set_major_locator(LinearLocator(3))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(axis="y", labelsize=8, pad=1)
    ax.yaxis.labelpad = 12
    ax.set_title(f"{group_id} · {len(points)} accepted objects · {label}\nOrthographic 3D · measured depths retained", pad=14)
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False)
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.12, top=0.90)
    fig.savefig(output_dir / "map_3d.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=cfg["plot_dpi"])
    for o, X in points.items():
        ax.add_patch(Polygon(X[:, [0, 2]], closed=True, facecolor=color(data, o), edgecolor=color(data, o), alpha=0.35))
        if cfg["label_map_objects"]:
            x, z = X[:, [0, 2]].mean(axis=0)
            ax.text(x, z, o.rsplit("_", 1)[-1], fontsize=6, ha="center")
    ax.set(xlim=(low[0], high[0]), ylim=(low[2], high[2]), xlabel="Along facade [m]", ylabel="Up facade [m]",
           title=f"{group_id} · {label}\nFront view · depth omitted for display")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(output_dir / "map_front.png", bbox_inches="tight")
    plt.close(fig)
    alignment["image_overlay"] = image_overlay(data, model, frame, output_dir / "image_overlay.png", cfg)
    print(f"  Plot frame: {frame}; axes: {alignment['method']}; image overlay: {alignment['image_overlay']}")
    if cfg["print_vps"]:
        print("  VPs have no finite 3D position. Reporting unit world directions [dx, dy, dz].")
        for name, vp in alignment["measured_vps"].items():
            if vp is not None:
                print(f"  {name}: image VP={vp['image_xy']}; world direction={np.round(vp['direction_world'], 6)}; "
                      f"inliers={vp['n_inliers']}/{vp['n_edges']}; median angular error={vp['median_angular_error_deg']:.3f} deg")
    return alignment
