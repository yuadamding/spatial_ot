from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .config import ExperimentConfig
from .preprocessing import PreparedSpatialOTData, prepare_data


def _format_spatial_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_aspect("equal")
    ax.invert_yaxis()


def _marker_size(n_points: int, *, low: float, high: float) -> float:
    if n_points <= 1000:
        return high
    if n_points >= 250000:
        return low
    scale = (np.log10(n_points) - np.log10(1000)) / (np.log10(250000) - np.log10(1000))
    scale = float(np.clip(scale, 0.0, 1.0))
    return high - (high - low) * scale


def _top_label_groups(labels: np.ndarray, max_labels: int = 12) -> tuple[list[str], dict[str, str]]:
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    ranked = unique[order].tolist()
    if len(ranked) <= max_labels:
        return ranked, {label: label for label in ranked}
    keep = ranked[: max_labels - 1]
    mapping = {label: (label if label in keep else "Other") for label in ranked}
    return keep + ["Other"], mapping


def _pick_focal_cell(data: PreparedSpatialOTData) -> int:
    center = np.median(data.cell_coords, axis=0, keepdims=True)
    dist = np.linalg.norm(data.cell_coords - center, axis=1)
    best_idx = int(np.argmin(dist))
    best_score = (0, -dist[best_idx])
    for idx in range(data.n_cells):
        shell_counts = []
        for shell in data.shell_edge_indices:
            if shell.size == 0:
                shell_counts.append(0)
                continue
            shell_counts.append(int(np.sum(shell[0] == idx)))
        score = (sum(c > 0 for c in shell_counts), -dist[idx])
        if score > best_score:
            best_idx = idx
            best_score = score
    return best_idx


def _pick_focal_index_from_coords(coords: np.ndarray) -> int:
    center = np.median(coords, axis=0, keepdims=True)
    dist = np.linalg.norm(coords - center, axis=1)
    return int(np.argmin(dist))


def _plot_shell_neighborhood(ax, data: PreparedSpatialOTData, focal_idx: int) -> None:
    coords = data.cell_coords
    ax.scatter(coords[:, 0], coords[:, 1], s=8, c="#d9d9d9", alpha=0.5, linewidths=0)
    shell_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    shell_labels = ["S0: 0-12 µm", "S1: 12-25 µm", "S2: 25-50 µm"]
    handles: list[Line2D] = []

    for shell_idx, shell in enumerate(data.shell_edge_indices[:3]):
        if shell.size == 0:
            continue
        mask = shell[0] == focal_idx
        neighbor_idx = shell[1][mask]
        if neighbor_idx.size == 0:
            continue
        color = shell_colors[shell_idx]
        for nbr in neighbor_idx:
            ax.plot(
                [coords[focal_idx, 0], coords[nbr, 0]],
                [coords[focal_idx, 1], coords[nbr, 1]],
                color=color,
                alpha=0.15,
                linewidth=0.6,
                zorder=1,
            )
        ax.scatter(coords[neighbor_idx, 0], coords[neighbor_idx, 1], s=18, c=color, alpha=0.8, linewidths=0, zorder=2)
        handles.append(Line2D([0], [0], marker="o", color="w", label=f"{shell_labels[shell_idx]} ({neighbor_idx.size})", markerfacecolor=color, markersize=7))

    ax.scatter(coords[focal_idx, 0], coords[focal_idx, 1], s=90, c="#d62728", marker="*", edgecolors="black", linewidths=0.5, zorder=3)
    handles.insert(0, Line2D([0], [0], marker="*", color="w", label=f"Focal cell: {data.cell_ids[focal_idx]}", markerfacecolor="#d62728", markeredgecolor="black", markersize=11))
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)
    _format_spatial_axis(ax, "Neighborhood shells used by the model")


def _plot_shell_neighborhood_from_coords(
    ax,
    coords: np.ndarray,
    cell_ids: np.ndarray,
    shell_bounds_um: tuple[float, ...],
    focal_idx: int,
    background_size: float,
) -> None:
    ax.scatter(coords[:, 0], coords[:, 1], s=background_size, c="#d9d9d9", alpha=0.35, linewidths=0, rasterized=coords.shape[0] > 20000)
    dist = np.linalg.norm(coords - coords[focal_idx], axis=1)
    lower = 0.0
    shell_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    shell_labels = ["S0", "S1", "S2"]
    handles: list[Line2D] = [
        Line2D([0], [0], marker="*", color="w", label=f"Focal cell: {cell_ids[focal_idx]}", markerfacecolor="#d62728", markeredgecolor="black", markersize=11)
    ]

    for shell_idx, upper in enumerate(shell_bounds_um[:3]):
        mask = (dist > lower) & (dist <= upper)
        mask[focal_idx] = False
        nbr_idx = np.flatnonzero(mask)
        color = shell_colors[shell_idx]
        if nbr_idx.size:
            draw_idx = nbr_idx[: min(nbr_idx.size, 400)]
            for nbr in draw_idx:
                ax.plot(
                    [coords[focal_idx, 0], coords[nbr, 0]],
                    [coords[focal_idx, 1], coords[nbr, 1]],
                    color=color,
                    alpha=0.08,
                    linewidth=0.4,
                    zorder=1,
                )
            ax.scatter(coords[nbr_idx, 0], coords[nbr_idx, 1], s=max(background_size * 4.0, 6.0), c=color, alpha=0.75, linewidths=0, zorder=2, rasterized=coords.shape[0] > 20000)
        handles.append(Line2D([0], [0], marker="o", color="w", label=f"{shell_labels[shell_idx]}: {lower:.0f}-{upper:.0f} µm ({nbr_idx.size})", markerfacecolor=color, markersize=7))
        lower = upper

    ax.scatter(coords[focal_idx, 0], coords[focal_idx, 1], s=90, c="#d62728", marker="*", edgecolors="black", linewidths=0.5, zorder=3)
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)
    _format_spatial_axis(ax, "Neighborhood shells around one focal cell")


def _load_full_view_inputs(config: ExperimentConfig) -> dict[str, object]:
    try:
        cell_adata = ad.read_h5ad(config.paths.cells_h5ad, backed="r")
    except TypeError:
        cell_adata = ad.read_h5ad(config.paths.cells_h5ad)
    coords_px = np.asarray(cell_adata.obsm["spatial"], dtype=np.float32)
    microns_per_pixel = float(
        next(iter(cell_adata.uns["spatial"].values()))["scalefactors"].get("microns_per_pixel", 1.0)
        if "spatial" in cell_adata.uns
        else 1.0
    )
    cell_coords = coords_px * microns_per_pixel
    if config.data.cell_type_key not in cell_adata.obs:
        available = ", ".join(map(str, cell_adata.obs.columns[:20]))
        raise KeyError(f"Requested cell_type_key '{config.data.cell_type_key}' was not found in obs. Available columns include: {available}")
    labels = cell_adata.obs[config.data.cell_type_key].astype(str).to_numpy()
    if "umi_count" in cell_adata.obs:
        cell_umi = np.asarray(cell_adata.obs["umi_count"], dtype=np.float32)
    elif "total_counts" in cell_adata.obs:
        cell_umi = np.asarray(cell_adata.obs["total_counts"], dtype=np.float32)
    else:
        cell_umi = np.ones(cell_adata.n_obs, dtype=np.float32)
    bin_positions = pd.read_parquet(config.paths.bins8_positions)
    if "in_tissue" in bin_positions.columns:
        bin_positions = bin_positions[bin_positions["in_tissue"].astype(bool)].copy()
    bin_coords = np.stack(
        [
            np.asarray(bin_positions["pxl_col_in_fullres"], dtype=np.float32) * microns_per_pixel,
            np.asarray(bin_positions["pxl_row_in_fullres"], dtype=np.float32) * microns_per_pixel,
        ],
        axis=1,
    )
    return {
        "cell_coords": cell_coords,
        "cell_labels": labels,
        "cell_umi": cell_umi,
        "cell_ids": cell_adata.obs_names.astype(str).to_numpy(),
        "bin_coords": bin_coords,
        "microns_per_pixel": microns_per_pixel,
        "n_genes": int(cell_adata.n_vars),
    }


def plot_preprocessed_inputs(
    config: ExperimentConfig,
    cell_subset: int | None = None,
    bin_subset: int | None = None,
    output_path: str | Path | None = None,
) -> Path:
    if cell_subset is not None:
        config.data.cell_subset = int(cell_subset)
    if bin_subset is not None:
        config.data.bin_subset = int(bin_subset)
    if output_path is None:
        output_path = Path(config.paths.output_dir) / "input_2d_overview.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if config.data.cell_subset == 0 and config.data.bin_subset == 0:
        view = _load_full_view_inputs(config)
        cell_coords = view["cell_coords"]
        cell_labels = view["cell_labels"]
        cell_umi = view["cell_umi"]
        cell_ids = view["cell_ids"]
        bin_coords = view["bin_coords"]
        n_genes = int(view["n_genes"])

        fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
        ax_cells, ax_cell_umi, ax_bins, ax_shells = axes.ravel()
        cell_size = _marker_size(len(cell_coords), low=0.15, high=10.0)
        bin_size = _marker_size(len(bin_coords), low=0.15, high=10.0)

        shown_labels, mapping = _top_label_groups(cell_labels, max_labels=12)
        palette = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(shown_labels), 2)))
        label_to_color = {label: palette[i] for i, label in enumerate(shown_labels)}
        grouped = np.array([mapping[label] for label in cell_labels], dtype=object)

        for label in shown_labels:
            mask = grouped == label
            if not np.any(mask):
                continue
            ax_cells.scatter(
                cell_coords[mask, 0],
                cell_coords[mask, 1],
                s=cell_size,
                color=label_to_color[label],
                alpha=0.85,
                linewidths=0,
                label=f"{label} ({int(mask.sum())})",
                rasterized=True,
            )
        ax_cells.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
        _format_spatial_axis(ax_cells, f"All segmented cells by {config.data.cell_type_key}")

        sc1 = ax_cell_umi.scatter(
            cell_coords[:, 0],
            cell_coords[:, 1],
            c=np.log1p(cell_umi),
            cmap="viridis",
            s=cell_size,
            linewidths=0,
            rasterized=True,
        )
        fig.colorbar(sc1, ax=ax_cell_umi, fraction=0.046, pad=0.04, label="log1p(cell UMI)")
        _format_spatial_axis(ax_cell_umi, "All cells by library size")

        ax_bins.scatter(
            bin_coords[:, 0],
            bin_coords[:, 1],
            s=bin_size,
            c="#6a3d9a",
            marker="s",
            linewidths=0,
            alpha=0.8,
            rasterized=True,
        )
        _format_spatial_axis(ax_bins, "All 8 µm teacher bin positions")

        focal_idx = _pick_focal_index_from_coords(cell_coords)
        _plot_shell_neighborhood_from_coords(
            ax_shells,
            coords=cell_coords,
            cell_ids=cell_ids,
            shell_bounds_um=tuple(config.data.shell_bounds_um),
            focal_idx=focal_idx,
            background_size=cell_size,
        )

        fig.suptitle(
            (
                "spatial_ot full preprocessed input overview\n"
                f"cells={len(cell_coords)}, bins={len(bin_coords)}, genes={n_genes}"
            ),
            fontsize=14,
        )
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
        return output_path

    data = prepare_data(config)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    ax_cells, ax_cell_umi, ax_bins, ax_shells = axes.ravel()
    cell_size = _marker_size(data.n_cells, low=0.5, high=10.0)
    bin_size = _marker_size(data.n_bins, low=0.5, high=18.0)

    labels = data.cell_adata.obs[config.data.cell_type_key].astype(str).to_numpy()
    shown_labels, mapping = _top_label_groups(labels, max_labels=12)
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(shown_labels), 2)))
    label_to_color = {label: palette[i] for i, label in enumerate(shown_labels)}
    grouped = np.array([mapping[label] for label in labels], dtype=object)

    for label in shown_labels:
        mask = grouped == label
        if not np.any(mask):
            continue
        ax_cells.scatter(
            data.cell_coords[mask, 0],
            data.cell_coords[mask, 1],
            s=cell_size,
            color=label_to_color[label],
            alpha=0.9,
            linewidths=0,
            label=f"{label} ({int(mask.sum())})",
            rasterized=data.n_cells > 20000,
        )
    ax_cells.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    _format_spatial_axis(ax_cells, f"Segmented cells by {config.data.cell_type_key}")

    cell_umi = np.log1p(data.cell_library_full[:, 0])
    sc1 = ax_cell_umi.scatter(
        data.cell_coords[:, 0],
        data.cell_coords[:, 1],
        c=cell_umi,
        cmap="viridis",
        s=cell_size,
        linewidths=0,
        rasterized=data.n_cells > 20000,
    )
    fig.colorbar(sc1, ax=ax_cell_umi, fraction=0.046, pad=0.04, label="log1p(cell UMI)")
    _format_spatial_axis(ax_cell_umi, "Cell library size")

    bin_umi = np.log1p(data.bin_library_full[:, 0])
    sc2 = ax_bins.scatter(
        data.bin_coords[:, 0],
        data.bin_coords[:, 1],
        c=bin_umi,
        cmap="magma",
        s=bin_size,
        marker="s",
        linewidths=0,
        rasterized=data.n_bins > 20000,
    )
    fig.colorbar(sc2, ax=ax_bins, fraction=0.046, pad=0.04, label="log1p(8 µm bin UMI)")
    _format_spatial_axis(ax_bins, "8 µm teacher bins")

    focal_idx = _pick_focal_cell(data)
    _plot_shell_neighborhood(ax_shells, data, focal_idx)

    fig.suptitle(
        (
            "spatial_ot preprocessed input overview\n"
            f"cells={data.n_cells}, bins={data.n_bins}, genes={data.n_genes}, "
            f"shell edges={[shell.shape[1] for shell in data.shell_edge_indices]}"
        ),
        fontsize=14,
    )
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path
