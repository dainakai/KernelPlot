from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def load_efficiency_table(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Load radius-ratio axis, larger-radius axis, and efficiency matrix from CSV."""
	with csv_path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.reader(f)
		rows = list(reader)

	if len(rows) < 2:
		raise ValueError(f"The CSV file is empty or incomplete: {csv_path}")

	radius_ratio = np.array([float(v) for v in rows[0][1:]], dtype=float)
	larger_radius_um = np.array([float(r[0]) for r in rows[1:]], dtype=float)
	efficiency = np.array([[float(v) for v in r[1:]] for r in rows[1:]], dtype=float)

	if efficiency.shape != (larger_radius_um.size, radius_ratio.size):
		raise ValueError(
			"Table shape mismatch: "
			f"got {efficiency.shape}, "
			f"expected {(larger_radius_um.size, radius_ratio.size)}"
		)
	if radius_ratio.size < 2 or larger_radius_um.size < 2:
		raise ValueError("Both axes must contain at least two values.")
	if not np.all(np.diff(radius_ratio) > 0):
		raise ValueError("Radius-ratio axis must be strictly increasing.")
	if not np.all(np.diff(larger_radius_um) > 0):
		raise ValueError("Larger-radius axis must be strictly increasing.")
	if np.any(efficiency <= 0.0):
		raise ValueError("Log color scale requires all efficiency values to be positive.")

	return radius_ratio, larger_radius_um, efficiency


def build_radius_axis_5um(larger_radius_um: np.ndarray, step_um: float = 5.0) -> np.ndarray:
	"""Build a 5-um-spaced radius axis within the original larger-radius range."""
	if step_um <= 0.0:
		raise ValueError("Interpolation step must be positive.")

	start = step_um * np.ceil(float(larger_radius_um.min()) / step_um)
	stop = step_um * np.floor(float(larger_radius_um.max()) / step_um)
	if stop < start:
		raise ValueError("No 5-um points fall inside the larger-radius range.")

	count = int(round((stop - start) / step_um)) + 1
	return start + step_um * np.arange(count, dtype=float)


def bilinear_interpolate_on_grid(
	x_axis: np.ndarray,
	y_axis: np.ndarray,
	values: np.ndarray,
	x_query: np.ndarray,
	y_query: np.ndarray,
) -> np.ndarray:
	"""Evaluate values(x, y) on a rectangular grid with bilinear interpolation."""
	x_axis = np.asarray(x_axis, dtype=float)
	y_axis = np.asarray(y_axis, dtype=float)
	values = np.asarray(values, dtype=float)
	x_query = np.asarray(x_query, dtype=float)
	y_query = np.asarray(y_query, dtype=float)

	if values.shape != (y_axis.size, x_axis.size):
		raise ValueError("Value matrix shape must be (len(y_axis), len(x_axis)).")
	if x_query.shape != y_query.shape:
		raise ValueError("x_query and y_query must have the same shape.")

	result = np.full(x_query.shape, np.nan, dtype=float)
	in_bounds = (
		np.isfinite(x_query)
		& np.isfinite(y_query)
		& (x_query >= x_axis[0])
		& (x_query <= x_axis[-1])
		& (y_query >= y_axis[0])
		& (y_query <= y_axis[-1])
	)
	if not np.any(in_bounds):
		return result

	xq = x_query[in_bounds]
	yq = y_query[in_bounds]

	ix1 = np.searchsorted(x_axis, xq, side="right")
	iy1 = np.searchsorted(y_axis, yq, side="right")
	ix1 = np.clip(ix1, 1, x_axis.size - 1)
	iy1 = np.clip(iy1, 1, y_axis.size - 1)
	ix0 = ix1 - 1
	iy0 = iy1 - 1

	x0 = x_axis[ix0]
	x1 = x_axis[ix1]
	y0 = y_axis[iy0]
	y1 = y_axis[iy1]

	wx = np.divide(
		xq - x0,
		x1 - x0,
		out=np.zeros_like(xq),
		where=(x1 > x0),
	)
	wy = np.divide(
		yq - y0,
		y1 - y0,
		out=np.zeros_like(yq),
		where=(y1 > y0),
	)

	f00 = values[iy0, ix0]
	f10 = values[iy0, ix1]
	f01 = values[iy1, ix0]
	f11 = values[iy1, ix1]

	interpolated = (
		(1.0 - wx) * (1.0 - wy) * f00
		+ wx * (1.0 - wy) * f10
		+ (1.0 - wx) * wy * f01
		+ wx * wy * f11
	)
	result[in_bounds] = interpolated
	return result


def radius_radius_efficiency_matrix(
	radius_ratio: np.ndarray,
	larger_radius_um: np.ndarray,
	efficiency: np.ndarray,
	radius_axis_um: np.ndarray,
) -> np.ndarray:
	"""Convert E(R, r/R) into a radius-radius matrix E(r_small, r_large)."""
	r_i_um, r_j_um = np.meshgrid(radius_axis_um, radius_axis_um)
	r_small_um = np.minimum(r_i_um, r_j_um)
	r_large_um = np.maximum(r_i_um, r_j_um)
	ratio_query = np.divide(
		r_small_um,
		r_large_um,
		out=np.full_like(r_small_um, np.nan, dtype=float),
		where=(r_large_um > 0.0),
	)

	interpolated = bilinear_interpolate_on_grid(
		x_axis=radius_ratio,
		y_axis=larger_radius_um,
		values=efficiency,
		x_query=ratio_query,
		y_query=r_large_um,
	)
	return interpolated


def bin_edges(values: np.ndarray) -> np.ndarray:
	"""Convert monotonic center coordinates into pcolormesh edge coordinates."""
	values = np.asarray(values, dtype=float)
	if values.ndim != 1:
		raise ValueError("Axis values must be one-dimensional.")
	if values.size < 2:
		raise ValueError("At least two axis values are required.")
	if not np.all(np.diff(values) > 0):
		raise ValueError("Axis values must be strictly increasing.")

	edges = np.empty(values.size + 1, dtype=float)
	edges[1:-1] = 0.5 * (values[:-1] + values[1:])
	edges[0] = values[0] - 0.5 * (values[1] - values[0])
	edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
	return edges


def main() -> None:
	project_root = Path(__file__).resolve().parents[2]
	csv_path = project_root / "data" / "SeeßelbergAndBott" / "ecoll_table.csv"
	output_path = Path(__file__).with_name("002_ecoll_heatmap.png")

	radius_ratio, larger_radius_um, efficiency = load_efficiency_table(csv_path)
	radius_axis_um = build_radius_axis_5um(larger_radius_um, step_um=5.0)
	eff_radius_radius = radius_radius_efficiency_matrix(
		radius_ratio,
		larger_radius_um,
		efficiency,
		radius_axis_um,
	)

	positive_values = eff_radius_radius[
		np.isfinite(eff_radius_radius) & (eff_radius_radius > 0.0)
	]
	if positive_values.size == 0:
		raise ValueError("No interpolated values were available for plotting.")
	plot_values = np.where(
		np.isfinite(eff_radius_radius) & (eff_radius_radius > 0.0),
		eff_radius_radius,
		np.nan,
	)

	x_edges = bin_edges(radius_axis_um)
	y_edges = bin_edges(radius_axis_um)
	norm = LogNorm(vmin=float(positive_values.min()), vmax=float(positive_values.max()))

	fig, ax = plt.subplots(figsize=(8, 8))
	mesh = ax.pcolormesh(
		x_edges,
		y_edges,
		plot_values,
		shading="auto",
		cmap="viridis",
		norm=norm,
	)

	cbar = fig.colorbar(mesh, ax=ax)
	cbar.set_label("Collision efficiency E_coll (log scale)")

	tick_candidates = np.array([10, 25, 50, 75, 100, 150, 200, 250, 300], dtype=float)
	ticks = tick_candidates[
		(tick_candidates >= radius_axis_um.min()) & (tick_candidates <= radius_axis_um.max())
	]

	ax.set_xlabel("Radius i (um)")
	ax.set_ylabel("Radius j (um)")
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_title("Collision efficiency heatmap (radius-radius, bilinear interpolation)")
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlim(float(radius_axis_um.min()), float(radius_axis_um.max()))
	ax.set_ylim(float(radius_axis_um.min()), float(radius_axis_um.max()))

	fig.tight_layout()
	fig.savefig(output_path, dpi=300)
	# plt.show()


if __name__ == "__main__":
	main()
