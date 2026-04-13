from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def load_efficiency_table(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Load radius ratio axis, larger radius axis, and efficiency matrix from CSV."""
	with csv_path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.reader(f)
		rows = list(reader)

	if len(rows) < 2:
		raise ValueError(f"The CSV file is empty or incomplete: {csv_path}")

	radius_ratio = np.array([float(v) for v in rows[0][1:]], dtype=float)
	larger_radius = np.array([float(r[0]) for r in rows[1:]], dtype=float)
	efficiency = np.array([[float(v) for v in r[1:]] for r in rows[1:]], dtype=float)

	if efficiency.shape != (larger_radius.size, radius_ratio.size):
		raise ValueError(
			"Table shape mismatch: "
			f"got {efficiency.shape}, "
			f"expected {(larger_radius.size, radius_ratio.size)}"
		)

	return radius_ratio, larger_radius, efficiency


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
	output_path = Path(__file__).with_name("001_tableplot.png")

	radius_ratio, larger_radius, efficiency = load_efficiency_table(csv_path)
	if np.any(efficiency <= 0.0):
		raise ValueError("Log color scale requires all efficiency values to be positive.")

	x_edges = bin_edges(radius_ratio)
	y_edges = bin_edges(larger_radius)
	norm = LogNorm(vmin=float(efficiency.min()), vmax=float(efficiency.max()))

	fig, ax = plt.subplots(figsize=(10, 6))
	mesh = ax.pcolormesh(
		x_edges,
		y_edges,
		efficiency,
		shading="auto",
		cmap="viridis",
		norm=norm,
	)

	cbar = fig.colorbar(mesh, ax=ax)
	cbar.set_label("Collision efficiency (log scale)")

	ax.set_xlabel("Radius ratio")
	ax.set_ylabel("Larger droplet radius (um, log scale)")
	ax.set_yscale("log")
	ax.set_yticks(larger_radius)
	ax.set_yticklabels([
		f"{int(v)}" if float(v).is_integer() else f"{v:g}" for v in larger_radius
	])
	ax.set_title("Collision efficiency heatmap")
	ax.set_xlim(radius_ratio.min(), radius_ratio.max())
	ax.set_ylim(larger_radius.min(), larger_radius.max())

	fig.tight_layout()
	fig.savefig(output_path, dpi=300)
	# plt.show()


if __name__ == "__main__":
	main()
