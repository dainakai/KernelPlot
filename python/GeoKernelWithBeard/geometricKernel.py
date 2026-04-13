from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def load_terminal_velocity_table(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
	"""Load droplet diameter (um) and terminal velocity (m/s) arrays from CSV."""
	with csv_path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		rows = list(reader)

	if not rows:
		raise ValueError(f"The CSV file is empty: {csv_path}")

	diameters_um = np.array([float(r["Diameter_um"]) for r in rows], dtype=float)
	terminal_velocities_m_s = np.array(
		[float(r["TerminalVelocity_m_s"]) for r in rows],
		dtype=float,
	)

	if diameters_um.size < 2:
		raise ValueError("At least two diameter values are required.")
	if diameters_um.shape != terminal_velocities_m_s.shape:
		raise ValueError(
			"Diameter and terminal-velocity arrays must have the same shape."
		)
	if not np.all(np.diff(diameters_um) > 0):
		raise ValueError("Diameter values must be strictly increasing.")
	if np.any(terminal_velocities_m_s <= 0.0):
		raise ValueError("Terminal velocity values must be strictly positive.")

	return diameters_um, terminal_velocities_m_s


def geometric_kernel_matrix(
	diameters_um: np.ndarray,
	terminal_velocities_m_s: np.ndarray,
) -> np.ndarray:
	"""Compute geometric collision kernel matrix K = pi*(r_i+r_j)^2*|v_i-v_j| in SI."""
	radii_m = 0.5 * diameters_um * 1.0e-6
	radius_sum = radii_m[:, None] + radii_m[None, :]
	velocity_diff = np.abs(
		terminal_velocities_m_s[:, None] - terminal_velocities_m_s[None, :]
	)

	kernel_m3_s = np.pi * radius_sum**2 * velocity_diff
	return kernel_m3_s


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
	csv_path = Path(__file__).with_name("beard1976_terminal_velocity.csv")
	output_path = Path(__file__).with_name("geometric_kernel_heatmap.png")

	diameters_um_all, terminal_velocities_m_s_all = load_terminal_velocity_table(csv_path)
	radii_um_all = 0.5 * diameters_um_all
	valid_mask = radii_um_all <= 300.0
	diameters_um = diameters_um_all[valid_mask]
	terminal_velocities_m_s = terminal_velocities_m_s_all[valid_mask]
	radii_um = 0.5 * diameters_um

	if diameters_um.size < 2:
		raise ValueError("Need at least two samples up to radius 300 um for plotting.")

	kernel_m3_s = geometric_kernel_matrix(diameters_um, terminal_velocities_m_s)
	positive_values = kernel_m3_s[kernel_m3_s > 0.0]
	if positive_values.size == 0:
		raise ValueError("No positive kernel values found for log color scale.")
	plot_values = np.where(kernel_m3_s > 0.0, kernel_m3_s, np.nan)

	x_edges = bin_edges(radii_um)
	y_edges = bin_edges(radii_um)
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
	cbar.set_label("Geometric kernel K_g (m^3/s, log scale)")

	radius_lower = 10.0
	radius_upper = float(radii_um.max())
	if radius_upper < radius_lower:
		raise ValueError("Radius upper limit is below the requested tick minimum 10 um.")
	ticks = np.concatenate(
		(np.array([radius_lower]), np.arange(50.0, radius_upper + 1.0e-12, 50.0))
	)
	ticks = np.unique(ticks[ticks <= radius_upper])

	ax.set_xlabel("Radius i (um)")
	ax.set_ylabel("Radius j (um)")
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_title("Geometric collision kernel heatmap")
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlim(radius_lower, radius_upper)
	ax.set_ylim(radius_lower, radius_upper)

	fig.tight_layout()
	fig.savefig(output_path, dpi=300)
	# plt.show()


if __name__ == "__main__":
	main()
