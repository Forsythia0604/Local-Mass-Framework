from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments.common import result_dirs, save_simple_png_figure


def _selected_centre(table: pd.DataFrame) -> str:
    nonzero = table[(table["centre_type"] == "nonzero") & (table["reliable_radius_count"] > 0)]
    if not nonzero.empty:
        return "nonzero"
    return "zero"


def main() -> None:
    dirs = result_dirs("exp2")
    table = pd.read_csv(dirs["processed"] / "exp2_kappa_mi_divergence_table.csv")
    arrays = np.load(dirs["processed"] / "exp2_divergence_arrays.npz")
    selected_centre = _selected_centre(table)
    radii = arrays[f"{selected_centre}_radii"]
    cases = ["q1", "q2", "q3"]
    colours = {"q1": "#2f6f9f", "q2": "#c49a2c", "q3": "#9d3f54"}
    labels = {"q1": "q1 close", "q2": "q2 MI-matched distorted", "q3": "q3 MI-mismatched"}

    try:
        import matplotlib.pyplot as plt
    except Exception:
        save_simple_png_figure(
            dirs["figures"] / "exp2_raw_local_divergence.png",
            [
                {
                    "title": "raw divergence",
                    "series": [{"x": radii, "y": arrays[f"{case}_{selected_centre}_raw_divergence_for_plot"], "label": case} for case in cases],
                    "ylabel": "display raw D",
                    "ylog": True,
                }
            ],
            "EXP2 raw local divergence",
        )
        save_simple_png_figure(
            dirs["figures"] / "exp2_normalised_local_divergence.png",
            [
                {
                    "title": "normalised divergence",
                    "series": [{"x": radii, "y": arrays[f"{case}_{selected_centre}_normalised_divergence_for_plot"], "label": case} for case in cases],
                    "ylabel": "display N",
                    "ylog": True,
                }
            ],
            "EXP2 normalised local divergence",
        )
        save_simple_png_figure(
            dirs["figures"] / "exp2_mi_vs_local_discrepancy.png",
            [
                {
                    "title": "MI vs discrepancy",
                    "series": [
                        {
                            "x": [table[(table["case"] == case) & (table["centre_type"] == selected_centre)].iloc[0]["estimated_MI"]],
                            "y": [np.exp(np.clip(table[(table["case"] == case) & (table["centre_type"] == selected_centre)].iloc[0]["log_normalised_divergence_smallest_r"], -50, 50))],
                            "label": case,
                        }
                        for case in cases
                    ],
                    "xlabel": "estimated MI",
                    "ylabel": "display N",
                    "xlog": False,
                    "ylog": True,
                }
            ],
            "EXP2 MI versus local discrepancy",
        )
        return

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 6})

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for case in cases:
        prefix = f"{case}_{selected_centre}"
        ax.loglog(radii, arrays[f"{prefix}_raw_divergence_for_plot"], marker="o", ms=2.4, lw=0.95, color=colours[case], label=labels[case])
    ax.set_title(r"Raw local divergence $D_\alpha^{B_r}(q\|p)$")
    ax.set_xlabel("radius r")
    ax.set_ylabel("display raw divergence")
    ax.grid(True, which="both", lw=0.25, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.7)
    fig.savefig(dirs["figures"] / "exp2_raw_local_divergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for case in cases:
        prefix = f"{case}_{selected_centre}"
        ax.loglog(radii, arrays[f"{prefix}_normalised_divergence_for_plot"], marker="s", ms=2.4, lw=0.95, color=colours[case], label=labels[case])
    ax.set_title(r"Normalised local divergence $N_{\alpha,r}^{p}(q\|p)$")
    ax.set_xlabel("radius r")
    ax.set_ylabel("display normalised divergence")
    ax.grid(True, which="both", lw=0.25, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.7)
    fig.savefig(dirs["figures"] / "exp2_normalised_local_divergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for case in cases:
        row = table[(table["case"] == case) & (table["centre_type"] == selected_centre)].iloc[0]
        yval = float(np.exp(np.clip(row["log_normalised_divergence_smallest_r"], -50.0, 50.0)))
        ax.scatter([row["estimated_MI"]], [yval], s=34, color=colours[case], label=case)
        ax.annotate(case, (row["estimated_MI"], yval), fontsize=6)
    ax.set_yscale("log")
    ax.set_title("MI versus local discrepancy")
    ax.set_xlabel("estimated MI")
    ax.set_ylabel("display N at smallest r")
    ax.grid(True, which="both", lw=0.25, alpha=0.35)
    fig.tight_layout(pad=0.7)
    fig.savefig(dirs["figures"] / "exp2_mi_vs_local_discrepancy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
