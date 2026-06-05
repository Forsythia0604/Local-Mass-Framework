from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments.common import result_dirs, save_simple_png_figure


def _plot_candidates(table: pd.DataFrame, centre: str) -> list[str]:
    out: list[str] = []
    for method in ["q_MF", "q_FR", "q_b"]:
        sub = table[(table["VI_method"] == method) & (table["centre_type"] == centre)]
        if not sub.empty:
            out.append(str(sub.sort_values("ELBO_best", ascending=False).iloc[0]["VI_candidate_name"]))
    return out


def _method_label(candidate: str) -> str:
    return "q_b" if candidate.startswith("q_b") else candidate


def main() -> None:
    dirs = result_dirs("exp3")
    table = pd.read_csv(dirs["processed"] / "exp3_vi_family_comparison.csv")
    arrays = np.load(dirs["processed"] / "exp3_mnist_arrays.npz")
    reliable_nonzero = table[
        (table["centre_type"] == "nonzero")
        & (table["reliable_radius_count_p_scale"] > 0)
        & (table["reliable_radius_count_q_scale"] > 0)
    ]
    selected_centre = "nonzero" if not reliable_nonzero.empty else "zero"
    methods = _plot_candidates(table, selected_centre)
    if not methods:
        raise RuntimeError("No VI methods found for plotting.")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        first = methods[0]
        radii = arrays[f"{first}_{selected_centre}_radii"]
        save_simple_png_figure(
            dirs["figures"] / "exp3_local_mass_scaling.png",
            [
                {
                    "title": f"local mass, {selected_centre}",
                    "series": [{"x": radii, "y": arrays[f"{first}_{selected_centre}_p_mass_for_display_only"], "label": "p"}]
                    + [
                        {
                            "x": arrays[f"{candidate}_{selected_centre}_radii"],
                            "y": arrays[f"{candidate}_{selected_centre}_q_mass_for_display_only"],
                            "label": _method_label(candidate),
                        }
                        for candidate in methods
                    ],
                    "ylabel": "display local mass",
                    "ylog": True,
                }
            ],
            "EXP3 local mass scaling",
        )
        save_simple_png_figure(
            dirs["figures"] / "exp3_p_scale_normalised_divergence.png",
            [
                {
                    "title": "p-scale N",
                    "series": [
                        {
                            "x": arrays[f"{candidate}_{selected_centre}_radii"],
                            "y": arrays[f"{candidate}_{selected_centre}_N_p_q_over_p_for_plot"],
                            "label": _method_label(candidate),
                        }
                        for candidate in methods
                    ],
                    "ylabel": "display N",
                    "ylog": True,
                }
            ],
            "EXP3 p-scale normalised divergence",
        )
        save_simple_png_figure(
            dirs["figures"] / "exp3_q_scale_normalised_divergence.png",
            [
                {
                    "title": "q-scale N",
                    "series": [
                        {
                            "x": arrays[f"{candidate}_{selected_centre}_radii"],
                            "y": arrays[f"{candidate}_{selected_centre}_N_q_p_over_q_for_plot"],
                            "label": _method_label(candidate),
                        }
                        for candidate in methods
                    ],
                    "ylabel": "display N",
                    "ylog": True,
                }
            ],
            "EXP3 q-scale normalised divergence",
        )
        return

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 6})
    colours = {"q_MF": "#2f6f9f", "q_FR": "#c49a2c", "q_b": "#9d3f54"}

    fig, ax = plt.subplots(figsize=(3.55, 2.75))
    first = methods[0]
    radii = arrays[f"{first}_{selected_centre}_radii"]
    ax.loglog(radii, arrays[f"{first}_{selected_centre}_p_mass_for_display_only"], color="0.15", marker="o", ms=2.2, lw=1.0, label="p")
    for candidate in methods:
        label = _method_label(candidate)
        rr = arrays[f"{candidate}_{selected_centre}_radii"]
        ax.loglog(rr, arrays[f"{candidate}_{selected_centre}_q_mass_for_display_only"], marker="s", ms=2.1, lw=0.95, color=colours[label], label=label)
    ax.set_title(f"EXP3 local mass scaling, {selected_centre} centre")
    ax.set_xlabel("radius r")
    ax.set_ylabel("display local mass")
    ax.grid(True, which="both", lw=0.25, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.7)
    fig.savefig(dirs["figures"] / "exp3_local_mass_scaling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.55, 2.75))
    for candidate in methods:
        label = _method_label(candidate)
        rr = arrays[f"{candidate}_{selected_centre}_radii"]
        ax.loglog(rr, arrays[f"{candidate}_{selected_centre}_N_p_q_over_p_for_plot"], lw=0.95, color=colours[label], label=label)
    ax.set_title(r"Reference-scale $N_{\alpha,r}^{p}(q\|p)$")
    ax.set_xlabel("radius r")
    ax.set_ylabel("display normalised divergence")
    ax.grid(True, which="both", lw=0.25, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.7)
    fig.savefig(dirs["figures"] / "exp3_p_scale_normalised_divergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.55, 2.75))
    for candidate in methods:
        label = _method_label(candidate)
        rr = arrays[f"{candidate}_{selected_centre}_radii"]
        ax.loglog(rr, arrays[f"{candidate}_{selected_centre}_N_q_p_over_q_for_plot"], lw=0.95, color=colours[label], label=label)
    ax.set_title(r"VI-scale $N_{\alpha,r}^{q}(p\|q)$")
    ax.set_xlabel("radius r")
    ax.set_ylabel("display normalised divergence")
    ax.grid(True, which="both", lw=0.25, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.7)
    fig.savefig(dirs["figures"] / "exp3_q_scale_normalised_divergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
