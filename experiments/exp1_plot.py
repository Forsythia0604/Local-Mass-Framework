from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments.common import result_dirs, save_simple_png_figure


def main() -> None:
    dirs = result_dirs("exp1")
    table = pd.read_csv(dirs["processed"] / "exp1_kappa_mi_table.csv")
    arrays = np.load(dirs["processed"] / "exp1_local_exponent_arrays.npz")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        for dataset in table["dataset"].drop_duplicates():
            for ctype in ["zero", "nonzero"]:
                sub = table[(table["dataset"] == dataset) & (table["centre_type"] == ctype)]
                if sub.empty:
                    continue
                series = []
                for _, row in sub.iterrows():
                    key = f"{dataset}_a{row['a']:g}_{ctype}"
                    prior_log = arrays[f"{key}_prior_log_mass"]
                    post_log = arrays[f"{key}_posterior_log_mass"]
                    prior_mi = 1.0 / float(row["a"])
                    series.append(
                        {
                            "x": arrays[f"{key}_radii"],
                            "y": prior_log - prior_log[np.isfinite(prior_log)][-1],
                            "label": f"prior MI={prior_mi:.2f}",
                        }
                    )
                    series.append(
                        {
                            "x": arrays[f"{key}_radii"],
                            "y": post_log - post_log[np.isfinite(post_log)][-1],
                            "label": f"posterior MI={row['posterior_MI_estimated']:.2f}",
                        }
                    )
                save_simple_png_figure(
                    dirs["figures"] / f"exp1_local_mass_scaling_{dataset}_{ctype}.png",
                    [{"title": f"{dataset}, {ctype}", "series": series, "ylabel": "centred log mass", "ylog": False}],
                    f"EXP1 local mass scaling, {dataset}, {ctype}",
                )
        return

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "legend.fontsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
        }
    )
    colours = {"prior": "#2f6f9f", "posterior": "#bd5d38"}
    for dataset in table["dataset"].drop_duplicates():
        for ctype in ["zero", "nonzero"]:
            sub = table[(table["dataset"] == dataset) & (table["centre_type"] == ctype)]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(3.55, 2.75))
            for _, row in sub.iterrows():
                key = f"{dataset}_a{row['a']:g}_{ctype}"
                radii = arrays[f"{key}_radii"]
                prior_log = arrays[f"{key}_prior_log_mass"]
                post_log = arrays[f"{key}_posterior_log_mass"]
                prior_y = prior_log - prior_log[np.isfinite(prior_log)][-1]
                post_y = post_log - post_log[np.isfinite(post_log)][-1]
                prior_style = "-" if bool(row["prior_fit_reliable"]) else "--"
                post_style = "-" if bool(row["posterior_fit_reliable"]) else ":"
                prior_mi = 1.0 / float(row["a"])
                ax.plot(
                    radii,
                    prior_y,
                    prior_style,
                    marker="o",
                    ms=2.2,
                    lw=0.9,
                    color=colours["prior"],
                    alpha=0.76,
                    label=f"prior MI={prior_mi:.2f}",
                )
                ax.plot(
                    radii,
                    post_y,
                    post_style,
                    marker="s",
                    ms=2.2,
                    lw=0.9,
                    color=colours["posterior"],
                    alpha=0.80,
                    label=f"posterior MI={row['posterior_MI_estimated']:.2f}",
                )
                if np.isfinite(row["fit_radius_min"]) and np.isfinite(row["fit_radius_max"]):
                    ax.axvspan(float(row["fit_radius_min"]), float(row["fit_radius_max"]), color="0.2", alpha=0.04)
            ax.set_xscale("log")
            ax.set_title(f"EXP1 {dataset}, {ctype} centre")
            ax.set_xlabel("radius r")
            ax.set_ylabel(r"$\log m(B_r)-\log m(B_{r_{\max}})$")
            ax.grid(True, which="both", lw=0.25, alpha=0.35)
            ax.legend(frameon=False, loc="best")
            fig.tight_layout(pad=0.7)
            out = dirs["figures"] / f"exp1_local_mass_scaling_{dataset}_{ctype}.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
