from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = PROJECT_ROOT / ".plot_deps"
if LOCAL_DEPS.exists():
    sys.path.append(str(LOCAL_DEPS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


PNG_DPI = 360
SINGLE = (3.35, 2.25)
SINGLE_TALL = (3.35, 2.65)
DOUBLE_SHORT = (6.95, 2.35)
DOUBLE = (6.95, 3.25)
DOUBLE_TALL = (6.95, 3.8)

COLORS = {
    "gaussian": "#4C78A8",
    "laplace": "#F58518",
    "student_t": "#54A24B",
    "mean_field_gaussian": "#E45756",
    "diagonal_gaussian": "#E45756",
    "diagonal_gaussian_mixture_2": "#4C78A8",
    "exclusive_kl": "#B279A2",
    "inclusive_kl": "#59A14F",
}
MARKERS = {"exclusive_kl": "o", "inclusive_kl": "s"}
LINESTYLES = {"exclusive_kl": "-", "inclusive_kl": "--"}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.8,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.8,
            "axes.linewidth": 0.55,
            "axes.titlepad": 2.5,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "legend.fontsize": 5.8,
            "legend.frameon": False,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.0,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def latest_run(experiments: str | list[str]) -> Path | None:
    names = [experiments] if isinstance(experiments, str) else experiments
    runs = []
    for experiment in names:
        root = PROJECT_ROOT / "results" / experiment
        if root.exists():
            runs.extend(path for path in root.iterdir() if path.is_dir())
    runs = sorted(runs, key=lambda path: path.name)
    return runs[-1] if runs else None


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def ensure_figures(run_dir: Path) -> Path:
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    for path in figures.glob("*.pdf"):
        path.unlink()
    return figures


def save_png(fig: plt.Figure, figures: Path, stem: str) -> None:
    path = figures / f"{stem}.png"
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def short_prior(name: str) -> str:
    return {"student_t": "Student-t", "laplace": "Laplace", "gaussian": "Gaussian"}.get(str(name), str(name))


def short_family(name: str) -> str:
    return {
        "mean_field_gaussian": "Mean-field Gaussian",
        "diagonal_gaussian": "Diag. Gaussian",
        "diagonal_gaussian_mixture_2": "2-comp. mixture",
    }.get(str(name), str(name))


def short_objective(name: str) -> str:
    return {"exclusive_kl": "Excl. KL", "inclusive_kl": "Incl. KL", "kl_q_p": "KL(q||p)"}.get(str(name), str(name))


def log_safe(values: pd.Series | np.ndarray, floor: float = 1.0e-300) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.log(np.maximum(arr, floor))


def grouped_quantiles(df: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    return (
        df.groupby(keys, as_index=False)[value]
        .agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
        .sort_values(keys)
    )


def plot_exp1() -> None:
    run_dir = latest_run(["exp1_small_bayes_regression", "exp1_bayesian_update"])
    if run_dir is None:
        return
    figures = ensure_figures(run_dir)
    metrics = run_dir / "metrics"
    radius = read_csv(metrics / "exp1_coordinate_radius_metrics.csv")
    slopes = read_csv(metrics / "exp1_local_slope_metrics.csv")
    hessian = read_csv(metrics / "exp1_hessian_diagnostics.csv")
    inactive = read_csv(metrics / "exp1_inactive_summary.csv")
    active = read_csv(metrics / "exp1_active_diagnostic_summary.csv")

    if radius.empty:
        raw = read_csv(run_dir / "raw_metrics.csv")
        summary = read_csv(run_dir / "summary_metrics.csv")
        if raw.empty:
            return
        radius = raw.rename(
            columns={
                "r": "radius",
                "coordinate_index": "coordinate",
                "ratio_posterior_to_prior": "mass_ratio",
                "m_prior": "prior_mass",
                "m_posterior": "post_mass",
            }
        )
        radius["is_active"] = radius["coordinate_type"].eq("active")
        if not summary.empty:
            slope_parts = []
            for source, target in [("estimated_prior_slope", "prior"), ("estimated_posterior_slope", "posterior")]:
                if source in summary:
                    slope_parts.append(
                        summary.rename(columns={source: "slope"})
                        .assign(mass_kind=target, mid_radius=radius["radius"].median())
                    )
            slopes = pd.concat(slope_parts, ignore_index=True) if slope_parts else pd.DataFrame()
            if not slopes.empty and "coordinate_type" in slopes:
                slopes["is_active"] = slopes["coordinate_type"].eq("active")
        log_ratios = radius.assign(log_mass_ratio=log_safe(radius["mass_ratio"]))
        compact = (
            log_ratios.groupby(["prior_name", "is_active"], as_index=False)
            .agg(
                mean_log_mass_ratio=("log_mass_ratio", "mean"),
                median_mass_ratio=("mass_ratio", "median"),
                row_count=("mass_ratio", "size"),
            )
        )
        inactive = compact[~compact["is_active"]].drop(columns=["is_active"])
        active = compact[compact["is_active"]].drop(columns=["is_active"])

    if not radius.empty:
        inactive_rows = radius[~radius["is_active"].astype(bool)].copy()

        fig, ax = plt.subplots(figsize=SINGLE)
        for prior, part in inactive_rows.groupby("prior_name", sort=True):
            q = grouped_quantiles(part, ["radius"], "mass_ratio")
            color = COLORS.get(prior, None)
            ax.plot(q["radius"], q["median"], color=color, label=short_prior(prior))
            ax.fill_between(q["radius"], q["q25"], q["q75"], color=color, alpha=0.14, linewidth=0)
        ax.axhline(1.0, color="#333333", linewidth=0.55, linestyle=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("radius r")
        ax.set_ylabel("posterior / prior mass")
        ax.set_title("Inactive local mass ratio")
        ax.grid(True, which="both")
        ax.legend(loc="lower left", ncol=1)
        save_png(fig, figures, "exp1_inactive_mass_ratio_curves")

        fig, ax = plt.subplots(figsize=SINGLE)
        for prior, part in inactive_rows.groupby("prior_name", sort=True):
            q_prior = grouped_quantiles(part.assign(log_prior=log_safe(part["prior_mass"])), ["radius"], "log_prior")
            q_post = grouped_quantiles(part.assign(log_post=log_safe(part["post_mass"])), ["radius"], "log_post")
            color = COLORS.get(prior, None)
            ax.plot(q_prior["radius"], q_prior["median"], color=color, linestyle=":", label=f"{short_prior(prior)} prior")
            ax.plot(q_post["radius"], q_post["median"], color=color, linestyle="-", label=f"{short_prior(prior)} post.")
        ax.set_xscale("log")
        ax.set_xlabel("radius r")
        ax.set_ylabel("log local mass")
        ax.set_title("Inactive log-mass curves")
        ax.grid(True, which="both")
        ax.legend(loc="lower right", ncol=2, columnspacing=0.7, handlelength=1.4)
        save_png(fig, figures, "exp1_inactive_log_mass_curves")

    if not slopes.empty:
        slope_rows = slopes[~slopes["is_active"].astype(bool)].copy()
        fig, ax = plt.subplots(figsize=SINGLE)
        for prior, prior_part in slope_rows.groupby("prior_name", sort=True):
            color = COLORS.get(prior, None)
            for kind, style in [("prior", ":"), ("posterior", "-")]:
                part = prior_part[prior_part["mass_kind"] == kind]
                if part.empty:
                    continue
                q = grouped_quantiles(part, ["mid_radius"], "slope")
                ax.plot(q["mid_radius"], q["median"], color=color, linestyle=style)
        ax.axhline(1.0, color="#333333", linewidth=0.55, linestyle=":")
        ax.set_xscale("log")
        ax.set_ylim(0.82, 1.18)
        ax.set_xlabel("mid-radius")
        ax.set_ylabel("local slope")
        ax.set_title("Inactive local power slope")
        ax.grid(True, which="both")
        handles = [Line2D([0], [0], color=COLORS[p], label=short_prior(p)) for p in sorted(slope_rows["prior_name"].unique())]
        handles += [
            Line2D([0], [0], color="#333333", linestyle=":", label="prior"),
            Line2D([0], [0], color="#333333", linestyle="-", label="post."),
        ]
        ax.legend(handles=handles, loc="lower right", ncol=2, columnspacing=0.7, handlelength=1.4)
        save_png(fig, figures, "exp1_inactive_local_slope_curves")

    if not hessian.empty:
        order = sorted(hessian["prior_name"].unique())
        fig, ax = plt.subplots(figsize=SINGLE)
        for pos, prior in enumerate(order):
            values = hessian.loc[hessian["prior_name"] == prior, "hessian_condition_number"].to_numpy(float)
            x = np.full(len(values), pos) + np.linspace(-0.11, 0.11, len(values))
            color = COLORS.get(prior, "#555555")
            ax.scatter(x, values, color=color, s=13, alpha=0.75, linewidth=0)
            ax.hlines(np.median(values), pos - 0.22, pos + 0.22, color="#222222", linewidth=1.0)
        ax.set_yscale("log")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([short_prior(p) for p in order], rotation=18, ha="right")
        ax.set_ylabel("Hessian condition number")
        ax.set_title("Laplace Hessian stability")
        ax.grid(True, axis="y", which="both")
        save_png(fig, figures, "exp1_hessian_condition_numbers")

    if not inactive.empty and not active.empty:
        inactive = inactive.assign(coordinate_type="inactive")
        active = active.assign(coordinate_type="active")
        combined = pd.concat([inactive, active], ignore_index=True)
        fig, ax = plt.subplots(figsize=SINGLE_TALL)
        x = np.arange(len(sorted(combined["prior_name"].unique())))
        width = 0.34
        for offset, coord_type in [(-width / 2, "inactive"), (width / 2, "active")]:
            part = combined[combined["coordinate_type"] == coord_type].set_index("prior_name")
            values = [part.loc[p, "mean_log_mass_ratio"] for p in sorted(combined["prior_name"].unique())]
            ax.bar(
                x + offset,
                values,
                width=width,
                label=coord_type,
                color="#4C78A8" if coord_type == "inactive" else "#E45756",
                alpha=0.86,
            )
        ax.axhline(0.0, color="#333333", linewidth=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels([short_prior(p) for p in sorted(combined["prior_name"].unique())], rotation=18, ha="right")
        ax.set_ylabel("mean log mass ratio")
        ax.set_title("Active coordinates lose near-zero mass")
        ax.grid(True, axis="y")
        ax.legend(loc="lower left")
        save_png(fig, figures, "exp1_active_coordinate_diagnostics")


def plot_exp2() -> None:
    run_dir = latest_run(["exp2_actual_vi_training", "exp2_global_vs_local"])
    if run_dir is None:
        return
    figures = ensure_figures(run_dir)
    metrics = run_dir / "metrics"
    final = read_csv(metrics / "exp2_final_metrics.csv")
    local = read_csv(metrics / "exp2_local_mass_by_radius.csv")
    training = read_csv(metrics / "exp2_training_curves.csv")
    params = read_csv(metrics / "exp2_learned_parameters.csv")
    reliability = read_csv(metrics / "exp2_q_mass_reliability.csv")

    if final.empty and local.empty:
        raw = read_csv(run_dir / "raw_metrics.csv")
        summary = read_csv(run_dir / "summary_metrics.csv")
        training_log = read_csv(run_dir / "training_log.csv")
        if raw.empty:
            return
        local = raw.rename(
            columns={
                "dim": "dimension",
                "eps": "epsilon",
                "variational_family": "family",
                "r": "radius",
                "local_mass_ratio": "plot_local_mass_ratio",
            }
        )
        final = summary.rename(
            columns={
                "dim": "dimension",
                "eps": "epsilon",
                "variational_family": "family",
                "final_global_kl_estimate": "estimated_DKL_q_p",
                "min_local_mass_ratio": "local_mass_ratio_q_over_p",
            }
        )
        training = training_log.rename(
            columns={
                "dim": "dimension",
                "eps": "epsilon",
                "variational_family": "family",
                "training_step": "step",
                "training_loss": "loss",
            }
        )
        if "objective" not in local:
            local["objective"] = "kl_q_p"
        if "objective" not in final:
            final["objective"] = "kl_q_p"
        if "objective" not in training:
            training["objective"] = "kl_q_p"

    if not training.empty:
        fig, ax = plt.subplots(figsize=SINGLE)
        grouped = (
            training.groupby(["step", "family", "objective"], as_index=False)["loss"]
            .median()
            .sort_values(["family", "objective", "step"])
        )
        for (family, objective), part in grouped.groupby(["family", "objective"], sort=True):
            ax.plot(
                part["step"],
                part["loss"],
                color=COLORS.get(family),
                linestyle=LINESTYLES.get(objective, "-"),
                label=f"{short_family(family)}, {short_objective(objective)}",
            )
        ax.set_xlabel("training step")
        ax.set_ylabel("median loss")
        ax.set_title("VI optimisation traces")
        ax.grid(True)
        ax.legend(loc="best", ncol=1, handlelength=1.5)
        save_png(fig, figures, "exp2_training_curves")

    if not final.empty:
        summary = (
            final.groupby(["dimension", "family", "objective", "epsilon", "tau"], as_index=False)
            .agg(
                kl_qp=("estimated_DKL_q_p", "median"),
                local_ratio=("local_mass_ratio_q_over_p", "median"),
            )
            .sort_values(["dimension", "family", "objective"])
        )
        fig, ax = plt.subplots(figsize=SINGLE)
        for (family, objective), part in summary.groupby(["family", "objective"], sort=True):
            ax.scatter(
                part["kl_qp"],
                part["local_ratio"],
                s=16 + 2.2 * part["dimension"].to_numpy(float),
                color=COLORS.get(family),
                marker=MARKERS.get(objective, "o"),
                alpha=0.76,
                edgecolor="#222222",
                linewidth=0.25,
            )
        ax.axhline(1.0, color="#333333", linewidth=0.55, linestyle=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("estimated KL(q||p)")
        ax.set_ylabel("local mass ratio")
        ax.set_title("Global KL can hide local miss")
        ax.grid(True, which="both")
        handles = [
            Patch(color=COLORS["diagonal_gaussian"], label="Diag. Gaussian"),
            Patch(color=COLORS["diagonal_gaussian_mixture_2"], label="2-comp. mixture"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#888888", label="Excl. KL", markersize=4),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="#888888", label="Incl. KL", markersize=4),
        ]
        ax.legend(handles=handles, loc="lower right", ncol=2, columnspacing=0.7, handletextpad=0.4)
        save_png(fig, figures, "exp2_global_kl_vs_local_mass_ratio")

        dims = sorted(summary["dimension"].unique())
        fig, axes = plt.subplots(1, len(dims), figsize=DOUBLE_SHORT, sharey=True)
        axes = np.atleast_1d(axes)
        for ax, dim in zip(axes, dims):
            part_dim = summary[summary["dimension"] == dim]
            heat = (
                part_dim.groupby(["family", "objective"])["local_ratio"]
                .median()
                .unstack("objective")
                .reindex(["diagonal_gaussian", "diagonal_gaussian_mixture_2"])
            )
            values = np.log10(np.maximum(heat.to_numpy(float), 1.0e-12))
            image = ax.imshow(values, cmap="RdYlBu", vmin=-12, vmax=0.25, aspect="auto")
            ax.set_xticks(np.arange(len(heat.columns)))
            ax.set_xticklabels([short_objective(c) for c in heat.columns], rotation=20, ha="right")
            ax.set_yticks(np.arange(len(heat.index)))
            ax.set_yticklabels([short_family(i) for i in heat.index])
            ax.set_title(f"d={dim}")
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=6.2)
        axes[0].set_ylabel("family")
        cbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.01)
        cbar.set_label("log10 median local ratio")
        fig.suptitle("Objective direction vs local coverage", y=0.99, fontsize=7.6)
        save_png(fig, figures, "exp2_objective_direction_local_coverage")

    if not local.empty:
        dims = sorted(local["dimension"].unique())
        objectives = sorted(local["objective"].unique())
        fig, axes = plt.subplots(len(dims), len(objectives), figsize=DOUBLE_TALL, sharex=True, sharey=True)
        axes = np.atleast_2d(axes)
        for i, dim in enumerate(dims):
            for j, objective in enumerate(objectives):
                ax = axes[i, j]
                part = local[(local["dimension"] == dim) & (local["objective"] == objective)]
                for family, fam in part.groupby("family", sort=True):
                    q = grouped_quantiles(fam, ["radius"], "plot_local_mass_ratio")
                    color = COLORS.get(family)
                    ax.plot(q["radius"], q["median"], color=color, label=short_family(family))
                    ax.fill_between(q["radius"], q["q25"], q["q75"], color=color, alpha=0.14, linewidth=0)
                ax.axhline(1.0, color="#333333", linewidth=0.5, linestyle=":")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid(True, which="both")
                ax.set_title(f"d={dim}, {short_objective(objective)}")
                if i == len(dims) - 1:
                    ax.set_xlabel("radius r")
                if j == 0:
                    ax.set_ylabel("q(B_r) / p(B_r)")
        axes[0, -1].legend(loc="lower right")
        fig.suptitle("Local mass ratio by radius", y=0.995, fontsize=7.8)
        save_png(fig, figures, "exp2_local_mass_ratio_by_radius")

    if not params.empty and "learned_weight_component_2" in params:
        mix = params.dropna(subset=["learned_weight_component_2"]).copy()
        if not mix.empty:
            fig, ax = plt.subplots(figsize=SINGLE)
            for (objective, dim), part in mix.groupby(["objective", "dimension"], sort=True):
                q = (
                    part.groupby("epsilon", as_index=False)["learned_weight_component_2"]
                    .median()
                    .sort_values("epsilon")
                )
                ax.plot(
                    q["epsilon"],
                    q["learned_weight_component_2"],
                    color=COLORS.get(objective),
                    marker="o" if int(dim) == 2 else "s",
                    linestyle="-" if int(dim) == 2 else "--",
                    label=f"{short_objective(objective)}, d={dim}",
                )
            eps_min = float(mix["epsilon"].min())
            eps_max = float(mix["epsilon"].max())
            ax.plot([eps_min, eps_max], [eps_min, eps_max], color="#333333", linestyle=":", linewidth=0.7, label="target")
            ax.set_xlabel("target spike weight epsilon")
            ax.set_ylabel("learned mixture spike weight")
            ax.set_title("Mixture recovers spike weight")
            ax.grid(True)
            ax.legend(loc="upper left", ncol=1)
            save_png(fig, figures, "exp2_learned_mixture_weights")

    if not reliability.empty:
        fig, ax = plt.subplots(figsize=SINGLE)
        rel_order = ["reliable", "weak", "unreliable"]
        colors = {"reliable": "#4C78A8", "weak": "#F58518", "unreliable": "#E45756"}
        groups = reliability.groupby(["dimension", "family", "q_mass_reliability"]).size().unstack(fill_value=0)
        groups = groups.div(groups.sum(axis=1), axis=0).reindex(columns=rel_order, fill_value=0)
        labels = [f"d={idx[0]}\n{short_family(idx[1])}" for idx in groups.index]
        bottom = np.zeros(len(groups))
        x = np.arange(len(groups))
        for rel in rel_order:
            vals = groups[rel].to_numpy(float)
            ax.bar(x, vals, bottom=bottom, color=colors[rel], label=rel, width=0.62)
            bottom += vals
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel("share of radii / runs")
        ax.set_title("QMC local-mass reliability")
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(True, axis="y")
        ax.legend(loc="lower left", ncol=3, columnspacing=0.7)
        save_png(fig, figures, "exp2_zero_count_diagnostics")


def plot_exp3() -> None:
    run_dir = latest_run("exp3_directional_normalisation")
    if run_dir is None:
        return
    figures = ensure_figures(run_dir)
    raw = read_csv(run_dir / "raw_metrics.csv")
    summary = read_csv(run_dir / "summary_metrics.csv")
    if raw.empty or summary.empty:
        return

    median_alpha = summary["alpha"].median()
    heat = summary[summary["alpha"] == median_alpha].pivot(index="a_p", columns="a_q", values="mi_preserved_estimated")
    fig, ax = plt.subplots(figsize=SINGLE)
    image = ax.imshow(heat.astype(float).values, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels([f"{v:g}" for v in heat.columns])
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels([f"{v:g}" for v in heat.index])
    ax.set_xlabel("a_q")
    ax.set_ylabel("a_p")
    ax.set_title("Estimated MI preservation")
    for i, a_p in enumerate(heat.index):
        for j, a_q in enumerate(heat.columns):
            ax.text(j, i, "1" if bool(heat.loc[a_p, a_q]) else "0", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.052, pad=0.02, label="preserved")
    save_png(fig, figures, "mi_preservation_heatmap")

    aq_values = sorted(raw["a_q"].unique())
    aq_colors = {value: plt.cm.viridis(i / max(1, len(aq_values) - 1)) for i, value in enumerate(aq_values)}
    for alpha, alpha_part in raw.groupby("alpha", sort=True):
        ap_values = sorted(alpha_part["a_p"].unique())
        fig, axes = plt.subplots(1, len(ap_values), figsize=DOUBLE_SHORT, sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        for ax, a_p in zip(axes, ap_values):
            part_ap = alpha_part[alpha_part["a_p"] == a_p]
            for a_q, part in part_ap.groupby("a_q", sort=True):
                ax.plot(part["r"], part["local_mass_ratio"], color=aq_colors[a_q], label=f"{a_q:g}")
            ax.axhline(1.0, color="#333333", linestyle=":", linewidth=0.55)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"a_p={a_p:g}")
            ax.grid(True, which="both")
            ax.set_xlabel("r")
        axes[0].set_ylabel("q(B_r) / p(B_r)")
        handles = [Line2D([0], [0], color=aq_colors[v], label=f"a_q={v:g}") for v in aq_values]
        axes[-1].legend(handles=handles, loc="lower left", ncol=1, title=None, handlelength=1.3)
        fig.suptitle(f"Local mass ratio, alpha={alpha:g}", y=0.995, fontsize=7.8)
        save_png(fig, figures, f"local_mass_ratio_alpha{alpha}")

        for a_p in ap_values:
            part_ap = alpha_part[alpha_part["a_p"] == a_p]
            fig, axes = plt.subplots(1, 2, figsize=DOUBLE_SHORT, sharex=True, sharey=True)
            max_value = 1.0e-16
            for ax, column, title in [
                (axes[0], "normalised_local_rekl_q_p", "q||p / p(B_r)"),
                (axes[1], "normalised_local_rekl_p_q", "p||q / q(B_r)"),
            ]:
                for a_q, part in part_ap.groupby("a_q", sort=True):
                    values = np.maximum(part[column].to_numpy(float), 1.0e-16)
                    max_value = max(max_value, float(np.nanmax(values)))
                    ax.plot(part["r"], values, color=aq_colors[a_q], label=f"{a_q:g}")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("r")
                ax.set_title(title)
                ax.grid(True, which="both")
            axes[0].set_ylabel("normalised local RE-KL")
            axes[0].set_ylim(1.0e-16, max_value * 1.8)
            handles = [Line2D([0], [0], color=aq_colors[v], label=f"{v:g}") for v in aq_values]
            axes[1].legend(handles=handles, title="a_q", loc="lower left", ncol=1, handlelength=1.4)
            fig.suptitle(f"Normalised local RE-KL, alpha={alpha:g}, a_p={a_p:g}", y=0.995, fontsize=7.8)
            save_png(fig, figures, f"normalised_rekl_alpha{alpha}_ap{a_p}")

    mi_values = []
    for a in sorted(set(summary["a_p"]).union(set(summary["a_q"]))):
        candidates = []
        candidates.extend(summary.loc[summary["a_p"] == a, ["theoretical_MI_p", "estimated_MI_p"]].rename(
            columns={"theoretical_MI_p": "theoretical", "estimated_MI_p": "estimated"}
        ).to_dict("records"))
        candidates.extend(summary.loc[summary["a_q"] == a, ["theoretical_MI_q", "estimated_MI_q"]].rename(
            columns={"theoretical_MI_q": "theoretical", "estimated_MI_q": "estimated"}
        ).to_dict("records"))
        frame = pd.DataFrame(candidates)
        mi_values.append([f"{a:g}", f"{frame['theoretical'].median():.3f}", f"{frame['estimated'].median():.3f}"])

    fig, axes = plt.subplots(1, 4, figsize=DOUBLE, gridspec_kw={"width_ratios": [1.05, 1, 1, 1]})
    axes[0].axis("off")
    table = axes[0].table(
        cellText=mi_values,
        colLabels=["a", "MI th.", "MI est."],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.3)
    table.scale(1.0, 1.15)
    axes[0].set_title("MI values")
    for ax, alpha in zip(axes[1:], sorted(summary["alpha"].unique())):
        part = summary[summary["alpha"] == alpha]
        matrix = part.pivot(index="a_p", columns="a_q", values="mi_preserved_estimated")
        values = matrix.astype(float).to_numpy()
        ax.imshow(values, origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_xticklabels([f"{v:g}" for v in matrix.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_yticklabels([f"{v:g}" for v in matrix.index])
        ax.set_title(f"alpha={alpha:g}")
        ax.set_xlabel("a_q")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, "1" if values[i, j] else "0", ha="center", va="center", fontsize=5.8)
    axes[1].set_ylabel("")
    fig.suptitle("MI values and estimated preservation", y=0.995, fontsize=7.8)
    save_png(fig, figures, "mi_comparison_table")


def main() -> None:
    configure_style()
    plot_exp1()
    plot_exp2()
    plot_exp3()
    print("Rebuilt compact two-column figures under results/.")


if __name__ == "__main__":
    main()
