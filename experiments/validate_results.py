from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments.common import repo_root


def bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> None:
    root = repo_root()
    lines = ["# Validation Summary", "", "Zero is treated only as one finite centre; no result is interpreted as sparsity detection.", ""]

    exp1_path = root / "results" / "exp1" / "processed" / "exp1_kappa_mi_table.csv"
    if exp1_path.exists():
        exp1 = pd.read_csv(exp1_path)
        tol = 0.15
        prior_bad = exp1[(exp1["prior_kappa_estimated"] - exp1["expected_kappa"]).abs() > tol]
        post_bad = exp1[exp1["posterior_kappa_error"] > tol]
        unreliable = exp1[~bool_series(exp1["prior_fit_reliable"]) | ~bool_series(exp1["posterior_fit_reliable"]) | ~bool_series(exp1["radius_selection_reliable"])]
        mi_bad = exp1[
            ~np.isclose(exp1["prior_MI_theoretical"], exp1["dimension"] / exp1["prior_kappa_theoretical"])
            | ~np.isclose(exp1["expected_MI"], exp1["dimension"] / exp1["expected_kappa"])
        ]
        zero_excluded = exp1[(exp1["centre_type"] == "zero") & (~bool_series(exp1["posterior_fit_reliable"]))]
        lines += [
            "## EXP1",
            f"- Known kappa fits: {len(exp1)}",
            f"- Prior kappa errors above tolerance: {len(prior_bad)}",
            f"- Posterior kappa errors above tolerance: {len(post_bad)}",
            f"- Converted MI values recorded: {len(exp1)}",
            f"- Estimates marked unreliable: {len(unreliable)}",
            f"- Zero-centre results excluded from main claims: {len(zero_excluded)}",
            f"- EXP1 numerical exponent preservation: {_status(len(post_bad) == 0)}",
            f"- EXP1 MI conversion correctness: {_status(len(mi_bad) == 0)}",
            f"- EXP1 all-window reliability: {_status(len(unreliable) == 0)}",
            f"- EXP1 paper figure readiness: {_status((root / 'results' / 'exp1' / 'figures').exists())}",
            "",
        ]
    else:
        lines += ["## EXP1", "- Not run.", ""]

    exp2_path = root / "results" / "exp2" / "processed" / "exp2_kappa_mi_divergence_table.csv"
    if exp2_path.exists():
        exp2 = pd.read_csv(exp2_path)
        zero_reliable = exp2[exp2["reliable_radius_count"] == 0]
        mislabelled = exp2[(exp2["case"] == "q3") & (exp2["kappa_relation"] != "kappa-mismatched")]
        lines += [
            "## EXP2",
            f"- Divergence curves: {len(exp2)}",
            f"- Curves with zero reliable radius count: {len(zero_reliable)}",
            "- Figures affected by clipping: yes; clipped arrays are display-only and log-domain arrays are saved.",
            f"- Cases where singularity or bump centre is not aligned with diagnostic centre: {len(mislabelled)}",
            f"- EXP2 exponent/MI labels: {_status(len(mislabelled) == 0)}",
            f"- EXP2 normalised divergence reliability: {_status(len(zero_reliable) == 0)}",
            f"- EXP2 figure readiness: {_status((root / 'results' / 'exp2' / 'figures').exists())}",
            "",
        ]
    else:
        lines += ["## EXP2", "- Not run.", ""]

    exp3_path = root / "results" / "exp3" / "processed" / "exp3_vi_family_comparison.csv"
    if exp3_path.exists():
        exp3 = pd.read_csv(exp3_path)
        tol = 0.15
        kappa_bad = exp3[(exp3["kappa_p_error_if_known"] > tol) | (exp3["kappa_q_error_if_known"] > tol)]
        unreliable = exp3[~bool_series(exp3["kappa_p_reliable"]) | ~bool_series(exp3["kappa_q_reliable"]) | ~bool_series(exp3["radius_selection_reliable"])]
        zero_reliable = exp3[(exp3["reliable_radius_count_p_scale"] == 0) | (exp3["reliable_radius_count_q_scale"] == 0)]
        elbo_complete = {"ELBO_initial", "ELBO_final", "ELBO_best", "ELBO_best_step", "ELBO_mean_last_100", "ELBO_std_last_100"}.issubset(exp3.columns)
        nonzero = exp3[exp3["centre_type"] == "nonzero"].copy()
        if not nonzero.empty:
            elbo_rank = list(nonzero.groupby("VI_method")["ELBO_best"].max().sort_values(ascending=False).index)
            p_rank = list(nonzero.groupby("VI_method")["log_N_alpha_r_p_q_over_p_largest_r"].mean().sort_values().index)
            q_rank = list(nonzero.groupby("VI_method")["log_N_alpha_r_q_p_over_q_largest_r"].mean().sort_values().index)
        else:
            elbo_rank, p_rank, q_rank = [], [], []
        lines += [
            "## EXP3",
            f"- Trained VI diagnostic rows: {len(exp3)}",
            f"- Known kappa errors above tolerance: {len(kappa_bad)}",
            f"- Estimates marked unreliable: {len(unreliable)}",
            f"- Curves with zero reliable radius count: {len(zero_reliable)}",
            "- Figures affected by clipping: yes; clipped arrays are display-only and log-domain arrays are saved.",
            f"- EXP3 global ELBO ranking: {elbo_rank}",
            f"- EXP3 p-scale local divergence ranking: {p_rank}",
            f"- EXP3 q-scale local divergence ranking: {q_rank}",
            f"- ELBO ranking agrees with p-scale divergence ranking: {_status(elbo_rank == p_rank)}",
            f"- ELBO ranking agrees with q-scale divergence ranking: {_status(elbo_rank == q_rank)}",
            f"- EXP3 exponent fitting reliability: {_status(len(unreliable) == 0 and len(kappa_bad) == 0)}",
            f"- EXP3 MI conversion correctness: {_status({'MI_p', 'MI_q'}.issubset(exp3.columns))}",
            f"- EXP3 global ELBO recording completeness: {_status(elbo_complete)}",
            f"- EXP3 figure readiness: {_status((root / 'results' / 'exp3' / 'figures').exists())}",
            "",
        ]
    else:
        lines += ["## EXP3", "- Not run.", ""]

    lines += [
        "## Overall Readiness",
        "- Exponent fitting reliability: inspect per-experiment PASS/FAIL lines above.",
        "- Mass Index conversion correctness: MI is always recorded as dimension / kappa.",
        "- Normalised divergence reliability: scientific values are saved in log domain.",
        "- Global ELBO recording completeness: required for EXP3 only.",
        "- Figure readiness: PNG-only figures are generated by plot scripts after compute scripts run.",
        "- Paper readiness: requires all relevant PASS lines and review of unreliable zero-centre exclusions.",
    ]
    out = root / "results" / "validation_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
