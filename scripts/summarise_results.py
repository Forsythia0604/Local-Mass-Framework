import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = [
    "exp1_bayesian_update",
    "exp2_global_vs_local",
    "exp3_directional_normalisation",
]


def latest_run(experiment_name):
    root = PROJECT_ROOT / "results" / experiment_name
    runs = sorted([path for path in root.glob("*") if path.is_dir()])
    return runs[-1] if runs else None


def read_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_yaml(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def link(path):
    return path.relative_to(PROJECT_ROOT).as_posix()


def exp1_statement(summary, failures):
    if summary.empty:
        return "No completed Exp1 rows are available."
    slope_gap = np.nanmedian(np.abs(summary["estimated_posterior_slope"] - summary["estimated_prior_slope"]))
    finite_ratio = np.isfinite(summary["min_ratio"]).mean()
    if finite_ratio < 1.0:
        return "Exp1 partially supports the intended diagnostic; some ratio estimates were non-finite and should be inspected."
    if slope_gap <= 0.3 and failures.empty:
        return "Exp1 results are consistent with local mass order preservation under the recorded Laplace posterior approximation."
    return "Exp1 partially supports the intended diagnostic; slope gaps or failures should be inspected before drawing stronger conclusions."


def exp2_statement(summary, failures):
    if summary.empty:
        return "No completed Exp2 rows are available."
    mf = summary[summary["variational_family"] == "mean_field_gaussian"]
    if not mf.empty and np.nanmin(mf["min_local_mass_ratio"]) < 0.5:
        return "Exp2 results are consistent with global discrepancy missing local under-coverage in at least one mean-field run."
    if failures.empty:
        return "Exp2 does not show strong local under-coverage by the simple threshold used in this summary; inspect raw curves."
    return "Exp2 partially supports the intended diagnostic, but failed runs must be inspected."


def exp3_statement(summary, failures):
    if summary.empty:
        return "No completed Exp3 rows are available."
    match_rate = float(np.mean(summary["metrics_match_theoretical_classification"].astype(bool)))
    if match_rate == 1.0 and failures.empty:
        return "Exp3 results are consistent with the one-sided MI preservation classification over the configured grid."
    return f"Exp3 partially supports the intended diagnostic; classification match rate is {match_rate:.3f} and warnings/failures should be inspected."


def compact_table(summary, exp_name):
    if summary.empty:
        return "_No summary rows._"
    if exp_name == "exp1_bayesian_update":
        cols = ["dimension", "n_samples", "prior_name", "coordinate_type", "estimated_prior_slope", "estimated_posterior_slope", "min_ratio", "max_ratio"]
    elif exp_name == "exp2_global_vs_local":
        cols = ["dim", "eps", "tau", "variational_family", "final_global_kl_estimate", "min_local_mass_ratio", "max_normalised_local_rekl_q_p"]
    else:
        cols = ["a_p", "a_q", "alpha", "theoretical_MI_p", "theoretical_MI_q", "estimated_MI_p", "estimated_MI_q", "mi_preserved_theoretical", "mi_preserved_estimated"]
    cols = [col for col in cols if col in summary.columns]
    return summary[cols].head(20).round(6).to_markdown(index=False)


def main():
    report = []
    report.append("# Summary Report")
    report.append("")
    report.append(f"Generated at: {datetime.now().isoformat()}")
    report.append("")

    statements = {}
    for exp_name in EXPERIMENTS:
        run_dir = latest_run(exp_name)
        report.append(f"## {exp_name}")
        if run_dir is None:
            report.append("No run directory found.")
            report.append("")
            continue

        metadata = read_json(run_dir / "metadata.json")
        config = read_yaml(run_dir / "config.yaml")
        summary_path = run_dir / "summary_metrics.csv"
        failures_path = run_dir / "logs" / "failures.csv"
        summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        failures = pd.read_csv(failures_path) if failures_path.exists() else pd.DataFrame()

        report.append(f"Run directory: `{link(run_dir)}`")
        report.append(f"Raw data: `{link(run_dir / 'raw_metrics.csv')}`")
        report.append(f"Summary data: `{link(summary_path)}`")
        report.append(f"Figures: `{link(run_dir / 'figures')}`")
        report.append(f"Failures: `{link(failures_path)}`")
        report.append("")
        report.append("Environment:")
        versions = metadata.get("package_versions", {})
        for key in ["python", "platform", "numpy", "scipy", "pandas", "matplotlib", "torch", "cuda_available", "cuda_device_name"]:
            if key in versions:
                report.append(f"- {key}: `{versions[key]}`")
        report.append("")
        report.append("Configuration:")
        report.append("```yaml")
        report.append(yaml.safe_dump(config, sort_keys=False).strip())
        report.append("```")
        report.append("")
        report.append("Main metrics:")
        report.append(compact_table(summary, exp_name))
        report.append("")

        failure_count = 0 if failures.empty else len(failures)
        warning_count = int(summary["warning_count"].sum()) if "warning_count" in summary.columns and not summary.empty else 0
        report.append(f"Unexpected or failed runs: {failure_count} failures, {warning_count} integration warnings recorded.")
        report.append("")

        if exp_name == "exp1_bayesian_update":
            statement = exp1_statement(summary, failures)
        elif exp_name == "exp2_global_vs_local":
            statement = exp2_statement(summary, failures)
        else:
            statement = exp3_statement(summary, failures)
        statements[exp_name] = statement
        report.append(f"Diagnostic statement: {statement}")
        report.append("")

    report.append("## Overall")
    for exp_name, statement in statements.items():
        report.append(f"- {exp_name}: {statement}")
    report.append("")
    report.append("These statements are diagnostic summaries only. They should be checked against the raw CSV files and unsmoothed figures.")

    output = PROJECT_ROOT / "results" / "summary_report.md"
    output.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
