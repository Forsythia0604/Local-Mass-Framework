import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def import_script(name):
    path = SCRIPTS_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def latest_run(experiment_name):
    root = PROJECT_ROOT / "results" / experiment_name
    runs = sorted([path for path in root.glob("*") if path.is_dir()])
    return runs[-1] if runs else None


def regenerate(exp_name, script_name):
    run_dir = latest_run(exp_name)
    if run_dir is None:
        print(f"No saved runs found for {exp_name}")
        return
    raw_path = run_dir / "raw_metrics.csv"
    summary_path = run_dir / "summary_metrics.csv"
    if not raw_path.exists():
        print(f"Missing raw_metrics.csv for {exp_name}: {raw_path}")
        return
    raw = pd.read_csv(raw_path)
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    module = import_script(script_name)
    module.make_figures(raw, summary, run_dir)
    print(f"Regenerated figures for {exp_name} in {run_dir / 'figures'}")


def main():
    regenerate("exp1_bayesian_update", "run_exp1_bayesian_update.py")
    regenerate("exp2_global_vs_local", "run_exp2_global_vs_local.py")
    regenerate("exp3_directional_normalisation", "run_exp3_directional_normalisation.py")


if __name__ == "__main__":
    main()
