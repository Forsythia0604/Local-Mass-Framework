r"""Pre-flight: all five thesis-anchored self-consistency gates.

Must be green before any training job runs (plan, "Verification" section):
  1-4  mass_index.self_consistency_tests   (Gaussian / spike-slab / radial / horseshoe)
  5    budgets.integrated_budget_recovers_kl (TKL alpha=1 recovers KL; closed-form budget)

Run:  python preflight.py
Exit code 0 iff every gate passes.
"""

import sys

from mass_index import self_consistency_tests
from budgets import integrated_budget_recovers_kl


def main():
    print("=" * 70)
    print("Mass-Index estimator gates (1-4)")
    print("=" * 70)
    mi = self_consistency_tests(verbose=True)

    print("\n" + "=" * 70)
    print("Budget <-> KL recovery gate (5)")
    print("=" * 70)
    bg = integrated_budget_recovers_kl(verbose=True)

    all_pass = mi["all_pass"] and bg["pass"]
    print("\n" + "=" * 70)
    print(f"PRE-FLIGHT {'PASSED' if all_pass else 'FAILED'}")
    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
