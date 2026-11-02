"""CLI Runner for Research Experiments and Locked Evaluation.

Usage:
    python scripts/run_experiments.py --exp E5      # SHAP Stability across Retraining
    python scripts/run_experiments.py --exp E7      # Applicant Subgroup Performance Audit
    python scripts/run_experiments.py --exp EVAL    # Frozen Test Set Batch Evaluation
    python scripts/run_experiments.py --exp all     # Run E5, E7, and EVAL sequentially
"""

import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.log import get_logger

logger = get_logger("RunExperiments", "run_experiments.log")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 11 Research Experiments & Frozen Evaluation")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        choices=["E5", "E7", "EVAL", "all"],
        help="Experiment to run: E5 (Stability), E7 (Subgroup), EVAL (Frozen Test Batch), or all",
    )
    parser.add_argument(
        "--n_retrains",
        type=int,
        default=15,
        help="Number of bootstrap retrains per strategy for E5 (default: 15)",
    )
    parser.add_argument(
        "--panel_size",
        type=int,
        default=5000,
        help="Fixed panel size for E5 explanation comparison (default: 5000)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.16,
        help="Operating decision threshold for evaluations (default: 0.16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()
    logger.info(f"Invoked run_experiments.py with arguments: {args}")

    t_start = time.time()

    if args.exp in ["E5", "all"]:
        logger.info("Executing Experiment E5: SHAP Stability across Retraining...")
        from src.explainability.stability import run_e5_experiment
        run_e5_experiment(n_retrains=args.n_retrains, panel_size=args.panel_size, base_seed=args.seed)

    if args.exp in ["E7", "all"]:
        logger.info("Executing Experiment E7: Applicant Subgroup Performance Audit...")
        from src.evaluation.subgroup import run_e7_subgroup_audit
        run_e7_subgroup_audit(threshold=args.threshold, random_state=args.seed)

    if args.exp in ["EVAL", "all"]:
        logger.info("Executing Frozen Test Set Batch Evaluation...")
        from src.evaluation.compare import run_frozen_test_evaluation
        run_frozen_test_evaluation()

    total_elapsed = time.time() - t_start
    logger.info(f"All requested experiments completed successfully in {total_elapsed:.1f}s ({total_elapsed/60:.2f} min).")


if __name__ == "__main__":
    main()
