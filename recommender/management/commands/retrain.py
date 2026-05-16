"""
Management command — retrain all recommendation models incrementally.

Usage:
    python manage.py retrain
    python manage.py retrain --retune          # grid-search before training
    python manage.py retrain --dry-run         # show what would run, don't execute
    python manage.py retrain --skip-preprocess # skip rebuild of matrices
"""

import time
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = (
        "Retrain all recommendation models (SVD, ALS, Content-Based). "
        "Rebuilds interaction matrices, retrains with best params, evaluates, "
        "hot-reloads the inference predictor, and logs the run to MLflow."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--retune",
            action="store_true",
            default=False,
            help="Run hyperparameter grid-search before retraining",
        )
        parser.add_argument(
            "--skip-preprocess",
            action="store_true",
            default=False,
            help="Skip matrix rebuild steps (use existing artifacts)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Print the steps that would run without executing them",
        )

    def handle(self, *args, **options):
        retune          = options["retune"]
        skip_preprocess = options["skip_preprocess"]
        dry_run         = options["dry_run"]

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"\n🔄  Retrain Pipeline"
            f"  [retune={retune}  skip_preprocess={skip_preprocess}  dry_run={dry_run}]"
        ))

        if dry_run:
            self._print_plan(retune, skip_preprocess)
            return

        t0 = time.time()
        try:
            report = self._run(retune, skip_preprocess)
        except Exception as exc:
            raise CommandError(f"Retrain failed: {exc}") from exc

        self._print_report(report)
        self.stdout.write(self.style.SUCCESS(
            f"\n✅  Retrain complete  ({time.time()-t0:.1f}s)\n"
        ))

    # ── Execution ─────────────────────────────────────────────────────────────

    def _run(self, retune: bool, skip_preprocess: bool) -> dict:
        import os, json

        # ── Optional: hyperparameter tuning ──────────────────────────────────
        if retune:
            self.stdout.write("  Step 0 — Hyperparameter tuning (grid search) …")
            from ml.training.tuning import main as run_tuning
            best = run_tuning(quick=False)
            self.stdout.write(self.style.SUCCESS(f"          best={best}"))

        # ── Preprocessing (rebuild matrices from DB) ──────────────────────────
        if not skip_preprocess:
            self.stdout.write("  Step 1 — Updating global encoders …")
            from ml.preprocessing.global_encoders import build_global_encoders
            user_enc, item_enc = build_global_encoders()
            self.stdout.write(self.style.SUCCESS(
                f"          users={len(user_enc.classes_):,}  items={len(item_enc.classes_):,}"
            ))

            self.stdout.write("  Step 2 — Re-transforming explicit interactions …")
            from ml.preprocessing.explicit_transform import main as explicit_transform
            explicit_transform()
            self.stdout.write(self.style.SUCCESS("          done"))

            self.stdout.write("  Step 3 — Rebuilding explicit matrix …")
            from ml.preprocessing.explicit_interaction_matrix import main as build_explicit
            build_explicit()
            self.stdout.write(self.style.SUCCESS("          done"))

            self.stdout.write("  Step 4 — Rebuilding implicit matrix (time-decay) …")
            from ml.preprocessing.implicit_interaction_matrix import main as build_implicit
            build_implicit(use_time_decay=True)
            self.stdout.write(self.style.SUCCESS("          done"))
        else:
            self.stdout.write(self.style.WARNING(
                "  Steps 1-4 skipped (--skip-preprocess)"
            ))

        # ── Load best params ──────────────────────────────────────────────────
        reports_dir  = "data/reports"
        params_path  = f"{reports_dir}/best_params.json"
        best_params  = {}
        if os.path.exists(params_path):
            with open(params_path) as f:
                best_params = json.load(f)
            self.stdout.write(f"  📂 Using tuned params from {params_path}")
        else:
            self.stdout.write("  ℹ️  No tuned params found — using defaults")

        svd_p = best_params.get("svd", {})
        als_p = best_params.get("als", {})

        # ── Train ─────────────────────────────────────────────────────────────
        self.stdout.write("  Step 5 — Training models …")
        from ml.training.train import main as train_models
        train_models(
            n_svd_factors      = svd_p.get("n_factors",      50),
            als_factors        = als_p.get("factors",        64),
            als_iterations     = als_p.get("iterations",     20),
            als_regularization = als_p.get("regularization", 0.01),
        )
        self.stdout.write(self.style.SUCCESS("          done"))

        # ── Evaluate ──────────────────────────────────────────────────────────
        self.stdout.write("  Step 6 — Evaluating models …")
        old_report = self._load_json(f"{reports_dir}/evaluation_report.json")
        from ml.training.evaluate import run_evaluation
        new_report = run_evaluation()
        self._compare(old_report, new_report)
        self.stdout.write(self.style.SUCCESS("          done"))

        # ── MLflow logging ────────────────────────────────────────────────────
        self.stdout.write("  Step 7 — Logging run to MLflow …")
        self._log_mlflow(best_params, new_report)
        self.stdout.write(self.style.SUCCESS("          done"))

        # ── Hot-reload ────────────────────────────────────────────────────────
        self.stdout.write("  Step 8 — Hot-reloading inference predictor …")
        try:
            from ml.inference.recommender import reload_predictor
            ok = reload_predictor()
            if ok:
                self.stdout.write(self.style.SUCCESS("          predictor reloaded ✅"))
            else:
                self.stdout.write(self.style.WARNING("          reload failed (check logs)"))
        except Exception as exc:
            self.stdout.write(self.style.WARNING(f"          could not reload: {exc}"))

        # ── Save retrain meta ─────────────────────────────────────────────────
        import json
        from datetime import datetime
        meta = {
            "retrained_at":  datetime.now().isoformat(timespec="seconds"),
            "params_used":   best_params,
            "skip_preprocess": skip_preprocess,
            "retune":          retune,
        }
        os.makedirs(reports_dir, exist_ok=True)
        with open(f"{reports_dir}/retrain_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        return new_report

    # ── MLflow ────────────────────────────────────────────────────────────────

    def _log_mlflow(self, params: dict, report: dict):
        try:
            import mlflow

            mlflow.set_tracking_uri("mlruns")
            mlflow.set_experiment("recommendation-system")

            with mlflow.start_run(run_name="retrain"):
                # Params
                svd_p = params.get("svd", {})
                als_p = params.get("als", {})
                mlflow.log_params({
                    "svd_n_factors":       svd_p.get("n_factors",      50),
                    "als_factors":         als_p.get("factors",        64),
                    "als_iterations":      als_p.get("iterations",     20),
                    "als_regularization":  als_p.get("regularization", 0.01),
                })

                # Metrics for each model × each K
                for model_name, model_data in report.items():
                    if not isinstance(model_data, dict):
                        continue
                    for k_label, metrics in model_data.items():
                        if not isinstance(metrics, dict):
                            continue
                        k = k_label.replace("@", "")
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                key = f"{model_name}_{metric_name}_at_{k}"
                                mlflow.log_metric(key, round(float(value), 6))

                # Save report as artifact
                import json, tempfile, os
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as tmp:
                    json.dump(report, tmp, indent=2)
                    tmp_path = tmp.name
                mlflow.log_artifact(tmp_path, artifact_path="evaluation")
                os.unlink(tmp_path)

            self.stdout.write(self.style.SUCCESS(
                "          MLflow run logged → mlruns/"
            ))

        except Exception as exc:
            self.stdout.write(self.style.WARNING(
                f"          MLflow logging skipped: {exc}"
            ))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_json(self, path: str) -> dict:
        import json, os
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def _compare(self, old: dict, new: dict):
        self.stdout.write("\n          📊 NDCG@10 comparison")
        self.stdout.write(f"          {'Model':20s} {'Before':>8} {'After':>8} {'Δ':>8}")
        self.stdout.write("          " + "─" * 44)
        for model in ("svd", "als", "content_based"):
            old_v = old.get(model, {}).get("@10", {}).get("NDCG")
            new_v = new.get(model, {}).get("@10", {}).get("NDCG")
            old_s = f"{old_v:.4f}" if old_v is not None else "   —"
            new_s = f"{new_v:.4f}" if new_v is not None else "   —"
            if old_v is not None and new_v is not None:
                delta = new_v - old_v
                sign  = "+" if delta >= 0 else ""
                arrow = "📈" if delta > 0.001 else ("📉" if delta < -0.001 else "≈ ")
                d_s   = f"{sign}{delta:.4f}"
            else:
                d_s, arrow = "   —", "  "
            self.stdout.write(
                f"          {model:20s} {old_s:>8} {new_s:>8}  {arrow} {d_s}"
            )
        self.stdout.write("")

    def _print_plan(self, retune: bool, skip_preprocess: bool):
        steps = []
        if retune:
            steps.append("Step 0 — Hyperparameter tuning (grid search)")
        if not skip_preprocess:
            steps += [
                "Step 1 — Update global encoders",
                "Step 2 — Re-transform explicit interactions",
                "Step 3 — Rebuild explicit sparse matrix",
                "Step 4 — Rebuild implicit matrix (time-decay)",
            ]
        steps += [
            "Step 5 — Train SVD, ALS, Content-Based models",
            "Step 6 — Evaluate models (Precision/Recall/MAP/NDCG @5,10,20)",
            "Step 7 — Log run to MLflow (mlruns/)",
            "Step 8 — Hot-reload inference predictor",
        ]
        self.stdout.write("\n  Dry run — would execute:\n")
        for s in steps:
            self.stdout.write(f"    • {s}")
        self.stdout.write("")

    def _print_report(self, report: dict):
        self.stdout.write("\n  📋 Final Evaluation Report")
        self.stdout.write("  " + "─" * 60)
        self.stdout.write(f"  {'Model':20s} {'K':>4}  {'P@K':>7} {'R@K':>7} {'MAP':>7} {'NDCG':>7}")
        self.stdout.write("  " + "─" * 60)
        for model, data in report.items():
            if not isinstance(data, dict):
                continue
            for k_label, m in data.items():
                if not isinstance(m, dict):
                    continue
                p  = m.get("precision", 0)
                r  = m.get("recall",    0)
                ap = m.get("MAP",       0)
                nd = m.get("NDCG",      0)
                self.stdout.write(
                    f"  {model:20s} {k_label:>4}  "
                    f"{p:7.4f} {r:7.4f} {ap:7.4f} {nd:7.4f}"
                )
        self.stdout.write("  " + "─" * 60)
