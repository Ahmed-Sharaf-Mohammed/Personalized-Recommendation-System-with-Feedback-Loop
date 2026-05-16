"""
Management command — run the full preprocessing pipeline.

Usage:
    python manage.py preprocess
    python manage.py preprocess --step encoders
    python manage.py preprocess --step explicit
    python manage.py preprocess --step implicit
    python manage.py preprocess --step all        (default)
"""

import time
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Run the ML preprocessing pipeline (encoders → explicit matrix → implicit matrix)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--step",
            choices=["all", "encoders", "explicit", "implicit"],
            default="all",
            help="Which step to run (default: all)",
        )
        parser.add_argument(
            "--no-time-decay",
            action="store_true",
            default=False,
            help="Disable time-decay weighting when building the implicit matrix",
        )

    def handle(self, *args, **options):
        step       = options["step"]
        time_decay = not options["no_time_decay"]

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"\n🚀  Preprocessing Pipeline  [step={step}]"
        ))
        t0 = time.time()

        try:
            if step in ("all", "encoders"):
                self._run_encoders()
            if step in ("all", "explicit"):
                self._run_explicit_transform()
                self._run_explicit_matrix()
            if step in ("all", "implicit"):
                self._run_implicit_matrix(time_decay)

        except Exception as exc:
            raise CommandError(f"Preprocessing failed: {exc}") from exc

        self.stdout.write(self.style.SUCCESS(
            f"\n✅  Preprocessing complete  ({time.time()-t0:.1f}s)\n"
        ))

    def _run_encoders(self):
        self.stdout.write("  Step 1 — Building global user / item encoders …")
        from ml.preprocessing.global_encoders import build_global_encoders
        user_enc, item_enc = build_global_encoders()
        self.stdout.write(self.style.SUCCESS(
            f"          users={len(user_enc.classes_):,}  items={len(item_enc.classes_):,}"
        ))

    def _run_explicit_transform(self):
        self.stdout.write("  Step 2 — Transforming explicit interactions …")
        from ml.preprocessing.explicit_transform import main as explicit_transform
        explicit_transform()
        self.stdout.write(self.style.SUCCESS("          done"))

    def _run_explicit_matrix(self):
        self.stdout.write("  Step 3 — Building explicit sparse matrix …")
        from ml.preprocessing.explicit_interaction_matrix import main as build_explicit
        build_explicit()
        self.stdout.write(self.style.SUCCESS("          done"))

    def _run_implicit_matrix(self, time_decay: bool):
        label = "time-decay" if time_decay else "uniform"
        self.stdout.write(f"  Step 4 — Building implicit matrix ({label}) …")
        from ml.preprocessing.implicit_interaction_matrix import main as build_implicit
        build_implicit(use_time_decay=time_decay)
        self.stdout.write(self.style.SUCCESS("          done"))
