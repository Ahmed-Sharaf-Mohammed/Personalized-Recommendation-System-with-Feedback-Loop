"""
Interaction Service — save/query user ratings & reviews from SQLite.
Schema-compatible with interactions.parquet for ML retraining.
"""
import logging
from django.utils import timezone
from recommender.models import UserInteraction

# Writes to logs/interactions.log via LOGGING config in settings.py
_ilog = logging.getLogger("recommender.interactions")


def log_interaction(user_id, item_id, rating=None, review_text=None,
                    review_title=None, verified=False, helpful_votes=0):
    obj = UserInteraction.objects.create(
        user_id=str(user_id),
        item_id=str(item_id),
        rating=rating,
        review_text=review_text or None,
        review_title=review_title or None,
        verified=verified,
        helpful_votes=helpful_votes,
    )

    # ── Write to interactions.log ──────────────────────────────────────────
    _ilog.info(
        "event=%-20s user=%-10s item=%-15s rating=%s verified=%s",
        "explicit_rating", user_id, item_id,
        f"{rating:.1f}" if rating is not None else "None",
        verified,
    )

    return obj


def get_item_reviews(item_id, limit=20):
    """Return reviews for a product page — ordered newest first."""
    return list(
        UserInteraction.objects
        .filter(item_id=str(item_id), rating__isnull=False)
        .order_by("-timestamp")[:limit]
    )


def get_user_interactions(user_id):
    return list(
        UserInteraction.objects
        .filter(user_id=str(user_id))
        .order_by("-timestamp")
        .values("item_id", "rating", "review_title", "timestamp")
    )


def get_user_rated_item_ids(user_id):
    return set(
        UserInteraction.objects
        .filter(user_id=str(user_id))
        .values_list("item_id", flat=True)
    )
