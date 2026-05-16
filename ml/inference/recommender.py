# ml/inference/recommender.py
"""
Recommender — Django-facing Singleton
──────────────────────────────────────
This is the single entry point that Django services call.
It lazy-loads the HybridPredictor once and reuses it across requests.

Usage (from recommender_service.py):
    from ml.inference.recommender import get_recommendations, get_similar_items
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Module-level singleton (loaded once per process) ─────────────────────────
_predictor = None
_load_error: Optional[str] = None


def _get_predictor():
    """Lazy-load the HybridPredictor. Thread-safe at module import level."""
    global _predictor, _load_error

    if _predictor is not None:
        return _predictor

    if _load_error is not None:
        return None   # already tried and failed

    try:
        from ml.inference.predict import HybridPredictor
        _predictor = HybridPredictor()
        logger.info("[Recommender] HybridPredictor loaded successfully.")
    except Exception as e:
        _load_error = str(e)
        logger.error("[Recommender] Failed to load HybridPredictor: %s", e)

    return _predictor


# ── Public API ────────────────────────────────────────────────────────────────

def get_recommendations(
    user_id:          str,
    user_item_ids:    list = None,
    k:                int  = 10,
    exclude_item_ids: list = None,
) -> list:
    """
    Return up to `k` recommended item_ids for a user.

    Parameters
    ----------
    user_id          : Django User.id (as string)
    user_item_ids    : items the user has interacted with (for content-based arm)
    k                : number of recommendations to return
    exclude_item_ids : items to suppress (e.g. already rated, in cart)

    Returns
    -------
    list of item_id strings (may be empty if models are not yet trained)
    """
    predictor = _get_predictor()
    if predictor is None:
        logger.warning("[Recommender] No predictor available — returning empty list.")
        return []

    try:
        return predictor.recommend(
            user_id=user_id,
            user_item_ids=user_item_ids or [],
            k=k,
            exclude_item_ids=exclude_item_ids or [],
        )
    except Exception as e:
        logger.error("[Recommender] get_recommendations error: %s", e)
        return []


def get_similar_items(item_id: str, k: int = 6) -> list:
    """
    Return up to `k` content-similar item_ids for a given item.
    Useful for product-detail page 'You may also like' sections.
    Falls back to [] if the CB model isn't trained yet.
    """
    predictor = _get_predictor()
    if predictor is None:
        return []

    try:
        return predictor.get_similar_items(item_id, k=k)
    except Exception as e:
        logger.error("[Recommender] get_similar_items error: %s", e)
        return []


def reload_predictor():
    """
    Force a reload of the predictor (call after retraining).
    Useful in management commands or Celery tasks after retrain_pipeline runs.
    """
    global _predictor, _load_error
    _predictor  = None
    _load_error = None
    logger.info("[Recommender] Predictor cache cleared — will reload on next request.")
    return _get_predictor() is not None
