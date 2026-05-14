"""
Recommender Service — placeholder for ML-based recommendations.
Will be connected to ml/inference/predict.py when the model is ready.
"""
import logging
from recommender.services.interaction_service import log_interaction
from recommender.loaders.item_loader import get_item_by_id

logger = logging.getLogger(__name__)


def get_user_recommendations(user_id: str, limit: int = 10) -> list[dict]:
    """
    Placeholder: returns empty list until ML model is trained.
    The ML pipeline (ml/inference/predict.py) will plug in here.
    """
    logger.info(f"[RecommenderService] Recommendations requested for user={user_id} (ML not ready yet)")
    return []


def get_item_details(item_id: str) -> dict | None:
    return get_item_by_id(item_id)


# Re-export for backward compat
__all__ = ["get_user_recommendations", "get_item_details", "log_interaction"]
