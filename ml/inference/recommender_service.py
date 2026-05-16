from ml.inference.recommender import get_recommendations
from ml.inference.loader import get_items_by_ids

from recommender.models import (
    UserInteraction,
    UserBrowsingLog
)


def recommend_for_user(user_id, k=10):
    """
    Main recommendation service for Django API.
    """

    # ------------------------------------------------
    # Collect user history
    # ------------------------------------------------

    explicit_items = list(
        UserInteraction.objects.filter(user_id=user_id)
        .values_list("item_id", flat=True)
    )

    implicit_items = list(
        UserBrowsingLog.objects.filter(user_id=user_id)
        .values_list("item_id", flat=True)
    )

    user_item_ids = list(set(explicit_items + implicit_items))

    # ------------------------------------------------
    # Get recommendation item_ids
    # ------------------------------------------------

    recommended_ids = get_recommendations(
        user_id=user_id,
        user_item_ids=user_item_ids,
        k=k,
        exclude_item_ids=user_item_ids
    )

    # ------------------------------------------------
    # Load full item metadata
    # ------------------------------------------------

    items = get_items_by_ids(recommended_ids)

    return items