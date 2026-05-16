# ml/inference/loader.py
"""
Item Loader
───────────
Thin wrapper around the Django ORM for ML inference code that needs to
look up Item metadata from the recommendation results.

Used by recommender_service.py and any inference code that needs
item details beyond just the item_id.
"""

import logging

logger = logging.getLogger(__name__)


def get_item_by_id(item_id: str) -> dict | None:
    """
    Fetch a single item's metadata dict from the DB.
    Returns None if the item doesn't exist.
    """
    try:
        import django
        import os
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
        from recommender.models import Item

        item = Item.objects.filter(item_id=str(item_id)).values(
            "item_id", "title", "category", "price",
            "avg_rating", "rating_count", "store",
        ).first()
        return dict(item) if item else None

    except Exception as e:
        logger.error("[Loader] get_item_by_id(%s) error: %s", item_id, e)
        return None


def get_items_by_ids(item_ids: list) -> list[dict]:
    """
    Bulk fetch item metadata for a list of item_ids.
    Preserves the order of item_ids (recommendation rank order).
    Returns only items that exist in the DB.
    """
    if not item_ids:
        return []
    try:
        import os
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
        from recommender.models import Item

        rows = Item.objects.filter(item_id__in=item_ids).values(
            "item_id", "title", "category", "price",
            "avg_rating", "rating_count", "store",
        )
        lookup = {r["item_id"]: dict(r) for r in rows}
        return [lookup[iid] for iid in item_ids if iid in lookup]

    except Exception as e:
        logger.error("[Loader] get_items_by_ids error: %s", e)
        return []
