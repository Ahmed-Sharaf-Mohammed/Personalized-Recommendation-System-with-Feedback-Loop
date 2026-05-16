"""
Browsing Tracker — persists browsing events to UserBrowsingLog DB model.

Schema matches browsing_logs for ML retraining:
  user_id, item_id, event_type, timestamp, session_id, device, source

event_type choices: view, click, add_to_cart, remove_from_cart, search
source choices: homepage, search, recommendation, category, direct
"""
import logging
from recommender.models import UserBrowsingLog, SearchLog
from recommender.tracking.session_tracker import (
    get_or_create_session_id,
    detect_device,
    track_item_view,
    track_search,
)

logger = logging.getLogger(__name__)

# Dedicated logger → writes to logs/interactions.log via LOGGING config
_ilog = logging.getLogger("recommender.interactions")


def _get_user_id(request) -> str:
    """Return a stable user identifier (auth user ID or session key)."""
    if request.user.is_authenticated:
        return str(request.user.id)
    return f"anon_{request.session.session_key or 'unknown'}"


def log_event(request, item_id: str, event_type: str, source: str = "direct"):
    """
    Log a single browsing event to the database AND to interactions.log.
    Also updates session tracking for recently-viewed items.
    """
    user_id    = _get_user_id(request)
    session_id = get_or_create_session_id(request)
    device     = detect_device(request)

    try:
        UserBrowsingLog.objects.create(
            user_id=user_id,
            item_id=str(item_id),
            event_type=event_type,
            session_id=session_id,
            device=device,
            source=source,
        )
        if event_type == "view":
            track_item_view(request, item_id)

        # ── Write to interactions.log ──────────────────────────────────────
        _ilog.info(
            "event=%-20s user=%-10s item=%-15s source=%-15s device=%s session=%s",
            event_type, user_id, item_id, source, device, session_id,
        )

    except Exception as e:
        logger.error("[BrowsingTracker] Failed to log event: %s", e)


def log_search_event(request, query: str, results_count: int = 0):
    """Log a search query event to DB and interactions.log."""
    user_id    = _get_user_id(request)
    session_id = get_or_create_session_id(request)
    device     = detect_device(request)

    try:
        SearchLog.objects.create(
            user_id=user_id,
            query=query,
            results_count=results_count,
            session_id=session_id,
            device=device,
        )
        track_search(request, query)

        # ── Write to interactions.log ──────────────────────────────────────
        _ilog.info(
            "event=%-20s user=%-10s query=%r results=%d device=%s session=%s",
            "search", user_id, query, results_count, device, session_id,
        )

    except Exception as e:
        logger.error("[BrowsingTracker] Failed to log search: %s", e)
