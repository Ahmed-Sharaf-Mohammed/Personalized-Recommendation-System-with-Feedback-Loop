"""
Session Tracker — lightweight behavioral browsing/session tracking.

This is NOT product caching. It tracks user behavior in the Django session
so the frontend can show recently viewed items, session history, etc.
All data is schema-compatible with browsing_logs for ML retraining.

Session keys used:
    _session_id       — UUID for this browser session
    _recently_viewed  — list of item_ids (max 20)
    _search_history   — list of search queries (max 10)
    _cart             — list of item_ids in cart
    _device           — detected device type
"""
import uuid
import logging
from django.utils.timezone import now

logger = logging.getLogger(__name__)

MAX_RECENTLY_VIEWED = 20
MAX_SEARCH_HISTORY = 10


def get_or_create_session_id(request) -> str:
    """Return the persistent session ID for this browser session."""
    if "_session_id" not in request.session:
        request.session["_session_id"] = str(uuid.uuid4())
    return request.session["_session_id"]


def detect_device(request) -> str:
    ua = request.META.get("HTTP_USER_AGENT", "").lower()
    if "mobile" in ua or "android" in ua:
        return "mobile"
    if "tablet" in ua or "ipad" in ua:
        return "tablet"
    return "desktop"


def track_item_view(request, item_id: str):
    """Add item to recently viewed list (session-based)."""
    viewed = request.session.get("_recently_viewed", [])
    item_id = str(item_id)
    if item_id in viewed:
        viewed.remove(item_id)
    viewed.insert(0, item_id)
    request.session["_recently_viewed"] = viewed[:MAX_RECENTLY_VIEWED]
    request.session.modified = True


def track_search(request, query: str):
    """Add search query to session history."""
    history = request.session.get("_search_history", [])
    if query and query not in history:
        history.insert(0, query)
    request.session["_search_history"] = history[:MAX_SEARCH_HISTORY]
    request.session.modified = True


def add_to_cart(request, item_id: str):
    """Add item to session cart."""
    cart = request.session.get("_cart", [])
    if item_id not in cart:
        cart.append(str(item_id))
    request.session["_cart"] = cart
    request.session.modified = True


def remove_from_cart(request, item_id: str):
    """Remove item from session cart."""
    cart = request.session.get("_cart", [])
    cart = [i for i in cart if i != str(item_id)]
    request.session["_cart"] = cart
    request.session.modified = True


def get_cart(request) -> list[str]:
    return request.session.get("_cart", [])


def get_recently_viewed(request) -> list[str]:
    return request.session.get("_recently_viewed", [])


def get_search_history(request) -> list[str]:
    return request.session.get("_search_history", [])


def clear_session_data(request):
    for key in ("_recently_viewed", "_search_history", "_cart"):
        request.session.pop(key, None)
    request.session.modified = True
