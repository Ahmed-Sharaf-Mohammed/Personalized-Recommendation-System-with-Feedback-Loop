"""
AJAX API endpoints — event tracking + review submission.
"""
import json
import logging
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from recommender.tracking.browsing_tracker import log_event, log_search_event
from recommender.tracking.session_tracker import add_to_cart, remove_from_cart, get_cart
from recommender.services.interaction_service import log_interaction

logger = logging.getLogger(__name__)


@require_POST
def api_track_event(request):
    try:
        data       = json.loads(request.body)
        item_id    = data.get("item_id", "")
        event_type = data.get("event_type", "view")
        source     = data.get("source", "direct")

        if not item_id:
            return JsonResponse({"ok": False, "error": "item_id required"}, status=400)

        log_event(request, item_id, event_type, source)

        if event_type == "add_to_cart":
            add_to_cart(request, item_id)
        elif event_type == "remove_from_cart":
            remove_from_cart(request, item_id)

        return JsonResponse({"ok": True, "cart_count": len(get_cart(request))})
    except Exception as e:
        logger.error(f"[TrackAPI] {e}")
        return JsonResponse({"ok": False, "error": str(e)}, status=500)


@require_POST
def api_submit_rating(request):
    try:
        data         = json.loads(request.body)
        item_id      = data.get("item_id", "")
        rating       = data.get("rating")
        review_title = data.get("review_title", "").strip()
        review_text  = data.get("review_text", "").strip()

        if not item_id or rating is None:
            return JsonResponse({"ok": False, "error": "item_id and rating required"}, status=400)

        if not request.user.is_authenticated:
            return JsonResponse({"ok": False, "error": "Login required to submit a review."}, status=403)

        user_id = str(request.user.id)

        interaction = log_interaction(
            user_id=user_id,
            item_id=item_id,
            rating=float(rating),
            review_title=review_title or None,
            review_text=review_text or None,
            verified=True,
        )

        # Build inline HTML for the new review card
        stars_html = ""
        r = int(float(rating))
        for i in range(1, 6):
            stars_html += f'<i class="bi bi-star{"-fill" if i <= r else ""} text-warning small"></i>'

        review_html = f"""
        <div class="card border-0 shadow-sm mb-3 review-card review-new">
          <div class="card-body">
            <div class="d-flex justify-content-between align-items-start mb-2">
              <div>
                <div class="d-flex text-warning mb-1">
                  {stars_html}
                  <span class="ms-2 fw-semibold text-dark small">{r}/5</span>
                </div>
                {f'<h6 class="fw-bold mb-1">{review_title}</h6>' if review_title else ''}
              </div>
              <div class="text-end">
                <span class="badge bg-success-subtle text-success small">
                  <i class="bi bi-patch-check-fill me-1"></i>Verified
                </span>
                <div class="small text-muted mt-1">Just now</div>
              </div>
            </div>
            {f'<p class="text-muted small mb-0 lh-lg">{review_text[:400]}</p>' if review_text else ''}
          </div>
        </div>"""

        return JsonResponse({"ok": True, "review_html": review_html})
    except Exception as e:
        logger.error(f"[RateAPI] {e}")
        return JsonResponse({"ok": False, "error": str(e)}, status=500)


def api_cart_count(request):
    return JsonResponse({"count": len(get_cart(request))})
