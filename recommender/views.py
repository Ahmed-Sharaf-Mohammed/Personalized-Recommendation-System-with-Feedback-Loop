import logging
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import Http404

from recommender.services.item_service import (
    search_items, get_featured_items, get_category_previews,
    get_item_detail, get_similar_items, get_all_categories, get_items_by_ids,
)
from recommender.services.interaction_service import (
    get_user_interactions, get_item_reviews,
)
from recommender.tracking.browsing_tracker import log_event, log_search_event
from recommender.tracking.session_tracker import (
    get_recently_viewed, get_search_history, get_cart,
)

logger = logging.getLogger(__name__)


# ─── HOME ────────────────────────────────────────────────────────────────────
def home(request):
    featured          = get_featured_items(limit=8)
    category_previews = get_category_previews(per_category=4)
    categories        = get_all_categories()[:10]
    rv_ids            = get_recently_viewed(request)
    recently_viewed   = get_items_by_ids(rv_ids[:6])

    return render(request, "recommender/home.html", {
        "featured_items":    featured,
        "category_previews": category_previews,
        "categories":        categories,
        "recently_viewed":   recently_viewed,
        "cart_count":        len(get_cart(request)),
    })


# ─── PRODUCT LIST ─────────────────────────────────────────────────────────────
def product_list(request):
    query    = request.GET.get("q", "").strip()
    category = request.GET.get("category", "")
    sort_by  = request.GET.get("sort", "relevance")
    page     = _int(request.GET.get("page", 1))
    min_p    = _float(request.GET.get("min_price"))
    max_p    = _float(request.GET.get("max_price"))

    result = search_items(query=query, category=category, page=page,
                          sort_by=sort_by, min_price=min_p, max_price=max_p)

    return render(request, "recommender/product_list.html", {
        **result,
        "query":             query,
        "selected_category": category,
        "sort_by":           sort_by,
        "min_price":         min_p or "",
        "max_price":         max_p or "",
        "all_categories":    get_all_categories(),
        "cart_count":        len(get_cart(request)),
    })


# ─── PRODUCT DETAIL ───────────────────────────────────────────────────────────
def product_detail(request, item_id):
    item = get_item_detail(item_id)
    if item is None:
        raise Http404("Product not found")

    source  = request.GET.get("source", "direct")
    log_event(request, item_id, "view", source=source)

    similar = get_similar_items(item_id, limit=6)
    reviews = get_item_reviews(item_id, limit=20)

    # Has this user already rated this item?
    user_already_rated = False
    if request.user.is_authenticated:
        user_id = str(request.user.id)
        user_already_rated = any(
            r.user_id == user_id for r in reviews
        )

    return render(request, "recommender/product_detail.html", {
        "item":               item,
        "similar_items":      similar,
        "reviews":            reviews,
        "review_count":       len(reviews),
        "user_already_rated": user_already_rated,
        "cart_count":         len(get_cart(request)),
    })


# ─── SEARCH ───────────────────────────────────────────────────────────────────
def search(request):
    query    = request.GET.get("q", "").strip()
    category = request.GET.get("category", "")
    sort_by  = request.GET.get("sort", "relevance")
    page     = _int(request.GET.get("page", 1))

    result = {"items": [], "total": 0, "page": 1, "total_pages": 1,
               "has_previous": False, "has_next": False, "page_range": [1]}

    if query:
        result = search_items(query=query, category=category, page=page, sort_by=sort_by)
        log_search_event(request, query, results_count=result["total"])

    return render(request, "recommender/search.html", {
        **result,
        "query":             query,
        "selected_category": category,
        "sort_by":           sort_by,
        "search_history":    get_search_history(request),
        "all_categories":    get_all_categories(),
        "cart_count":        len(get_cart(request)),
    })


# ─── DASHBOARD ────────────────────────────────────────────────────────────────
@login_required
def dashboard(request):
    user_id      = str(request.user.id)
    interactions = get_user_interactions(user_id)
    rated_ids    = [i["item_id"] for i in interactions[:20]]
    rated_map    = {i.item_id: i for i in get_items_by_ids(rated_ids)}

    rated_with_meta = []
    for inter in interactions[:20]:
        item = rated_map.get(inter["item_id"])
        rated_with_meta.append({
            "item":         item,
            "item_id":      inter["item_id"],
            "rating":       inter["rating"],
            "review_title": inter["review_title"],
            "timestamp":    inter["timestamp"],
        })

    rv_ids      = get_recently_viewed(request)
    cart_ids    = get_cart(request)

    return render(request, "recommender/dashboard.html", {
        "rated_items":      rated_with_meta,
        "recently_viewed":  get_items_by_ids(rv_ids[:12]),
        "cart_items":       get_items_by_ids(cart_ids),
        "search_history":   get_search_history(request),
        "recommendations":  [],
        "cart_count":       len(cart_ids),
        "interaction_count": len(interactions),
    })


# ─── AUTH ─────────────────────────────────────────────────────────────────────
def login_view(request):
    if request.user.is_authenticated:
        return redirect("recommender:home")
    if request.method == "POST":
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}!")
            return redirect(request.GET.get("next") or "recommender:home")
        messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()
    return render(request, "recommender/auth/login.html",
                  {"form": form, "next": request.GET.get("next", "")})


def logout_view(request):
    logout(request)
    messages.info(request, "You've been logged out.")
    return redirect("recommender:home")


def register_view(request):
    if request.user.is_authenticated:
        return redirect("recommender:home")
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f"Welcome, {user.username}! Start exploring.")
            return redirect("recommender:home")
        messages.error(request, "Please fix the errors below.")
    else:
        form = UserCreationForm()
    return render(request, "recommender/auth/register.html", {"form": form})


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _float(v):
    try:
        return float(v) if v else None
    except (ValueError, TypeError):
        return None


def _int(v, default=1):
    try:
        return int(v)
    except (ValueError, TypeError):
        return default
