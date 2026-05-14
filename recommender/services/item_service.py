"""
Item Service — queries Items from SQLite (fast, no parquet I/O at runtime).
All heavy parquet loading happens once via: python manage.py import_data
"""
import math
from django.db.models import Q
from recommender.models import Item

PAGE_SIZE = 24


def search_items(query="", category="", page=1, sort_by="relevance",
                 min_price=None, max_price=None):
    qs = Item.objects.all()

    if query:
        qs = qs.filter(
            Q(title__icontains=query) |
            Q(category__icontains=query) |
            Q(store__icontains=query)
        )

    if category and category.lower() != "all":
        qs = qs.filter(category__iexact=category)

    if min_price is not None:
        qs = qs.filter(price__gte=min_price)
    if max_price is not None:
        qs = qs.filter(price__lte=max_price)

    # Sorting
    order_map = {
        "rating":     "-avg_rating",
        "popular":    "-rating_count",
        "price_asc":  "price",
        "price_desc": "-price",
        "relevance":  "-rating_count",
    }
    qs = qs.order_by(order_map.get(sort_by, "-rating_count"))

    total = qs.count()
    total_pages = max(1, math.ceil(total / PAGE_SIZE))
    page = max(1, min(page, total_pages))
    start = (page - 1) * PAGE_SIZE

    items = list(qs[start:start + PAGE_SIZE])

    return {
        "items": items,
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "has_previous": page > 1,
        "has_next": page < total_pages,
        "previous_page": page - 1,
        "next_page": page + 1,
        "page_range": _page_range(page, total_pages),
    }


def get_featured_items(limit=8):
    return list(
        Item.objects.filter(avg_rating__gte=4.0)
        .order_by("-rating_count")[:limit]
    )


def get_category_previews(per_category=4):
    from django.db.models import Count
    cats = (
        Item.objects.values_list("category", flat=True)
        .annotate(n=Count("id"))
        .order_by("-n")[:6]
        .values_list("category", flat=True)
    )
    # Simpler: just get top categories
    top_cats = list(
        Item.objects.values("category")
        .annotate(n=Count("id"))
        .order_by("-n")[:6]
        .values_list("category", flat=True)
    )
    result = []
    for cat in top_cats:
        items = list(Item.objects.filter(category=cat).order_by("-rating_count")[:per_category])
        result.append({"category": cat, "items": items})
    return result


def get_item_detail(item_id):
    try:
        return Item.objects.get(item_id=item_id)
    except Item.DoesNotExist:
        return None


def get_similar_items(item_id, limit=6):
    try:
        item = Item.objects.get(item_id=item_id)
    except Item.DoesNotExist:
        return []
    return list(
        Item.objects.filter(category=item.category)
        .exclude(item_id=item_id)
        .order_by("-avg_rating")[:limit]
    )


def get_items_by_ids(item_ids):
    if not item_ids:
        return []
    items = {i.item_id: i for i in Item.objects.filter(item_id__in=item_ids)}
    return [items[iid] for iid in item_ids if iid in items]


def get_all_categories():
    from django.db.models import Count
    return list(
        Item.objects.values_list("category", flat=True)
        .annotate(n=Count("id"))
        .order_by("-n")
        .values_list("category", flat=True)
    )


def _page_range(current, total):
    if total <= 7:
        return list(range(1, total + 1))
    pages = sorted({1, total, *range(max(1, current-2), min(total+1, current+3))})
    out, prev = [], None
    for p in pages:
        if prev and p - prev > 1:
            out.append(-1)
        out.append(p)
        prev = p
    return out
