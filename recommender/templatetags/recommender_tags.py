"""
Custom template tags for the recommender app.
Handles parsing of parquet list-strings for display.
"""
import re
from django import template

register = template.Library()


@register.filter(name="parse_list")
def parse_list(value) -> list:
    """
    Turn a stringified Python list from parquet into a real list.
    e.g. "['Electronics', 'Computers & Accessories']" → ['Electronics', 'Computers & Accessories']
    Safe — never crashes.
    """
    if not value:
        return []
    s = str(value).strip()
    if s in ("[]", "None", "nan", ""):
        return []
    # Extract quoted items
    items = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
    result = [a or b for a, b in items if (a or b).strip()]
    if result:
        return result
    # Fallback: strip brackets and return as single string
    cleaned = s.strip("[]").strip()
    return [cleaned] if cleaned else []


@register.filter(name="clean_text")
def clean_text(value) -> str:
    """
    Clean a parquet text field — remove list brackets, fix escaped quotes.
    Returns a plain readable string.
    """
    if not value:
        return ""
    s = str(value).strip()
    if s in ("[]", "None", "nan", ""):
        return ""
    # If it's a list-string, join items
    items = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
    if items:
        parts = [a or b for a, b in items if (a or b).strip()]
        return " ".join(parts)
    # Plain string: strip brackets
    return s.strip("[]'\"").strip()


@register.filter(name="parse_video_urls")
def parse_video_urls(value) -> list:
    """
    Extract video URLs from the raw videos field.
    Handles Amazon video URLs (https://...) stored in the parquet videos column.
    """
    if not value:
        return []
    s = str(value)
    urls = re.findall(r"https?://[^\s'\",\]]+\.(?:mp4|m3u8|mov|webm|com/[^\s'\",\]]{10,})", s)
    # fallback: any https URL in the field
    if not urls:
        urls = re.findall(r"https?://[^\s'\",\]]{20,}", s)
    return [u for u in urls if u][:5]


@register.filter(name="star_range")
def star_range(rating) -> list:
    """Return list of (filled, index) for star rendering."""
    try:
        r = round(float(rating))
    except (TypeError, ValueError):
        r = 0
    return [(i <= r, i) for i in range(1, 6)]
