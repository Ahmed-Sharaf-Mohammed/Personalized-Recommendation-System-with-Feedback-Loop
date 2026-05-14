from django.contrib import admin
from django.utils.html import format_html
from recommender.models import Item, UserInteraction, UserBrowsingLog, SearchLog


@admin.register(Item)
class ItemAdmin(admin.ModelAdmin):
    list_display   = ["item_id", "thumbnail", "title_short", "category",
                      "price", "avg_rating", "rating_count", "has_image"]
    list_filter    = ["category"]
    search_fields  = ["item_id", "title", "store"]
    ordering       = ["-avg_rating"]
    readonly_fields = ["thumbnail"]

    def title_short(self, obj): return obj.title[:55]
    title_short.short_description = "Title"

    def has_image(self, obj):
        return format_html('<i class="{}"></i>',
            "bi bi-check-circle-fill" if obj.images else "bi bi-x-circle")
    has_image.short_description = "Image"

    def thumbnail(self, obj):
        img = obj.display_image  # property بتستخرج من images
        if "picsum" not in img:  # لو مش fallback يعرضها
            return format_html(
                '<img src="{}" style="height:60px;border-radius:4px" />',
                img
            )
        return "—"
    thumbnail.short_description = "Preview"


@admin.register(UserInteraction)
class UserInteractionAdmin(admin.ModelAdmin):
    list_display  = ["user_id_short", "item_id", "rating", "review_title_short",
                     "verified", "helpful_votes", "timestamp"]
    list_filter   = ["verified", "rating"]
    search_fields = ["user_id", "item_id", "review_title"]
    ordering      = ["-timestamp"]

    def user_id_short(self, obj): return obj.user_id[:20]
    user_id_short.short_description = "User"

    def review_title_short(self, obj):
        return (obj.review_title or "")[:50]
    review_title_short.short_description = "Review Title"


@admin.register(UserBrowsingLog)
class UserBrowsingLogAdmin(admin.ModelAdmin):
    list_display  = ["user_id_short", "item_id", "event_type", "source", "device", "timestamp"]
    list_filter   = ["event_type", "source", "device"]
    search_fields = ["user_id", "item_id", "session_id"]
    ordering      = ["-timestamp"]

    def user_id_short(self, obj): return obj.user_id[:20]
    user_id_short.short_description = "User"


@admin.register(SearchLog)
class SearchLogAdmin(admin.ModelAdmin):
    list_display  = ["user_id_short", "query", "results_count", "device", "timestamp"]
    search_fields = ["user_id", "query"]
    ordering      = ["-timestamp"]

    def user_id_short(self, obj): return obj.user_id[:20]
    user_id_short.short_description = "User"