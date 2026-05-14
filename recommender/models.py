from django.db import models


class Item(models.Model):
    """
    SQLite cache of items.parquet — fast ORM queries at runtime.
    Extra columns discovered: images, videos, bought_together
    Populated once via: python manage.py import_data
    """
    item_id       = models.CharField(max_length=50, unique=True, db_index=True)
    title         = models.CharField(max_length=500, default="Untitled")
    category      = models.CharField(max_length=200, default="General", db_index=True)
    categories    = models.TextField(blank=True, null=True)
    description   = models.TextField(blank=True, null=True)
    features      = models.TextField(blank=True, null=True)
    price         = models.FloatField(null=True, blank=True)
    avg_rating    = models.FloatField(default=0.0)
    rating_count  = models.IntegerField(default=0)
    store           = models.CharField(max_length=200, blank=True, null=True)
    details         = models.TextField(blank=True, null=True)
    images          = models.TextField(blank=True, null=True)          # raw images (كما في المصدر)
    videos          = models.TextField(blank=True, null=True)          # raw videos
    bought_together = models.TextField(blank=True, null=True)          # raw bought_together

    class Meta:
        indexes = [
            models.Index(fields=["category"]),
            models.Index(fields=["avg_rating"]),
            models.Index(fields=["price"]),
        ]

    def __str__(self):
        return f"{self.item_id} — {self.title[:50]}"

    @property
    def price_display(self):
        return f"{self.price:.2f}" if self.price else None

    @property
    def display_image(self):
        """Extract first Amazon image URL from raw images field, else picsum fallback."""
        import re
        if self.images:            
            m = re.search(r'https://m\.media-amazon\.com/images/[^\s\'"\],]+', str(self.images))
            if m:
                return m.group(0)
        seed = abs(hash(self.item_id)) % 1000
        return f"https://picsum.photos/seed/{seed}/400/300"

    @property
    def rating_stars(self):
        return round(self.avg_rating)


class UserInteraction(models.Model):
    user_id       = models.CharField(max_length=100, db_index=True)
    item_id       = models.CharField(max_length=50, db_index=True)
    rating        = models.FloatField(null=True, blank=True)
    review_text   = models.TextField(blank=True, null=True)
    review_title  = models.CharField(max_length=300, blank=True, null=True)
    verified      = models.BooleanField(default=False)
    helpful_votes = models.IntegerField(default=0)
    timestamp     = models.DateTimeField(null=True, blank=True)  # من الداتا عند الاستيراد، auto_now لو مش موجود
    images        = models.TextField(blank=True, null=True)    # raw review images

    class Meta:
        indexes = [
            models.Index(fields=["item_id", "timestamp"]),
            models.Index(fields=["user_id", "timestamp"]),
        ]

    @property
    def rating_int(self):
        return int(self.rating) if self.rating else 0


class UserBrowsingLog(models.Model):
    EVENT_TYPES  = [('view','View'),('click','Click'),('add_to_cart','Add to Cart'),
                    ('remove_from_cart','Remove from Cart'),('search','Search')]
    SOURCE_TYPES = [('homepage','Homepage'),('search','Search'),
                    ('recommendation','Recommendation'),('category','Category'),('direct','Direct')]

    user_id    = models.CharField(max_length=100, db_index=True)
    item_id    = models.CharField(max_length=50, db_index=True)
    event_type = models.CharField(max_length=20, choices=EVENT_TYPES)
    session_id = models.CharField(max_length=100, blank=True, null=True)
    device     = models.CharField(max_length=50, blank=True, null=True)
    source     = models.CharField(max_length=50, blank=True, null=True, choices=SOURCE_TYPES)
    timestamp  = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["user_id", "timestamp"]),
            models.Index(fields=["item_id"]),
            models.Index(fields=["session_id"]),
        ]


class SearchLog(models.Model):
    user_id       = models.CharField(max_length=100, db_index=True)
    query         = models.CharField(max_length=500)
    results_count = models.IntegerField(default=0)
    session_id    = models.CharField(max_length=100, blank=True, null=True)
    device        = models.CharField(max_length=50, blank=True, null=True)
    timestamp     = models.DateTimeField(auto_now_add=True)
