from django.db import models
from django.contrib.auth.models import User

class UserBrowsingLog(models.Model):
    user_id = models.CharField(max_length=100, db_index=True)
    item_id = models.CharField(max_length=100, db_index=True)

    event_type = models.CharField(max_length=20)  # view, click, add_to_cart
    session_id = models.CharField(max_length=100, null=True, blank=True)

    device = models.CharField(max_length=50, null=True, blank=True)
    source = models.CharField(max_length=50, null=True, blank=True)

    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['user_id', 'timestamp']),
            models.Index(fields=['item_id']),
        ]