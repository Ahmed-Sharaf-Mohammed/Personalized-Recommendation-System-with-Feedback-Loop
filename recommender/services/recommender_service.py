from ml.inference.predict import get_recommendations
from ml.utils.loader import get_item
from recommender.models import UserInteraction
from django.utils import timezone

def log_interaction(user_id, item_id, action_type='view', rating=None):
    """تسجيل تفاعل المستخدم في قاعدة البيانات"""
    UserInteraction.objects.create(
        user_id=user_id,
        item_id=item_id,
        rating=rating,
        action_type=action_type,
        timestamp=timezone.now()
    )

def get_item_details(item_id):
    return get_item(item_id)

def get_user_recommendations(user_id):
    # هنا يمكن استخدام الموديل من ML
    return get_recommendations(user_id)