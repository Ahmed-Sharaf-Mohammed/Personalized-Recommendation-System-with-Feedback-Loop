from django.urls import path
from recommender import views
from recommender.api_helpers.track_api import api_track_event, api_submit_rating, api_cart_count

app_name = "recommender"

urlpatterns = [
    # Pages
    path("", views.home, name="home"),
    path("products/", views.product_list, name="product_list"),
    path("products/<str:item_id>/", views.product_detail, name="product_detail"),
    path("search/", views.search, name="search"),
    path("dashboard/", views.dashboard, name="dashboard"),

    # Auth
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("register/", views.register_view, name="register"),

    # AJAX tracking API
    path("api/track/", api_track_event, name="api_track"),
    path("api/rate/", api_submit_rating, name="api_rate"),
    path("api/cart/count/", api_cart_count, name="api_cart_count"),
]
