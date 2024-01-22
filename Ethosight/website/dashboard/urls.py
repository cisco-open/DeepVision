# website/urls.py or website/dashboard/urls.py
from django.urls import path
from .views import dashboard, create_app, model_interaction
from . import views

urlpatterns = [
    # ... other URL patterns
    path('', dashboard, name='dashboard'),
    path('create_app/', create_app, name='create_app'),
    path('model_interaction/', views.model_interaction, name='model_interaction'),
    path('delete_app/<int:app_id>/', views.delete_app, name='delete_app'),
    path('add_labels/', views.add_labels, name='add_labels'),
]
