# configmanager/urls.py

from django.urls import path, re_path
from . import views

urlpatterns = [
    # ... your other URL patterns ...
    re_path(r'^save_to_yaml/(?P<config_title>[\w\-]+)/$', views.save_config_to_yaml, name='save_config_to_yaml'),
]

