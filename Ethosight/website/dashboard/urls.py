
# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

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
