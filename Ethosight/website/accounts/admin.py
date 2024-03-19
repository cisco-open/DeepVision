
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

from django.contrib import admin
from .models import PendingUser
from .utils import approve_user

@admin.register(PendingUser)
class PendingUserAdmin(admin.ModelAdmin):
    list_display = ('full_name', 'email', 'requested_at', 'status')
    actions = ['approve_users', 'deny_users']

    def approve_users(self, request, queryset):
        for user in queryset:
            approve_user(user)

        self.message_user(request, f"{queryset.count()} users approved.")

    def deny_users(self, request, queryset):
        queryset.update(status='denied')
        self.message_user(request, f"{queryset.count()} users denied.")

    def save_model(self, request, obj, form, change):
        # Check if status has changed to 'approved'
        if 'status' in form.changed_data and obj.status == 'approved':
            approve_user(obj)
        super().save_model(request, obj, form, change)

    approve_users.short_description = "Approve selected users and send them an access code"
    deny_users.short_description = "Deny access code request for selected users"

