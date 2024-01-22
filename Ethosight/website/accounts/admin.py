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

