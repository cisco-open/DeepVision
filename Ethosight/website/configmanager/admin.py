from django.contrib import admin
from .models import Config

class ConfigAdmin(admin.ModelAdmin):
    class Media:
        js = ('configmanager/js/ace_editor_init.js',)

admin.site.register(Config, ConfigAdmin)

