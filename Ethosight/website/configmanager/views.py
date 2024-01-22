from django.conf import settings
from django.contrib.auth.decorators import user_passes_test
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from .models import Config

def is_superuser_check(user):
    return user.is_superuser

@user_passes_test(is_superuser_check)
def save_config_to_yaml(request, config_title=None):
    if config_title:
        config = get_object_or_404(Config, title=config_title)
    else:
        return HttpResponse("Config title not provided", status=400)

    directory_path = settings.CONFIG_YAML_DIRECTORY
    config.save_to_yaml(directory_path)
    return HttpResponse(f"Config with title '{config_title}' saved to {directory_path}")

