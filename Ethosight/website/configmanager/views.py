
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

