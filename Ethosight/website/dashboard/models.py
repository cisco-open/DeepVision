
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

from django.db import models
from django.contrib.auth import get_user_model
from django.conf import settings
from Ethosight.EthosightApp import EthosightApp
from django.core.exceptions import ValidationError
from PIL import Image
import re

class EthosightApplication(models.Model): 
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE) 
    name = models.CharField(max_length=255) 
    analytics = models.TextField() 
    images_processed = models.IntegerField(default=0) 
    videos_analyzed = models.IntegerField(default=0) 

    def clean(self):
        # Check if name contains only valid directory characters
        if not re.match("^[a-zA-Z0-9_-]+$", self.name):
            raise ValidationError({
                'name': "Name can only contain alphanumeric characters, hyphens, and underscores."
            })

    def save(self, *args, **kwargs):
        self.clean()
        super().save(*args, **kwargs)

    def get_app_directory(self):
        return os.path.join(settings.ETHOSIGHT_APP_BASE_DIR, self.user.username, self.name)

    def instantiate_app(self):
        # Assuming EthosightApp takes the name as a parameter for instantiation
        return EthosightApp(self.name)


def validate_image_extension(value):
    ext = os.path.splitext(value.name)[1]
    valid_extensions = ['.jpg', '.png', '.jpeg']
    if ext.lower() not in valid_extensions:
        raise ValidationError('Unsupported file extension. Please upload a JPG or PNG image.')

class UploadedImage(models.Model):
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)  # Link the uploaded image to a user
    image = models.ImageField(upload_to='uploads/', validators=[validate_image_extension])
    def save(self, *args, **kwargs):
        # Ensure the image is opened using Pillow
        img = Image.open(self.image)

        # Check dimensions
#        if img.height > 800 or img.width > 800:
#            raise ValueError("Image is too large. Please upload images with max dimensions 800x800.")

        # Check the file size
        if self.image.size > 5*1024*1024:  # 5 MB
            raise ValueError("Image file too large ( > 5mb )")

        super().save(*args, **kwargs)

    def get_image_url(self):
        return self.image.url