
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

# website/dashboard/views.py
from django.shortcuts import render, redirect
from .models import EthosightApplication, UploadedImage
from configmanager.models import Config
from django.contrib.auth.decorators import login_required
from django.conf import settings
from Ethosight.EthosightApp import EthosightApp
from django.core.files.uploadedfile import InMemoryUploadedFile
import os
import logging
from django.http import JsonResponse

@login_required
def dashboard(request):
    user_apps = EthosightApplication.objects.filter(user=request.user)
    return render(request, 'dashboard.html', {'user_apps': user_apps})

@login_required
def create_app(request):
    configs = Config.objects.all()
    if request.method == 'POST':
        # Handle the form submission to create a new EthosightApplication
        # e.g., extract form data and save to the database
        name = request.POST['name']
        analytics = request.POST['analytics']
        config = request.POST['config']
        EthosightApplication.objects.create(user=request.user, name=name, analytics=analytics)
        print(f"creating app with name: {name} and config: {config}")
        EthosightApp.create_app(name,config)
        return redirect('dashboard')

    context = {
        'configs': configs
    }
    return render(request, 'create_app.html', context)

def compute_results(image_path, app_name):
    app = EthosightApp(app_name, base_dir=settings.ETHOSIGHT_APP_BASE_DIR)
    result = app.run(image_path)

    logger = logging.getLogger(__name__)
    logger.debug(f"Results: {result}")
    labels = result.output['labels']
    scores = result.output['scores']
    results = zip(labels, scores)
    sorted_results = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)

    return sorted_results

@login_required
def model_interaction(request):
    img_url = None
    ethosight_apps = [app.name for app in EthosightApplication.objects.filter(user=request.user)]

    if request.method == 'POST':
        selected_app_name = request.POST.get('ethosightApp')

        if 'image' in request.FILES:  # If new image is uploaded
            image = request.FILES.get('image')
            if isinstance(image, InMemoryUploadedFile):
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    for chunk in image.chunks():
                        tmp_file.write(chunk)
                    tmp_path = tmp_file.name
            else:
                tmp_path = image.temporary_file_path()
            
            # Create the UploadedImage instance and get the image URL
            uploaded_image = UploadedImage.objects.create(user=request.user, image=image)
            img_url = uploaded_image.image.url
        else:  
            img_url = request.POST.get('image_url')
            tmp_path = img_url  # Assuming img_url is a file path on the server

        sorted_results = compute_results(tmp_path, selected_app_name)

        # Optionally delete the temporary file after processing, if a new image was uploaded
        if 'image' in request.FILES:
            os.remove(tmp_path)

        context = {
            'results': sorted_results,
            'image_url': img_url,
            'selected_app_name': selected_app_name,
        }
        return render(request, 'results_image.html', context)

    return render(request, 'model_interaction.html', {'image_url': img_url,
        'ethosight_apps': ethosight_apps})

@login_required
def delete_app(request, app_id):
    # Check if the method is POST
    if request.method == "POST":
        try:
            app = EthosightApplication.objects.get(id=app_id, user=request.user)
            appname = app.name
            EthosightApp.delete_app(appname)
            app.delete()
            return JsonResponse({"status": "success", "message": "App deleted successfully."})
        except EthosightApplication.DoesNotExist:
            return JsonResponse({"status": "error", "message": "App not found."}, status=404)
    else:
        return JsonResponse({"status": "error", "message": "Invalid request method."}, status=400)

@login_required
def add_labels(request):
    if request.method == 'POST':
        new_labels = request.POST.get('new_labels')
        labels_list = [label.strip().replace(' ','') for label in new_labels.split(',')]
        # ... your logic for adding labels and computing embeddings ...

        # Get the image URL from the results page
        image_url = request.POST.get('image_url')  # Adjust this based on how you're passing the image URL

        # Convert the image URL to an absolute file path
        if settings.MEDIA_URL and image_url.startswith(settings.MEDIA_URL):
            image_path = os.path.join(settings.MEDIA_ROOT, image_url[len(settings.MEDIA_URL):])
        else:
            image_path = os.path.join(settings.MEDIA_ROOT, image_url.lstrip('/'))

        # Get the name of the EthosightApp
        selected_app_name = request.POST.get('selected_app_name')  # Make sure this is being passed correctly
        app = EthosightApp(selected_app_name, base_dir=settings.ETHOSIGHT_APP_BASE_DIR)
        app.add_labels(labels_list)


        print(f"selected_app_name: {selected_app_name}")
        print(f"image_url: {image_url}")
        # Recompute the results using the saved image's URL
        sorted_results = compute_results(image_path, selected_app_name)
        
        context = {
            'results': sorted_results,
            'image_url': image_url,
            'selected_app_name': selected_app_name,
        }
        return render(request, 'results_image.html', context)
