
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

from django.shortcuts import render, redirect 
from django.contrib.auth import authenticate, login 
from .models import CustomUser, AccessCode 
import redis
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect 
from django.contrib.auth import authenticate, login 
from .models import CustomUser, AccessCode, PendingUser

r = redis.Redis(host='localhost', port=6379, db=0)

def register(request): 
    if request.method == 'POST':
        # Extract data from form
        username = request.POST['username']
        password = request.POST['password']
        full_name = request.POST.get('full_name', '')  # New field
        email = request.POST.get('email', '')  # New field

        # Check if it's a request for an access code
        if 'request_code' in request.POST:
            # Check if email already exists
            if PendingUser.objects.filter(email=email).exists():
                return render(request, 'register.html', {'error': 'Access code request already exists for this email.'})

            # Save the pending user data
            pending_user = PendingUser(full_name=full_name, email=email)
            pending_user.save()

            # Inform the user
            return render(request, 'register.html', {'message': 'Access code request submitted. Please wait for approval.'})

        if CustomUser.objects.filter(username=username).exists(): 
            return render(request, 'register.html', {'error': 'User already exists.'})

        access_code = request.POST['access_code']

        # Check if the access code exists and is unused in Redis
        status = r.hget("access_codes", access_code)
        if not status or status.decode('utf-8') != "unused":
            return render(request, 'register.html', {'error': 'Invalid or used access code.'})

        user = CustomUser.objects.create_user(username=username, password=password, email=email)  # Assuming your CustomUser model has an email field

        # After a successful registration, mark the access code as used:
        r.hset("access_codes", access_code, "used")        

#        code_entry.used = True 
#        code_entry.save() 
        return redirect('login')

    return render(request, 'register.html')

def user_login(request): 
    if request.method == 'POST': 
        username = request.POST['username'] 
        password = request.POST['password'] 
        user = authenticate(request, username=username, password=password) 
        if user: 
            login(request, user) 
            return redirect('home') 
        else: 
            return render(request, 'login.html', {'error': 'Invalid username or password.'}) 
    return render(request, 'login.html') 

@login_required 
def home(request): 
    return render(request, 'home.html') 
 
