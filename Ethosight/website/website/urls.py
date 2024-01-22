from django.urls import path, include, re_path 
from accounts.views import register, user_login, home 
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [ 
    path('', home, name='home'),
    path('admin/', admin.site.urls),
    path('register/', register, name='register'), 
    path('login/', user_login, name='login'), 
    path('home/', home, name='home'), 
    path('dashboard/', include('dashboard.urls')),
    path('configmanager/', include('configmanager.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

urlpatterns += staticfiles_urlpatterns()
 
