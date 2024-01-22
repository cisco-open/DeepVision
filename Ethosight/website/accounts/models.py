from django.contrib.auth.models import AbstractUser 
from django.db import models 
from django.utils.translation import gettext as _
 
class CustomUser(AbstractUser):
    # your other fields
    
    groups = models.ManyToManyField(
        'auth.Group',
        verbose_name=_('groups'),
        blank=True,
        help_text=_('The groups this user belongs to. A user will get all permissions granted to each of their groups.'),
        related_name="customuser_groups",
        related_query_name="customuser",
    )
    
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        verbose_name=_('user permissions'),
        blank=True,
        help_text=_('Specific permissions for this user.'),
        related_name="customuser_user_permissions",
        related_query_name="customuser",
    )

class PendingUser(models.Model):
    full_name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    requested_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(choices=[('pending', 'Pending'), ('approved', 'Approved'), ('denied', 'Denied')], default='pending', max_length=10)


class AccessCode(models.Model): 
    code = models.CharField(max_length=255, unique=True) 
    used = models.BooleanField(default=False) 
 
