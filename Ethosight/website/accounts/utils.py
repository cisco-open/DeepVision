from django.core.mail import send_mail
import redis
import uuid
import logging

r = redis.Redis(host='localhost', port=6379, db=0)

logger = logging.getLogger(__name__)

def approve_user(pending_user):
    try:
        # 1. Generate Access Code
        access_code = str(uuid.uuid4())
        r.hset("access_codes", access_code, "unused")
        logger.info(f"Access code generated for {pending_user.email}: {access_code}")

        # 2. Email Notification
        send_mail(
            'Your Access Code for Our Website',
            f'Congratulations! You have been approved. Use the following access code to register: {access_code}',
            'hlatapie@cisco.com',
            [pending_user.email],
            fail_silently=False,
        )
        logger.info(f"Approval email sent to {pending_user.email}")

        # 3. Status Update
        pending_user.status = 'approved'
        pending_user.save()

    except Exception as e:
        logger.error(f"Error approving user {pending_user.email}: {e}")

