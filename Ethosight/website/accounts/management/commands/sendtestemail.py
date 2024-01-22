from django.core.management.base import BaseCommand
from django.core.mail import send_mail

class Command(BaseCommand):
    help = 'Sends a test email.'

    def handle(self, *args, **kwargs):
        send_mail(
            'Test Email Subject',
            'This is a test email.',
            'hlatapie@cisco.com', #from
            ['hugo@latapiefamily.net'], #to 
            fail_silently=False,
        )
        self.stdout.write(self.style.SUCCESS('Test email sent!'))

