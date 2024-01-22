from django.db import models
from django.conf import settings
from ruamel.yaml import YAML 
import string
import os

def sanitize_filename(filename):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return ''.join(c for c in filename if c in valid_chars)

class Config(models.Model):
    title = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True, null=True)
    content = models.TextField()  # This will store the YAML content
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title

    def save_to_yaml(self, directory_path):
        """
        Save the configuration to a YAML file using ruamel.yaml for comment preservation.
        :param directory_path: Directory path to save the YAML file.
        """
        sanitized_title = sanitize_filename(self.title)
        file_name = f"{sanitized_title}.yaml"
        file_path = os.path.join(directory_path, file_name)

        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.default_flow_style = False

        with open(file_path, 'w') as yaml_file:
            yaml.dump(yaml.load(self.content), yaml_file)

        # Post-processing to replace Windows line endings with UNIX line endings
        with open(file_path, 'rb') as f:
            content = f.read()

        content = content.replace(b'\r\n', b'\n')

        with open(file_path, 'wb') as f:
            f.write(content)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)  # First call the original save method
        self.save_to_yaml(settings.CONFIG_YAML_DIRECTORY)  # Then save to YAML

