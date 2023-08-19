import pkg_resources
import os

def get_install_path(package_name: str) -> str:
    distribution = pkg_resources.get_distribution(package_name)
    install_path = distribution.location
    return install_path
