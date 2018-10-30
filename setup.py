from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = open('./trainer.requirements.txt').read().splitlines()

setup(
    name='rpi-vision-trainer',
    version='0.0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Training application package for raspberry-pi-vision'
)
