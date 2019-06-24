<<<<<<< HEAD
from setuptools import find_packages
from setuptools import setup
import os
import trainers


REQUIRED_PACKAGES = [
    "absl-py==0.6.1",
    "appnope==0.1.0",
    "astor==0.7.1",
    "backcall==0.1.0",
    "bleach==3.0.2",
    "cachetools==2.1.0",
    "certifi==2018.10.15",
    "chardet==3.0.4",
    "decorator==4.3.0",
    "defusedxml==0.5.0",
    "entrypoints==0.2.3",
    "gast==0.2.0",
    "google-api-core==1.5.0",
    "google-auth==1.5.1",
    "google-cloud-core==0.28.1",
    "google-cloud-storage==1.13.0",
    "google-resumable-media==0.3.1",
    "googleapis-common-protos==1.5.3",
    "grpcio==1.16.0",
    "h5py==2.8.0",
    "idna==2.7",
    "jedi==0.13.1",
    "Jinja2==2.10",
    "jsonschema==2.6.0",
    "Keras==2.2.4",
    "Keras-Applications==1.0.6",
    "Keras-Preprocessing==1.0.5",
    "Markdown==3.0.1",
    "MarkupSafe==1.0",
    "mistune==0.8.4",
    "nbconvert==5.4.0",
    "nbformat==4.4.0",
    "numpy==1.14.5",
    "pandas==0.23.4",
    "pandocfilters==1.4.2",
    "parso==0.3.1",
    "pexpect==4.6.0",
    "pickleshare==0.7.5",
    "Pillow==5.3.0",
    "prometheus-client==0.4.2",
    "prompt-toolkit==2.0.6",
    "protobuf==3.6.1",
    "ptyprocess==0.6.0",
    "pyasn1==0.4.4",
    "pyasn1-modules==0.2.2",
    "Pygments==2.2.0",
    "python-dateutil==2.7.5",
    "pytz==2018.6",
    "PyYAML==3.13",
    "pyzmq==17.1.2",
    "qtconsole==4.4.2",
    "requests==2.20.0",
    "rsa==4.0",
    "scikit-learn==0.20.0",
    "scipy==1.1.0",
    "Send2Trash==1.5.0",
    "six==1.11.0",
    "sklearn==0.0",
    "termcolor==1.1.0",
    "terminado==0.8.1",
    "testpath==0.4.2",
    "tornado==5.1.1",
    "traitlets==4.3.2",
    "urllib3==1.24",
    "wcwidth==0.1.7",
    "webencodings==0.5.1",
]

try:
    PACKAGE_NAME
except NameError:
    PACKAGE_NAME = "rpivision"

setup(
    name=PACKAGE_NAME,
    version=trainers.__version__,
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="Training application package for raspberry-pi-vision",
    package_data={"trainers": ["shapes/data/*.png"]},
=======
#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The setup script.'''

import subprocess
import platform
from setuptools import setup, find_packages, Command
from distutils.command.build import build as _build


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

common_requirements = []

# tensorflow 2.0 wheel has not been released for Raspbian yet
trainer_requirements = ['ansible==2.8.1', 'tensorflow==2.0.0-beta0s']
trainer_requirements = list(map(
    lambda x: x + ';platform_machine=="x86_64"', trainer_requirements
))

rpi_requirements = ['picamera==1.13.0',
                    'tensorflow @ https://github.com/leigh-johnson/tensorflow-community-wheels/blob/master/tensorflow-2.0.0b0-cp35-none-linux_armv7l.whl']
rpi_requirements = list(map(
    lambda x: x + ';platform_machine=="armv7l"', rpi_requirements))

requirements = common_requirements + trainer_requirements + rpi_requirements

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

RPI_LIBS = ['python3-dev', 'cmake']
RPI_CUSTOM_COMMANDS = [['sudo', 'apt-get', 'update'],
                       ['sudo', 'apt-get', 'install', '-y'] + RPI_LIBS
                       ]

TRAINER_DEBIAN_LIBS = ['python3-dev cmake zlib1g-dev']

TRAINER_DEBIAN_CUSTOM_COMMANDS = [['apt-get', 'update'],
                                  ['apt-get', 'install', '-y'] + TRAINER_DEBIAN_LIBS]

TRAINER_DARWIN_LIBS = ['cmake']
TRAINER_DARWIN_CUSTOM_COMMANDS = [['brew', 'update'],
                                  ['brew', 'install'] + TRAINER_DARWIN_LIBS
                                  ]


class CustomCommands(Command):
    '''A setuptools Command class able to run arbitrary commands.'''

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def RunCustomCommand(self, command_list):
        print('Running command: %s' % command_list)
        p = subprocess.Popen(
            command_list,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # Can use communicate(input='y\n'.encode()) if the command run requires
        # some confirmation.
        stdout_data, _ = p.communicate()
        print('Command output: %s' % stdout_data)
        if p.returncode != 0:
            raise RuntimeError(
                'Command %s failed: exit code: %s' % (
                    command_list, p.returncode)
            )

    def run(self):
        system = platform.system()
        machine = platform.machine()
        distro = platform.linux_distribution()

        if 'x86' in machine and system == 'Linux' and 'debian' in distro:
            if 'debian' in distro:
                for command in TRAINER_DEBIAN_CUSTOM_COMMANDS:
                    self.RunCustomCommand(command)
        elif 'arm' in machine and system == 'Linux' and 'debian' in distro:
            for command in TRAINER_DEBIAN_CUSTOM_COMMANDS:
                self.RunCustomCommand(command)
        elif system == 'Darwin':
            for command in TRAINER_DARWIN_CUSTOM_COMMANDS:
                self.RunCustomCommand(command)
        else:
            raise NotImplementedError(
                'Unsupported Platform: {}. Supported platforms are Debian-derived Linux and Darwin (OS X)'.format(system))


setup(
    author='Leigh Johnson',
    author_email='leigh@data-literate.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description='Examples and utilities for getting started with computer vision on a Raspberry Pi usingng Tensorflow',
    install_requires=requirements,
    license='MIT license',
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='rpi_vision',
    name='rpi-vision',
    packages=find_packages(include=['detector']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/leigh-johnson/rpi-vision',
    version='0.1.0',
    zip_safe=False,
>>>>>>> release-v1.0.0
)
