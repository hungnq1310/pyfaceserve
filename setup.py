import os
import re
import pathlib
import subprocess
from distutils.core import setup
from setuptools import find_packages


def get_version():
    """
    Get dynamic version from git tags.
    """
    version_re = re.compile('^Version: (.+)$', re.M)
    d = os.path.dirname(__file__)
    if os.path.isdir(os.path.join(d, '.git')):
        # Get the version using "git describe".
        cmd = 'git describe --tags --match [0-9]*'.split()
        try:
            version = subprocess.check_output(cmd).decode().strip()
        except subprocess.CalledProcessError:
            print('Unable to get version number from git tags')
            exit(1)
        # PEP 386 compatibility
        if '-' in version:
            version = '.post'.join(version.split('-')[:2])
        # Don't declare a version "dirty" merely because a time stamp has changed. 
        # If it is dirty, append a ".dev1" suffix to indicate a development revision after the release.
        with open(os.devnull, 'w') as fd_devnull:
            subprocess.call(['git', 'status'], stdout=fd_devnull, stderr=fd_devnull)
        cmd = 'git diff-index --name-only HEAD'.split()
        try:
            dirty = subprocess.check_output(cmd).decode().strip()
        except subprocess.CalledProcessError:
            print('Unable to get git index status')
            exit(1)
        if dirty != '':
            version += '.dev1'
    else:
        # Extract the version from the PKG-INFO file.
        with open(os.path.join(d, 'PKG-INFO')) as f:
            version = version_re.search(f.read()).group(1)
    return version


def get_readme_description(path: str = os.path.join(os.getcwd(), 'README.md')):
    """
    Get long description from readme file.
    """
    path = pathlib.Path(path).expanduser().resolve()
    with open(path, encoding='utf-8') as file:
        description = file.read()


setup(
    name='faceserve',
    version=get_version(),
    packages=find_packages(),
    license='Copyright (c) 2024 Hieu Pham',
    zip_safe=True,
    description='Face recognition serving APIs',
    long_description=get_readme_description(),
    long_description_content_type='text/markdown',
    author='Hieu Pham',
    author_email='64821726+hieupth@users.noreply.github.com',
    url='https://gitlab.com/hieupth/pyfaceserve',
    keywords=[],
    install_requires=["pyrefactoring"],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3'
    ],
)
