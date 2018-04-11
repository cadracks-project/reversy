#!/usr/bin/env python
# coding: utf-8

"""setuptools based setup module for reversy

"""

from setuptools import setup
# To use a consistent encoding
import codecs
from os import path

import reversy

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with codecs.open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name=reversy.__name__,
    version=reversy.__version__,
    description=reversy.__description__,
    long_description=long_description,
    url=reversy.__url__,
    download_url=reversy.__download_url__,
    author=reversy.__author__,
    author_email=reversy.__author_email__,
    license=reversy.__license__,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6'],
    keywords='CAD reverse engineering',
    packages=['reversy'],
    install_requires=[],
    extras_require={
        'dev': [],
        'test': ['pytest', 'coverage'],
    },
    package_data={},
    data_files=[],
    entry_points={})
