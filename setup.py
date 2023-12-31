# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='eider',
    version='0.0.0',
    description='Empirically Informed Differential emission measure models for EUV Reconstruction',
    long_description=readme,
    author='Girish M. Duvvuri',
    author_email='girish.duvvuri@gmail.com',
    url='https://github.com/gmduvvuri/eider',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
