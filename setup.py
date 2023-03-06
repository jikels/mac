from distutils.core import setup
from setuptools import find_packages

setup(
    name='mbmrl_torch',
    packages=find_packages(),
    version='0.0.1',
    description='Model-Based Meta RL',
    long_description=open('./README.md').read(),
    author='Joel Ikels',
    zip_safe=True,
    license='MIT'
)