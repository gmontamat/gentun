"""
Module setup for the 'gentun' package.

See:
https://github.com/gmontamat/gentun
"""

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    name='gentun',
    version='0.0.1',
    description='Hyperparameter tuning for machine learning models using a distributed genetic algorithm',
    long_description=long_description,
    url='https://github.com/gmontamat/gentun',
    author='Gustavo Montamat',
    # author_email='',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='gentun machine-learning parameter-tuning xgboost',
    packages=find_packages(exclude=['tests']),
    install_requires=['pandas', 'pika'],
    extras_require={
        'xgboost': ['xgboost'],
        'full': ['xgboost'],
    },
)
