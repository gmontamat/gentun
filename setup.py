#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
from pkg_resources import parse_requirements


with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()


with open('requirements/prod.txt') as prod_req:
    requirements = [str(ir) for ir in parse_requirements(prod_req)]
with open('requirements/test.txt') as test_req:
    test_requirements = [str(ir) for ir in parse_requirements(test_req)]


setup(
    author="Gustavo Montamat",
    url="https://github.com/gmontamat/gentun",
    author_email='',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    description="Hyperparameter tuning for machine learning models using a distributed genetic algorithm",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    name='gentun',
    keywords='gentun machine-learning parameter-tuning xgboost keras',
    packages=find_packages(include=['gentun', 'gentun.*']),
    test_suite='tests',
    tests_require=test_requirements + requirements,
    version='0.0.1',
    zip_safe=False,
)
