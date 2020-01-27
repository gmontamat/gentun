#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
from pip._internal.req import parse_requirements
from pip._internal.download import PipSession

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

parsed_requirements = parse_requirements(
    'requirements/prod.txt',
    session=PipSession()
)


parsed_test_requirements = parse_requirements(
    'requirements/test.txt',
    session=PipSession()
)


parsed_extra_requirements = parse_requirements(
    'requirements/extras.txt',
    session=PipSession()
)


requirements = [str(ir.req) for ir in parsed_requirements]
test_requirements = [str(tr.req) for tr in parsed_test_requirements]
extra_requirements = [str(tr.req) for tr in parsed_extra_requirements]


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
    tests_require=test_requirements,
    version='0.0.1',
    zip_safe=False,
)
