#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

console_scripts = ['counterfactualms-train=counterfactualms.experiments.medical.trainer:main',
                   'counterfactualms-test=counterfactualms.experiments.medical.tester:main']

with open('README.md', encoding="utf-8") as f:
    readme = f.read()

with open('LICENSE.md', encoding="utf-8") as f:
    license = f.read()

args = dict(
    name='counterfactualms',
    version='0.0.1',
    description="ask counterfactuals for MS subjects given images, demographics, and other data",
    long_description=readme,
    license=license,
    packages=find_packages(),
    entry_points = {
        'console_scripts': console_scripts
    },
)

setup(install_requires=['torch', 'torchvision', 'nibabel', 'numpy', 'statsmodels',
                        'tensorboard', 'pytorch-lightning', 'pyro-ppl', 'scikit-image',
                        'scikit-learn', 'seaborn', 'jupyter'], **args)
