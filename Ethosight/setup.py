# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup, find_packages

setup(
    name='Ethosight',
    version='0.1.0',
    description='A package for media analysis with Ethosight',
    author='Hugo Latapie',
    author_email='hlatapie@cisco.com',
    packages=find_packages(),
    install_requires=[
        # list of dependencies, e.g. 'numpy>=1.18.0'
    ],
    entry_points={
        'console_scripts': [
            'EthosightCLI = Ethosight.EthosightCLI:cli',
            'EthosightMediaAnalyzerCLI = Ethosight.EthosightMediaAnalyzerCLI:cli',
            'EthosightDatasetCLI = Ethosight.EthosightDatasetCLI:cli',
            'EthosightAppCLI = Ethosight.EthosightAppCLI:cli',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',],
    keywords='media analysis, ethosight',  # keywords for your project
)
