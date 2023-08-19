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
