from setuptools import setup, find_packages
from os import path


here = path.abspath(path.dirname(__file__))


# Get the long description from the readme file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='typex',

    # Versions should comply with PEP440
    version='0.0.1',

    description='TypeX machine implementation',
    long_description=long_description,

    url='https://github.com/gregariouspanda/typex',

    author='Julia Marden',
    author_email='julia@marden.org',

    license='MIT',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Education',
        'Topic :: Security :: Cryptography',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],

    keywords='typex rotor machine',

    packages=find_packages(exclude=['tests*']),

    # We have no additional package data /shrug
    include_package_data=False
)
