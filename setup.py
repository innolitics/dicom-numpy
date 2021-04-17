"""
A setuptools based setup module.
"""

from setuptools import setup, find_packages
from codecs import open
from os import path

from dicom_numpy import __version__

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dicom_numpy',
    version=__version__,
    description='Extract image data into a 3D numpy array from a set of DICOM files.',
    long_description=long_description,
    url='https://github.com/innolitics/dicom-numpy',
    author='Innolitics, LLC',
    author_email='info@innolitics.com',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Developers',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Software Development :: Build Tools',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    keywords='dicom numpy',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=[
        'pydicom',
        'numpy',
    ],

    python_requires='>= 3.6',

    extras_require={
        'dev': ['check-manifest', 'sphinx', 'sphinx-autobuild', 'mock'],
        'test': ['coverage', 'pytest'],
    },

    package_data={},
    data_files=[],
    entry_points={},
)
