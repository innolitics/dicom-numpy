"""
A setuptools based setup module.
"""

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dicom_numpy',
    version='0.1.6',
    description='Extract image data into a 3D numpy array from a set of DICOM files.',
    long_description=long_description,
    url='https://github.com/innolitics/dicom-numpy',
    author='Innolitics, LLC',
    author_email='info@innolitics.com',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',

        'Intended Audience :: Developers',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Software Development :: Build Tools',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='dicom numpy',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=[
        'pydicom',
        'numpy',
    ],

    extras_require={
        'dev': ['check-manifest', 'sphinx', 'sphinx-autobuild', 'mock'],
        'test': ['coverage'],
    },

    package_data={},
    data_files=[],
    entry_points={},
)
