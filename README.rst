.. image:: https://travis-ci.org/innolitics/dicom-numpy.svg?branch=master

===========
DICOM Numpy
===========

Overview
--------

This python module provides utilities for extracting image data contained in DICOM files into Numpy ndarrays.

Dependencies
------------

- Python 2.7 or Python 3.4+
- Numpy
- PyDicom

Installation
------------

.. code:: bash

    pip install dicom_numpy

Basic Usage
-----------

.. code:: python

    import dicom
    import dicom_numpy

    def extract_voxel_data(list_of_dicom_files):
        datasets = [dicom.read_file(f) for f in list_of_dicom_files]
        try:
            voxel_ndarray = dicom_numpy.combine_slices(datasets)
        except dicom_numpy.DicomImportException as e:
            # Either the DICOM files are not from the same series, there are missing
            # internal slices (we can't detect missing end slices), or the data
            # is inconsistent in some way (e.g. the size of each slice is
            # inconsistent).
            raise
        return voxel_ndarray
