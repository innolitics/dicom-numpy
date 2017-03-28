.. image:: https://travis-ci.org/innolitics/dicom-numpy.svg?branch=master

===========
DICOM Numpy
===========

Overview
--------

This python module provides utilities for extracting image data contained in DICOM files into Numpy ndarrays.

Dependencies
------------

- Python 2.7 or Python 3.3+
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
            # the DICOM files are not from the same series, there are missing
            # internal slices, or the data is inconsistent
            pass
        return voxel_ndarray
