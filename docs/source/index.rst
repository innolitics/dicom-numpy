***********
DICOM-Numpy
***********

This python module provides a set of utilities for extracting data contained in
DICOM files into Numpy ndarrays.  It is a higher-level library that builds on the excellent lower-level `pydicom
<http://pydicom.readthedocs.io/en/stable/>`_ library.

The library is quite small at the moment, however, if you have a DICOM-related
utility function that you think would be appropriate to include, create a
Github Issue!

Dependencies
============

- Python 2.7 or Python 3.5+
- Numpy
- PyDicom


Installation
============

.. code:: bash

    pip install dicom_numpy


Source Code
===========

The source code is hosted on `Github <https://github.com/innolitics/dicom-numpy>`_.


Combine DICOM Slices
====================

The DICOM standard stores MR, CT, and PET scans as a series of images  saved in
a separate files.  A common task is to combine all of the images that make up a
single 3D image into a single scan.

The function that performs this task is `combine_slices`.  Since this library
builds on pydicom, `combine_slices` takes an list of `pydicom
datasets <http://pydicom.readthedocs.io/en/stable/pydicom_user_guide.html#dataset>`_.

Example
-------

.. code:: python

    import dicom
    import dicom_numpy

    def extract_voxel_data(list_of_dicom_files):
        datasets = [dicom.read_file(f) for f in list_of_dicom_files]
        try:
            voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(datasets)
        except dicom_numpy.DicomImportException as e:
            # invalid DICOM data
            raise
        return voxel_ndarray


Details
-------

.. autofunction:: dicom_numpy.combine_slices


Change Log
==========

Version 0.4.0
-------------
- Ignore DICOMDIR files
- Fix bug that was triggered when using `from dicom_numpy import *`
- Make `combine_slices` work with a single slice
- Add support for "channeled slices" (e.g., RGB slices)
- Allow HighBit and BitsStored DICOM attributes to be non-uniform
- Drop support for Python 3.4; test Python 3.7
- Require the SamplesPerPixel DICOM attribute to be invariant among the slices

Version 0.3.0
-------------

- Reverted slice ordering change from v0.2.0, since the DICOM standard defines
  the Z-axis direction to be increasing in the direction of the head.
- Added support for both PyDicom 0.X and 1.X

Version 0.2.0
-------------

- Changed the behavior of `combine_slices` to stack slices from head (slice 0)
  to foot (slice -1). Note that this is the reverse of the behavior in v0.1.*.

Version 0.1.5
-------------

- Added the `rescale` option to `combine_slices`
- Made `combine_slices`'s returned ndarray use column-major ordering

Contributing
============

Process
-------

Contributions are welcome.  Please create a Github issue describing the change
you would like to make so that you can discuss your approach with the
maintainers.  Assuming the maintainers like your approach, then create a pull
request.

Tests
-----

Most new functionality will require unit tests.

Run all of the tests for each supported python version using:

.. code:: bash

    tox

Run all of the tests for the currently active python version using:

.. code:: bash

    pytest

Other Contributors
------------------

Additional contributions made by:

- Jonathan Daniel

Thank you!


About Innolitics
================

Innolitics is a team of talented software developers with medical and
engineering backgrounds.  We help companies produce top quality medical imaging
and workflow applications.  If you work with DICOM frequently, our `DICOM
Standard Browser <https://dicom.innolitics.com>`_ may be useful to you.

If you could use help with DICOM, `let us know <http://innolitics.com/#contact>`_!  We offer training sessions and
can provide advice or development services.


Licenses
========

.. include:: ../../LICENSE.txt

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
