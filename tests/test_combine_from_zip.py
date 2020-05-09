import os

import numpy as np

from dicom_numpy.zip_archive import combined_series_from_zip

TEST_DIR = os.path.dirname(__file__)
TEST_DICOM_ZIP_PATH = os.path.join(TEST_DIR, 'test_dicom.zip')
GOLDEN_FILE_PATH = os.path.join(TEST_DIR, 'golden_values.npz')


def test_combine_from_zip():
    """
    An integration test checking that a known DICOM zip archive can be
    processed and produces a known golden value.
    """
    voxels, ijk_to_xyz = combined_series_from_zip(TEST_DICOM_ZIP_PATH)
    with np.load(GOLDEN_FILE_PATH) as dataset:
        np.testing.assert_array_equal(voxels, dataset['voxels'])
        np.testing.assert_array_equal(ijk_to_xyz, dataset['ijk_to_xyz'])
