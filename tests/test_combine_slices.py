import numpy as np
import pytest

from dicom_numpy.combine_slices import (
    combine_slices,
    _merge_slice_pixel_arrays,
)
from dicom_numpy.exceptions import DicomImportException
from .conftest import MockSlice


class TestCombineSlices:
    def test_simple_axial_set(self, axial_slices):
        combined, _ = combine_slices(axial_slices[0:2])
        manually_combined = np.dstack((axial_slices[0].pixel_array.T, axial_slices[1].pixel_array.T))
        assert np.array_equal(combined, manually_combined)

    def test_simple_axial_set_w_dicomdir(self, axial_slices):
        dicomdir_dataset = axial_slices[2]
        dicomdir_dataset.MediaStorageSOPClassUID = '1.2.840.10008.1.3.10'
        datasets = [dicomdir_dataset, axial_slices[0], axial_slices[1]]
        combined, _ = combine_slices(datasets)
        manually_combined = np.dstack((axial_slices[0].pixel_array.T, axial_slices[1].pixel_array.T))
        assert np.array_equal(combined, manually_combined)

    def test_single_slice(self, axial_slices):
        dataset = axial_slices[-1]
        array, _ = combine_slices([dataset])
        assert np.array_equal(array, dataset.pixel_array.T[:, :, None])

    def test_single_slice_spacing(self, axial_slices):
        slice_spacing = 0.65
        dataset = axial_slices[0]
        dataset.SpacingBetweenSlices = slice_spacing
        array, affine = combine_slices([dataset])
        assert np.array_equal(array, dataset.pixel_array.T[:, :, None])
        assert np.isclose(np.linalg.norm(affine[:, 2]), np.abs(slice_spacing))

    def test_rgb_axial_set(self, axial_rgb_slices):
        combined, _ = combine_slices(axial_rgb_slices)

        manually_combined = np.stack([ds.pixel_array for ds in axial_rgb_slices], axis=0).T
        assert np.array_equal(combined, manually_combined)


class TestMergeSlicePixelArrays:
    def test_casting_if_only_rescale_slope(self):
        """
        If the `RescaleSlope` DICOM attribute is present, the
        `RescaleIntercept` attribute should also be present, however, we handle
        this case anyway.
        """
        slices = [
            MockSlice(np.ones((10, 20), dtype=np.uint8), 0, RescaleSlope=2),
            MockSlice(np.ones((10, 20), dtype=np.uint8), 1, RescaleSlope=2),
        ]

        voxels = _merge_slice_pixel_arrays(slices)
        assert voxels.dtype == np.dtype('float32')
        assert voxels[0, 0, 0] == 2.0

    def test_casting_rescale_slope_and_intercept(self):
        """
        Some DICOM modules contain the `RescaleSlope` and `RescaleIntercept` DICOM attributes.
        """
        slices = [
            MockSlice(np.ones((10, 20), dtype=np.uint8), 0, RescaleSlope=2, RescaleIntercept=3),
            MockSlice(np.ones((10, 20), dtype=np.uint8), 1, RescaleSlope=2, RescaleIntercept=3),
        ]

        voxels = _merge_slice_pixel_arrays(slices)
        assert voxels.dtype == np.dtype('float32')
        assert voxels[0, 0, 0] == 5.0

    def test_robust_to_ordering(self, axial_slices):
        """
        The DICOM slices should be able to be passed in in any order, and they
        should be recombined appropriately.
        """
        assert np.array_equal(
            _merge_slice_pixel_arrays([axial_slices[0], axial_slices[1], axial_slices[2]]),
            _merge_slice_pixel_arrays([axial_slices[1], axial_slices[0], axial_slices[2]])
        )

        assert np.array_equal(
            _merge_slice_pixel_arrays([axial_slices[0], axial_slices[1], axial_slices[2]]),
            _merge_slice_pixel_arrays([axial_slices[2], axial_slices[0], axial_slices[1]])
        )

    def test_rescales_if_forced_true(self):
        slice_datasets = [MockSlice(np.ones((10, 20), dtype=np.uint8), 0)]
        voxels = _merge_slice_pixel_arrays(slice_datasets, rescale=True)
        assert voxels.dtype == np.float32

    def test_no_rescale_if_forced_false(self):
        slice_datasets = [MockSlice(np.ones((10, 20), dtype=np.uint8), 0, RescaleSlope=2, RescaleIntercept=3)]
        voxels = _merge_slice_pixel_arrays(slice_datasets, rescale=False)
        assert voxels.dtype == np.uint8


class TestValidateSlicesFormUniformGrid:
    def test_missing_middle_slice_strict(self, axial_slices):
        """
        By default, all slices must be present. Slice position is determined
        using the ImagePositionPatient (0020,0032) tag.
        """
        with pytest.raises(DicomImportException):
            combine_slices([axial_slices[0], axial_slices[2], axial_slices[3]])

    def test_missing_middle_slice_lax(self, axial_slices):
        """
        if `enforce_slice_spacing` is set to False, the no missing slices
        constraint is relaxed. In this case, slices are stacked together as if
        there were no missing slices.
        """
        voxels, _transform = combine_slices(
            [axial_slices[0], axial_slices[2], axial_slices[3]],
            enforce_slice_spacing=False,
        )
        assert voxels.shape[2] == 3

    def test_insignificant_difference_in_direction_cosines(self, axial_slices):
        """
        We have seen DICOM series in the field where slices have lightly
        different direction cosines.
        """
        axial_slices[0].ImageOrientationPatient[0] += 1e-6
        combine_slices(axial_slices)

    def test_significant_difference_in_direction_cosines(self, axial_slices):
        axial_slices[0].ImageOrientationPatient[0] += 1e-4
        with pytest.raises(DicomImportException):
            combine_slices(axial_slices, enforce_slice_spacing=False)

    def test_slices_from_different_series(self, axial_slices):
        """
        As a sanity check, slices that don't come from the same DICOM series should
        be rejected.
        """
        axial_slices[2].SeriesInstanceUID += 'Ooops'
        with pytest.raises(DicomImportException):
            combine_slices(axial_slices, enforce_slice_spacing=False)

    @pytest.mark.xfail(reason='Not sure how to detect this in DICOM')
    def test_missing_end_slice(self, axial_slices):
        """
        Ideally, we would detect missing edge slices, however given that we don't
        know any way to determine the number of slices are in a DICOM series, this
        seems impossible.
        """
        with pytest.raises(DicomImportException):
            combine_slices(
                [axial_slices[0], axial_slices[1], axial_slices[2]],
                enforce_slice_spacing=False,
            )
