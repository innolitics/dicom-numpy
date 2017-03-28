import numpy as np
import pytest

from .dicom_import import (
    combine_slices,
    validate_slices_form_uniform_grid,
    merge_slice_pixel_arrays,
    DicomImportException
)

# direction cosines
x_cos = (1, 0, 0)
y_cos = (0, 1, 0)
z_cos = (0, 0, 1)
negative_x_cos = (-1, 0, 0)
negative_y_cos = (0, -1, 0)
negative_z_cos = (0, 0, -1)

arbitrary_shape = (10, 11)

class MockSlice:
    '''
    A minimal DICOM dataset representing a dataslice at a particular
    slice location.  The `slice_position` is the coordinate value along the
    remaining unused axis (i.e. the axis perpendicular to the direction
    cosines).
    '''

    def __init__(self, pixel_array, slice_position, row_cosine, column_cosine):
        na, nb = pixel_array.shape

        self.pixel_array = pixel_array

        self.SeriesInstanceUID = 'arbitrary uid'
        self.SOPClassUID = 'arbitrary sopclass uid'
        self.PixelSpacing = [1.0, 1.0]
        self.Rows = na
        self.Columns = nb
        self.Modality = 'MR'

        # assume that the images are centered on the remaining unused axis
        a_component = [-na/2.0*c for c in row_cosine]
        b_component = [-nb/2.0*c for c in column_cosine]
        c_component = [(slice_position if c == 0 and cc == 0 else 0) for c, cc in zip(row_cosine, column_cosine)]
        patient_position = [a + b + c for a, b, c in zip(a_component, b_component, c_component)]

        self.ImagePositionPatient = patient_position

        self.ImageOrientationPatient = list(row_cosine) + list(column_cosine)


@pytest.fixture
def axial_slices():
    return [
        MockSlice(randi(*arbitrary_shape), 0, x_cos, y_cos),
        MockSlice(randi(*arbitrary_shape), 1, x_cos, y_cos),
        MockSlice(randi(*arbitrary_shape), 2, x_cos, y_cos),
        MockSlice(randi(*arbitrary_shape), 3, x_cos, y_cos),
    ]


class TestMergeSlices:
    def test_simple_axial_set(self, axial_slices):
        combined, _ = combine_slices(axial_slices[0:2])

        manually_combined = np.dstack((axial_slices[0].pixel_array.T, axial_slices[1].pixel_array.T))
        assert np.array_equal(combined, manually_combined)


class TestMergeSlicePixelArrays:
    def test_casts_to_float(self, axial_slices):
        '''
        Integer DICOM pixel data should retain its type.
        '''
        assert merge_slice_pixel_arrays(axial_slices).dtype == np.dtype('uint16')


    def test_robust_to_ordering(self, axial_slices):
        '''
        The DICOM slices should be able to be passed in in any order, and they
        should be recombined appropriately.
        '''
        assert np.array_equal(
            merge_slice_pixel_arrays([axial_slices[0], axial_slices[1], axial_slices[2]]),
            merge_slice_pixel_arrays([axial_slices[1], axial_slices[0], axial_slices[2]])
        )

        assert np.array_equal(
            merge_slice_pixel_arrays([axial_slices[0], axial_slices[1], axial_slices[2]]),
            merge_slice_pixel_arrays([axial_slices[2], axial_slices[0], axial_slices[1]])
        )


class TestValidateSlicesFormUniformGrid:
    def test_missing_middle_slice(self, axial_slices):
        '''
        All slices must be present.  Slice position is determined using the
        ImagePositionPatient (0020,0032) tag.
        '''
        with pytest.raises(DicomImportException):
            validate_slices_form_uniform_grid([axial_slices[0], axial_slices[2], axial_slices[3]])

    def test_slices_from_different_series(self, axial_slices):
        '''
        As a sanity check, slices that don't come from the same DICOM series should
        be rejected.
        '''
        axial_slices[2].SeriesInstanceUID += 'Ooops'
        with pytest.raises(DicomImportException):
            validate_slices_form_uniform_grid(axial_slices)

    @pytest.mark.xfail(reason='Not sure how to detect this in DICOM')
    def test_missing_end_slice(self, axial_slices):
        '''
        Ideally, we would detect missing edge slices, however given that we don't
        know any way to determine the number of slices are in a DICOM series, this
        seems impossible.
        '''
        with pytest.raises(DicomImportException):
            validate_slices_form_uniform_grid([axial_slices[0], axial_slices[1], axial_slices[2]])


def randi(*shape):
    return np.random.randint(1000, size=shape, dtype='uint16')
