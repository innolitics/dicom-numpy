import pytest
import numpy as np


# direction cosines
x_cos = (1, 0, 0)
y_cos = (0, 1, 0)
z_cos = (0, 0, 1)
negative_x_cos = (-1, 0, 0)
negative_y_cos = (0, -1, 0)
negative_z_cos = (0, 0, -1)

arbitrary_shape = (10, 11)
arbitrary_rgb_shape = (10, 11, 3)


class MockSlice:
    """
    A minimal DICOM dataset representing a dataslice at a particular
    slice location.  The `slice_position` is the coordinate value along the
    remaining unused axis (i.e. the axis perpendicular to the direction
    cosines).
    """

    def __init__(self, pixel_array, slice_position, row_cosine=None, column_cosine=None, **kwargs):
        if row_cosine is None:
            row_cosine = x_cos

        if column_cosine is None:
            column_cosine = y_cos

        shape = pixel_array.shape
        if len(shape) == 2:
            na, nb = shape
            SamplesPerPixel = 1
        else:
            na, nb, SamplesPerPixel = shape

        self.pixel_array = pixel_array

        self.SeriesInstanceUID = 'arbitrary uid'
        self.SOPClassUID = 'arbitrary sopclass uid'
        self.PixelSpacing = [1.0, 1.0]
        self.Columns = na
        self.Rows = nb
        self.SamplesPerPixel = SamplesPerPixel
        self.Modality = 'MR'

        # assume that the images are centered on the remaining unused axis
        a_component = [-na/2.0*c for c in row_cosine]
        b_component = [-nb/2.0*c for c in column_cosine]
        c_component = [(slice_position if c == 0 and cc == 0 else 0) for c, cc in zip(row_cosine, column_cosine)]
        patient_position = [a + b + c for a, b, c in zip(a_component, b_component, c_component)]

        self.ImagePositionPatient = patient_position

        self.ImageOrientationPatient = list(row_cosine) + list(column_cosine)

        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def axial_slices():
    return [
        MockSlice(randi(*arbitrary_shape), 0),
        MockSlice(randi(*arbitrary_shape), 1),
        MockSlice(randi(*arbitrary_shape), 2),
        MockSlice(randi(*arbitrary_shape), 3),
    ]


@pytest.fixture
def axial_rgb_slices():
    # SamplesPerPixel = 3
    return [
        MockSlice(randi(*arbitrary_rgb_shape), 0),
        MockSlice(randi(*arbitrary_rgb_shape), 1),
        MockSlice(randi(*arbitrary_rgb_shape), 2),
        MockSlice(randi(*arbitrary_rgb_shape), 3),
    ]


def randi(*shape):
    return np.random.randint(1000, size=shape, dtype='uint16')
