import logging

import numpy as np

from .utils import isclose
from .exceptions import DicomImportException


logger = logging.getLogger(__name__)


def combine_slices(slice_datasets):
    '''
    Given a list of pydicom datasets for an image series, stitch them together into a
    three-dimensional numpy array.  Also return a 4x4 affine transformation
    matrix that converts the ijk-pixel-indices into the xyz-coordinates in the
    patient's coordinate system.
    See http://dicom.innolitics.com/ciods/ct-image/image-plane for details.
    This matrix, M, should fulfill:
    [x, y, z, 1].T = M @ [i, j, k, 1].T
    The pixel array will be cast to floating point values.
    This function requires that the datasets fulfill these properties:
    - Are all part of the same series (0020,000E)
    - The set of slices provided has no internal "gaps"; note that missing
      slices on the ends of the dataset are not detected
    - The size of each slice and the spacing between the pixels in the slice
      must be the same (0020,0032)
    '''
    if len(slice_datasets) == 0:
        raise DicomImportException("Must provide at least one DICOM dataset")

    _validate_slices_form_uniform_grid(slice_datasets)

    voxels = _merge_slice_pixel_arrays(slice_datasets)
    transform = _ijk_to_patient_xyz_transform_matrix(slice_datasets)

    return voxels, transform


def _merge_slice_pixel_arrays(slice_datasets):
    first_slice_dataset = slice_datasets[0]
    num_rows = first_slice_dataset.Rows
    num_columns = first_slice_dataset.Columns
    num_slices = len(slice_datasets)

    dtype = first_slice_dataset.pixel_array.dtype
    voxels = np.empty((num_columns, num_rows, num_slices), dtype=dtype)

    sorted_slice_datasets = _sort_by_slice_spacing(slice_datasets)
    for k, dataset in enumerate(sorted_slice_datasets):
        try:
            slope = float(getattr(dataset, 'RescaleSlope'))
            intercept = float(getattr(dataset, 'RescaleIntercept'))
            voxels[:, :, k] = dataset.pixel_array.astype(np.float32).T*slope + intercept
        except AttributeError:
            voxels[:, :, k] = dataset.pixel_array.T

    return voxels


def _ijk_to_patient_xyz_transform_matrix(slice_datasets):
    first_slice_dataset = _sort_by_slice_spacing(slice_datasets)[0]
    image_orientation = first_slice_dataset.ImageOrientationPatient
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)

    row_spacing, column_spacing = first_slice_dataset.PixelSpacing
    slice_spacing = _slice_spacing(slice_datasets)

    transform = np.identity(4, dtype=np.float32)

    transform[:3, 0] = row_cosine*column_spacing
    transform[:3, 1] = column_cosine*row_spacing
    transform[:3, 2] = slice_cosine*slice_spacing

    transform[:3, 3] = first_slice_dataset.ImagePositionPatient

    return transform


def _validate_slices_form_uniform_grid(slice_datasets):
    '''
    Perform various data checks to ensure that the list of slices form a
    evenly-spaced grid of data.
    Some of these checks are probably not required if the data follows the
    DICOM specification, however it seems pertinent to check anyway.
    '''
    invariant_properties = [
        'Modality',
        'SOPClassUID',
        'SeriesInstanceUID',
        'Rows',
        'Columns',
        'ImageOrientationPatient',
        'PixelSpacing',
        'PixelRepresentation',
        'BitsAllocated',
        'BitsStored',
        'HighBit',
    ]

    for property_name in invariant_properties:
        _slice_attribute_equal(slice_datasets, property_name)

    _validate_image_orientation(slice_datasets[0].ImageOrientationPatient)

    slice_positions = _slice_positions(slice_datasets)
    _check_for_missing_slices(slice_positions)


def _validate_image_orientation(image_orientation):
    '''
    Ensure that the image orientation is supported
    - The direction cosines have magnitudes of 1 (just in case)
    - The direction cosines are perpendicular
    - The direction cosines are oriented along the patient coordinate system's axes
    '''
    # TODO: deduplicate this
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)

    if not _almost_zero(np.dot(row_cosine, column_cosine), 1e-4):
        raise DicomImportException("Non-orthogonal direction cosines: {}, {}".format(row_cosine, column_cosine))
    elif not _almost_zero(np.dot(row_cosine, column_cosine), 1e-8):
        logger.warn("Direction cosines aren't quite orthogonal: {}, {}".format(row_cosine, column_cosine))

    if not _almost_one(np.linalg.norm(row_cosine), 1e-4):
        raise DicomImportException("The row direction cosine's magnitude is not 1: {}".format(row_cosine))
    elif not _almost_one(np.linalg.norm(row_cosine), 1e-8):
        logger.warn("The row direction cosine's magnitude is not quite 1: {}".format(row_cosine))

    if not _almost_one(np.linalg.norm(column_cosine), 1e-4):
        raise DicomImportException("The column direction cosine's magnitude is not 1: {}".format(column_cosine))
    elif not _almost_one(np.linalg.norm(column_cosine), 1e-8):
        logger.warn("The column direction cosine's magnitude is not quite 1: {}".format(column_cosine))


def _almost_zero(value, abs_tol):
    return isclose(value, 0.0, abs_tol=abs_tol)


def _almost_one(value, abs_tol):
    return isclose(value, 1.0, abs_tol=abs_tol)


def _extract_cosines(image_orientation):
    row_cosine = np.array(image_orientation[:3])
    column_cosine = np.array(image_orientation[3:])
    slice_cosine = np.cross(row_cosine, column_cosine)
    return row_cosine, column_cosine, slice_cosine


def _slice_attribute_equal(slice_datasets, property_name):
    initial_value = getattr(slice_datasets[0], property_name, None)
    for dataset in slice_datasets[1:]:
        value = getattr(dataset, property_name, None)
        if value != initial_value:
            msg = 'All slices must have the same value for "{}": {} != {}'
            raise DicomImportException(msg.format(property_name, value, initial_value))


def _slice_positions(slice_datasets):
    image_orientation = slice_datasets[0].ImageOrientationPatient
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)
    return [np.dot(slice_cosine, d.ImagePositionPatient) for d in slice_datasets]


def _check_for_missing_slices(slice_positions):
    slice_positions_diffs = np.diff(sorted(slice_positions))
    if not np.allclose(slice_positions_diffs, slice_positions_diffs[0], atol=0, rtol=1e-5):
        # TODO: figure out how we should handle non-even slice spacing
        msg = "The slice spacing is non-uniform. Slice spacings:\n{}"
        logger.warn(msg.format(slice_positions_diffs))

    if not np.allclose(slice_positions_diffs, slice_positions_diffs[0], atol=0, rtol=1e-1):
        raise DicomImportException('It appears there are missing slices')


def _slice_spacing(slice_datasets):
    if len(slice_datasets) > 1:
        slice_positions = _slice_positions(slice_datasets)
        slice_positions_diffs = np.diff(sorted(slice_positions))
        return np.mean(slice_positions_diffs)
    else:
        return 0.0


def _sort_by_slice_spacing(slice_datasets):
    slice_spacing = _slice_positions(slice_datasets)
    return [d for (s, d) in sorted(zip(slice_spacing, slice_datasets))]
