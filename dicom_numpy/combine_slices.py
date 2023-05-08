import logging
from math import isclose

import numpy as np

from .exceptions import DicomImportException, MissingInstanceNumberException


logger = logging.getLogger(__name__)


def combine_slices(
    datasets,
    rescale=None,
    enforce_slice_spacing=True,
    sort_by_instance=False,
    skip_sorting=False,
    c_order_axes=False,
):
    """
    Given a list of pydicom datasets for an image series, stitch them together into a
    three-dimensional numpy array.  Also calculate a 4x4 affine transformation
    matrix that converts the ijk-pixel-indices into the xyz-coordinates in the
    DICOM patient's coordinate system.

    Returns a two-tuple containing the 3D-ndarray and the affine matrix.

    If `rescale` is set to `None` (the default), then the image array dtype
    will be preserved, unless any of the DICOM images contain either the
    `Rescale Slope
    <https://dicom.innolitics.com/ciods/ct-image/ct-image/00281053>`_ or the
    `Rescale Intercept <https://dicom.innolitics.com/ciods/ct-image/ct-image/00281052>`_
    attributes.  If either of these attributes are present, they will be
    applied to each slice individually.

    If `rescale` is `True` the voxels will be cast to `float32`, if set to
    `False`, the original dtype will be preserved even if DICOM rescaling information is present.

    If `enforce_slice_spacing` is set to `True`, `combine_slices` will raise a
    `DicomImportException` if there are missing slices detected in the
    datasets. If `enforce_slice_spacing` is set to `False`, missing slices will
    be ignored.

    If `sort_by_instance` is set to `False`, `combine_slices` will sort the
    image instances by position along the slice axis in increasing order. This
    is the default for backwards-compatibility reasons. If `True`, the image
    instances will be sorted according to decreasing `InstanceNumber`. If
    images in the series do not have an `InstanceNumber` and `sort_by_instance`
    is `True`, a `MissingInstanceNumberException` will be raised.

    If `skip_sorting` is set to `True`, `combine_slices` will not attempt to
    sort the slices. This can be useful if the volume must be ordered on other
    tags besides slice position or instance number. This overrides any value
    passed to `sort_by_instance`.

    If `c_order_axes` is set to `True`, the returned array will have its axes
    returned in the order of `(k, i, j)` rather than `(j, i, k)`. This is done
    to optimize slice accesses by ensuring that each slice is contiguous in
    memory. By default, this is done by keeping the axes `(j, i, k)` and storing
    the array using Fortran ordering. This can cause issues with some serialization
    libraries that require C-ordering, such as HDF5. In those cases, the axes may
    be reordered such that slices remain contiguous in memory, but the array is
    returned in C-ordering.

    The returned array has the column-major byte-order.

    Datasets produced by reading DICOMDIR files are ignored.

    This function requires that the datasets:

    - Be in same series (have the same
      `Series Instance UID <https://dicom.innolitics.com/ciods/ct-image/general-series/0020000e>`_,
      `Modality <https://dicom.innolitics.com/ciods/ct-image/general-series/00080060>`_,
      and `SOP Class UID <https://dicom.innolitics.com/ciods/ct-image/sop-common/00080016>`_).
    - The binary storage of each slice must be the same (have the same
      `Bits Allocated <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280100>`_ and
      `Pixel Representation <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280103>`_).
    - The image slice must approximately form a grid. This means there can not
      be any missing internal slices (missing slices on the ends of the dataset
      are not detected). This requirement is relaxed if `enforce_slice_spacing` is set to `False`.
    - Each slice must have the same
      `Rows <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280010>`_,
      `Columns <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280011>`_,
      `Samples Per Pixel <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280002>`_,
      `Pixel Spacing <https://dicom.innolitics.com/ciods/ct-image/image-plane/00280030>`_, and
      `Image Orientation (Patient) <https://dicom.innolitics.com/ciods/ct-image/image-plane/00200037>`_
      attribute values.
    - The direction cosines derived from the
      `Image Orientation (Patient) <https://dicom.innolitics.com/ciods/ct-image/image-plane/00200037>`_
      attribute must, within 1e-4, have a magnitude of 1.  The cosines must
      also be approximately perpendicular (their dot-product must be within
      1e-4 of 0).  Warnings are displayed if any of these approximations are
      below 1e-8, however, since we have seen real datasets with values up to
      1e-4, we let them pass.
    - The `Image Position (Patient) <https://dicom.innolitics.com/ciods/ct-image/image-plane/00200032>`_
      values must approximately form a line.

    If any of these conditions are not met, a `dicom_numpy.DicomImportException` is raised.
    """
    slice_datasets = [ds for ds in datasets if not _is_dicomdir(ds)]

    if len(slice_datasets) == 0:
        raise DicomImportException("Must provide at least one image DICOM dataset")

    if skip_sorting:
        sorted_datasets = slice_datasets
    elif sort_by_instance:
        sorted_datasets = sort_by_instance_number(slice_datasets)
    else:
        sorted_datasets = sort_by_slice_position(slice_datasets)

    _validate_slices_form_uniform_grid(sorted_datasets, enforce_slice_spacing=enforce_slice_spacing)

    voxels = _merge_slice_pixel_arrays(sorted_datasets, rescale, c_order_axes=c_order_axes)
    transform = _ijk_to_patient_xyz_transform_matrix(sorted_datasets)

    return voxels, transform


def sort_by_instance_number(slice_datasets):
    """
    Given a list of pydicom Datasets, return the datasets sorted by instance
    number in the image orientation direction.

    This does not require `pixel_array` to be present, and so may be used to
    associate instance Datasets with the voxels returned from `combine_slices`.
    """
    instance_numbers = [getattr(ds, 'InstanceNumber', None) for ds in slice_datasets]
    if any(n is None for n in instance_numbers):
        raise MissingInstanceNumberException

    return [
        d for (s, d) in sorted(
            zip(instance_numbers, slice_datasets),
            key=lambda v: int(v[0]),
            # Stacked in reverse to order in direction of increasing slice axis
            reverse=True
        )
    ]


def sort_by_slice_position(slice_datasets):
    """
    Given a list of pydicom Datasets, return the datasets sorted in the image orientation direction.

    This does not require `pixel_array` to be present, and so may be used to associate instance Datasets
    with the voxels returned from `combine_slices`.
    """
    slice_positions = _slice_positions(slice_datasets)
    return [
        d for (s, d) in sorted(
            zip(slice_positions, slice_datasets),
            key=lambda v: v[0],
        )
    ]


def _is_dicomdir(dataset):
    media_sop_class = getattr(dataset, 'MediaStorageSOPClassUID', None)
    return media_sop_class == '1.2.840.10008.1.3.10'


def _merge_slice_pixel_arrays(sorted_datasets, rescale=None, c_order_axes=False):
    if rescale is None:
        rescale = any(_requires_rescaling(d) for d in sorted_datasets)

    first_dataset = sorted_datasets[0]
    slice_dtype = first_dataset.pixel_array.dtype
    num_slices = len(sorted_datasets)
    voxels_dtype = np.float32 if rescale else slice_dtype

    if c_order_axes:
        slice_shape = first_dataset.pixel_array.shape
        voxels_shape = (num_slices,) + slice_shape
        voxels = np.empty(voxels_shape, dtype=voxels_dtype)
    else:
        slice_shape = first_dataset.pixel_array.T.shape
        voxels_shape = slice_shape + (num_slices,)
        voxels = np.empty(voxels_shape, dtype=voxels_dtype, order='F')

    for k, dataset in enumerate(sorted_datasets):
        pixel_array = dataset.pixel_array if c_order_axes else dataset.pixel_array.T
        if rescale:
            slope = float(getattr(dataset, 'RescaleSlope', 1))
            intercept = float(getattr(dataset, 'RescaleIntercept', 0))
            pixel_array = pixel_array.astype(np.float32) * slope + intercept
        if c_order_axes:
            voxels[k, ...] = pixel_array
        else:
            voxels[..., k] = pixel_array

    return voxels


def _requires_rescaling(dataset):
    return hasattr(dataset, 'RescaleSlope') or hasattr(dataset, 'RescaleIntercept')


def _ijk_to_patient_xyz_transform_matrix(sorted_datasets):
    first_dataset = sorted_datasets[0]
    image_orientation = first_dataset.ImageOrientationPatient
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)

    row_spacing, column_spacing = first_dataset.PixelSpacing
    slice_spacing = _slice_spacing(sorted_datasets)

    transform = np.identity(4, dtype=np.float32)

    transform[:3, 0] = row_cosine * column_spacing
    transform[:3, 1] = column_cosine * row_spacing
    transform[:3, 2] = slice_cosine * slice_spacing

    transform[:3, 3] = first_dataset.ImagePositionPatient

    return transform


def _validate_slices_form_uniform_grid(sorted_datasets, enforce_slice_spacing=True):
    """
    Perform various data checks to ensure that the list of slices form a
    evenly-spaced grid of data. Optionally, this can be slightly relaxed to
    allow for missing slices in the volume.

    Some of these checks are probably not required if the data follows the
    DICOM specification, however it seems pertinent to check anyway.
    """
    invariant_properties = [
        'Modality',
        'SOPClassUID',
        'SeriesInstanceUID',
        'Rows',
        'Columns',
        'SamplesPerPixel',
        'PixelSpacing',
        'PixelRepresentation',
        'BitsAllocated',
    ]

    for property_name in invariant_properties:
        _slice_attribute_equal(sorted_datasets, property_name)

    _validate_image_orientation(sorted_datasets[0].ImageOrientationPatient)
    _slice_ndarray_attribute_almost_equal(sorted_datasets, 'ImageOrientationPatient', 1e-5)

    if enforce_slice_spacing:
        slice_positions = _slice_positions(sorted_datasets)
        _check_for_missing_slices(slice_positions)


def _validate_image_orientation(image_orientation):
    """
    Ensure that the image orientation is supported
    - The direction cosines have magnitudes of 1 (just in case)
    - The direction cosines are perpendicular
    """
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)

    if not _almost_zero(np.dot(row_cosine, column_cosine), 1e-4):
        raise DicomImportException(f"Non-orthogonal direction cosines: {row_cosine}, {column_cosine}")
    elif not _almost_zero(np.dot(row_cosine, column_cosine), 1e-8):
        logger.warning(f"Direction cosines aren't quite orthogonal: {row_cosine}, {column_cosine}")

    if not _almost_one(np.linalg.norm(row_cosine), 1e-4):
        raise DicomImportException(f"The row direction cosine's magnitude is not 1: {row_cosine}")
    elif not _almost_one(np.linalg.norm(row_cosine), 1e-8):
        logger.warning(f"The row direction cosine's magnitude is not quite 1: {row_cosine}")

    if not _almost_one(np.linalg.norm(column_cosine), 1e-4):
        raise DicomImportException(f"The column direction cosine's magnitude is not 1: {column_cosine}")
    elif not _almost_one(np.linalg.norm(column_cosine), 1e-8):
        logger.warning(f"The column direction cosine's magnitude is not quite 1: {column_cosine}")


def _almost_zero(value, abs_tol):
    return isclose(value, 0.0, abs_tol=abs_tol)


def _almost_one(value, abs_tol):
    return isclose(value, 1.0, abs_tol=abs_tol)


def _extract_cosines(image_orientation):
    row_cosine = np.array(image_orientation[:3])
    column_cosine = np.array(image_orientation[3:])
    slice_cosine = np.cross(row_cosine, column_cosine)
    return row_cosine, column_cosine, slice_cosine


def _slice_attribute_equal(sorted_datasets, property_name):
    initial_value = getattr(sorted_datasets[0], property_name, None)
    for dataset in sorted_datasets[1:]:
        value = getattr(dataset, property_name, None)
        if value != initial_value:
            msg = f'All slices must have the same value for "{property_name}": {value} != {initial_value}'
            raise DicomImportException(msg)


def _slice_ndarray_attribute_almost_equal(sorted_datasets, property_name, abs_tol):
    initial_value = getattr(sorted_datasets[0], property_name, None)
    for dataset in sorted_datasets[1:]:
        value = getattr(dataset, property_name, None)
        if not np.allclose(value, initial_value, atol=abs_tol):
            msg = (f'All slices must have the same value for "{property_name}" within "{abs_tol}": {value} != '
                   f'{initial_value}')
            raise DicomImportException(msg)


def _slice_positions(sorted_datasets):
    image_orientation = sorted_datasets[0].ImageOrientationPatient
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)
    return [np.dot(slice_cosine, d.ImagePositionPatient) for d in sorted_datasets]


def _check_for_missing_slices(slice_positions):
    if len(slice_positions) > 1:
        slice_positions_diffs = np.diff(sorted(slice_positions))
        if not np.allclose(slice_positions_diffs, slice_positions_diffs[0], atol=0, rtol=1e-5):
            # TODO: figure out how we should handle non-even slice spacing
            msg = f"The slice spacing is non-uniform. Slice spacings:\n{slice_positions_diffs}"
            logger.warning(msg)

        if not np.allclose(slice_positions_diffs, slice_positions_diffs[0], atol=0, rtol=1e-1):
            raise DicomImportException('It appears there are missing slices')


def _slice_spacing(sorted_datasets):
    if len(sorted_datasets) > 1:
        slice_positions = _slice_positions(sorted_datasets)
        slice_positions_diffs = np.diff(slice_positions)
        return np.median(slice_positions_diffs)

    return getattr(sorted_datasets[0], 'SpacingBetweenSlices', 0)
