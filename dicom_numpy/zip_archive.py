import zipfile
import logging
import tempfile

try:
    import pydicom as dicom
except ImportError:
    import dicom

from .exceptions import DicomImportException
from .combine_slices import combine_slices


logger = logging.getLogger(__name__)


def combined_series_from_zip(zip_filename):
    logger.info('Extracting voxel data from "{}"'.format(zip_filename))

    if not zipfile.is_zipfile(zip_filename):
        raise DicomImportException('Invalid zipfile {}'.format(zip_filename))

    with zipfile.ZipFile(zip_filename, 'r') as zip_file:
        datasets = dicom_datasets_from_zip(zip_file)

    voxels, ijk_to_xyz = combine_slices(datasets)
    return voxels, ijk_to_xyz


def dicom_datasets_from_zip(zip_file):
    datasets = []
    for entry in zip_file.namelist():
        if entry.endswith('/'):
            continue  # skip directories

        entry_pseudo_file = zip_file.open(entry)

        # the pseudo file does not support `seek`, which is required by
        # dicom's lazy loading mechanism; use temporary files to get around this;
        # relies on the temporary files not being removed until the temp
        # file is garbage collected, which should be the case because the
        # dicom datasets should retain a reference to the temp file
        temp_file = tempfile.TemporaryFile()
        temp_file.write(entry_pseudo_file.read())
        temp_file.flush()
        temp_file.seek(0)

        try:
            dataset = dicom.read_file(temp_file)
            datasets.append(dataset)
        except dicom.errors.InvalidDicomError as e:
            msg = 'Skipping invalid DICOM file "{}": {}'
            logger.info(msg.format(entry, e))

    if len(datasets) == 0:
        raise DicomImportException('Zipfile does not contain any valid DICOM files')

    return datasets
