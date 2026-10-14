import io
import zipfile
from unittest.mock import patch

from dicom_numpy import zip_archive


def test_zip_transfer_is_bounded_and_reader_receives_complete_bytes():
    payload = b"synthetic DICOM bytes" * 10000
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("folder/", b"")
        archive.writestr("folder/scan.dcm", payload)
    archive_bytes.seek(0)
    read_sizes = []
    streams = []
    original_open = zipfile.ZipFile.open
    original_read = zipfile.ZipExtFile.read

    def observe_open(archive, *args, **kwargs):
        stream = original_open(archive, *args, **kwargs)
        streams.append(stream)
        return stream

    def observe_read(stream, size=-1):
        read_sizes.append(size)
        return original_read(stream, size)

    def read_dataset(stream):
        assert stream.tell() == 0
        assert stream.read() == payload
        return "synthetic dataset"

    with zipfile.ZipFile(archive_bytes) as archive:
        with patch.object(zipfile.ZipFile, "open", observe_open), \
                patch.object(zipfile.ZipExtFile, "read", observe_read), \
                patch.object(zip_archive.pydicom, "read_file", read_dataset):
            assert zip_archive.dicom_datasets_from_zip(archive) == ["synthetic dataset"]
    assert read_sizes and all(0 < size <= 64 * 1024 for size in read_sizes)
    assert len(streams) == 1 and streams[0].closed
