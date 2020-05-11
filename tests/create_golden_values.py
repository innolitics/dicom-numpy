"""
Generate a golden NPZ file from a dicom ZIP archive.
"""
import argparse

import numpy as np

from dicom_numpy.zip_archive import combined_series_from_zip


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help='Output golden NPZ file', required=False)
    parser.add_argument('input', help="Input DICOM zip archive")
    return parser.parse_args()


def generate_golden_values(input_zip, output_path='golden_values'):
    """
    Generate a golden NPZ file for a given DICOM zip archive.
    """
    voxels, ijk_to_xyz = combined_series_from_zip(input_zip)
    np.savez_compressed(output_path, voxels=voxels, ijk_to_xyz=ijk_to_xyz)


if __name__ == '__main__':
    args = parse_args()
    if args.output:
        generate_golden_values(args.input, args.output)
    else:
        generate_golden_values(args.input)
