"""
This script uses the gdown Python library to download the data folder from
Google Drive.
"""

from pathlib import Path
from zipfile import ZipFile

import gdown
from colorama import deinit, init

from utils import print_blue_bold

if __name__ == '__main__':
    drive_file_id = '1vU4S7q4b_w29SS3IMhTD8wK4TP80MiuS'
    url = f'https://drive.google.com/uc?id={drive_file_id}'
    output_path = Path('data/data.zip')

    # initialise terminal colors
    init()

    # download file from Google Drive
    print_blue_bold(f'> Download file from Google Drive to {output_path}...')
    gdown.download(url, str(output_path), quiet=False)

    # unzip and inflate .zip file
    print_blue_bold(
        f'> Unzip and inflate {output_path} in {output_path.parent}/...'
    )
    with ZipFile(str(output_path), 'r') as zip_file:
        zip_file.extractall(output_path.parent)

    # remove downloaded .zip file
    print_blue_bold(f'> Remove {output_path}...')
    output_path.unlink()

    # de-initialise terminal colors
    deinit()
