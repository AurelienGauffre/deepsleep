"""This script uses the gdown Python library to download the data folder from
Google Drive.
"""

from colorama import Fore, Style, deinit, init
import gdown
from pathlib import Path
from zipfile import ZipFile


def print_blue_bold(message: str) -> str:
    print(f'{Fore.BLUE}{Style.BRIGHT}{message}{Style.RESET_ALL}')


if __name__ == '__main__':
    # initialize terminal colors
    init()

    url = 'https://drive.google.com/uc?id=1AZq1k-kft9sZ9Rj3y4OrCi6nJaKbyh4U'
    output = Path('data/data.zip')

    print_blue_bold(f'> Downloading from Google Drive ({url}) to {output}...')
    gdown.download(url, str(output), quiet=False)

    print_blue_bold(f'\n> Unziping and inflating {output}...')
    with ZipFile(str(output), 'r') as zip_file:
       zip_file.extractall()

    print_blue_bold(f'\n> Removing {output}...')
    output.unlink()

    # de-initialize terminal colors
    deinit()
