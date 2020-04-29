"""
This script uses the pydrive Python library to upload the data folder as a .zip
file to Google Drive.

Before using this script please make sure that there is a single file named as
`output_file_name_drive` in Google Drive.
"""

import shutil
from pathlib import Path

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from utils import print_blue_bold, print_warning

if __name__ == '__main__':
    folder_path = Path('data')
    output_file_name = 'tmp-data.zip'
    output_file_name_drive = 'data.zip'

    # archive the input folder as a local temporary .zip file
    print_blue_bold(
        f'> Archive {folder_path}/ folder as {output_file_name}...'
    )
    shutil.make_archive(output_file_name[:-4], 'zip', str(folder_path))

    # authentificate with Google
    print_blue_bold('> Authentificate user using Google auth...')
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()

    # get Google Drive object
    drive = GoogleDrive(gauth)

    # find existing output file in Google Drive
    print_blue_bold(f'> Find {output_file_name_drive} file in Google Drive...')

    files_list = drive.ListFile(
        {'q': "'root' in parents and trashed=false"}
    ).GetList()

    for file in files_list:
        if file['title'] == output_file_name_drive:
            output_file_drive = file
            break
    else:
        raise Exception(
            f'The file {output_file_name_drive} doesn\'t exist in Google Drive'
        )

    print(
        f'Found {output_file_name_drive} in Google Drive with id {file["id"]}'
    )
    print_warning(
        'Please make sure this value is up-to-date in dev/get_data.py'
    )

    # upload local temporary .zip file to drive
    print_blue_bold(
        f'> Update {output_file_name_drive} in Google Drive using local '
        f'file {output_file_name}...'
    )
    output_file_drive.SetContentFile(f'{output_file_name}')
    output_file_drive.Upload()

    # remove local temporary .zip file
    print_blue_bold(f'> Remove {output_file_name}...')
    Path(output_file_name).unlink()
