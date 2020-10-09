# deepsleep :zzz:
Sleep analysis from sound recording using Pytorch. 
First aim of this project is to build and tune general sound classifiers to detect automatically somniloquy.
The current most effective implemented approach is based on CNN applied directly to the sound spectrogram images.

### Get the datasets
Run the following command to populate your `data/` folder with the datasets:
```shell
# fetch the datasets from Google Drive
poetry run python dev/get_data.py
```

### Update the datasets
The datasets from your local `data/` folder will be automatically updated to Google 
Drive as a `.zip` (reserved to maintainers):
```shell
# upload the local data/ folder to Google Drive
poetry run python dev/upload_data_to_drive.py
```

## Contributing
If you wish to contribute to this project please head to [CONTRIBUTING.md
](https://github.com/AurelienGauffre/deepsleep/blob/master/CONTRIBUTING.md).
