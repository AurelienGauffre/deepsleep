# DeepSleep
Sleep analysis using deep learning.

### Get the datasets
Run the following command to populate your `data/` folder with the datasets:
```shell
# fetch the datasets from Google Drive
python dev/get_data.py
```

:warning: Please contact the repository maintainers if you get the following 
error, they will know what to do:
```
Permission denied: https://drive.google.com/uc?id=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Maybe you need to change permission over 'Anyone with the link'?
```

### Update the datasets (maintainers only)
To update the datasets, you need to upload your local `data/` folder to Google 
Drive as a `.zip` (reserved to maintainers):
```shell
# upload the local data/ folder to Google Drive
python dev/upload_data_to_drive.py
```
