from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

from sound import Sound


class SoundDataset(Dataset):
    """Sound dataset"""

    DATA_FOLDER = Path(__file__).resolve().parent.parent / 'data'

    class DatasetFolderNotFound(Exception):
        pass

    def __init__(self, dataset_name, transform, data_nature='2D'):
        """
        Args:
            dataset_name (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_name = dataset_name
        self.root_dir = SoundDataset.DATA_FOLDER / dataset_name
        if not self.root_dir.exists():
            raise SoundDataset.DatasetFolderNotFound(
                f'No folder called \'{dataset_name}\' found in the dataset '
                f'folder{SoundDataset.DATA_FOLDER}'
            )
        self.sounds_list = list(self.root_dir.rglob('*.wav'))
        self.labels = [f.stem for f in self.root_dir.glob('*/') if f.is_dir()]
        self.nb_labels = len(self.labels)
        self.label_to_num = {label: i for i, label in enumerate(self.labels)}
        self.data_nature = data_nature
        # TODO: Define a proper init_transforms
        self.transform = transforms.Compose([ToTensor(self.data_nature)])

    def __len__(self):
        return len(self.sounds_list)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        sound_path = self.sounds_list[idx]
        label = sound_path.parent.stem  # label is the folder name
        sound = Sound(
            path=sound_path,
            process=False if self.data_nature == '1D' else True,
        )
        sample = {'sound': sound, 'label': self.label_to_num[label]}

        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays to Tensors. This class is used as a transform in the
    data loading` """

    class ModeNotRecognised(Exception):
        pass

    def __init__(self, data_nature):
        self.data_nature = data_nature

    def __call__(self, sample):

        sound, label = sample['sound'], sample['label']
        if self.data_nature == '1D':
            return {
                'sound': torch.from_numpy(sound.y).unsqueeze(0),
                'label': label,
            }
        # conv1D expect sample following the shape [bs,nb_channel,lenght]
        elif self.data_nature == '2D':
            return {
                'sound': torch.from_numpy(sound.spectrogram),
                'label': label,
            }
        else:
            raise ToTensor.ModeNotRecognised(
                "Enter a valid value for " "data_nature."
            )
