from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

from sound import Sound


class SoundDataset(Dataset):
    """Sound dataset"""

    DATA_FOLDER = Path(__file__).resolve().parent.parent / 'data'

    class DatasetFolderNotFound(Exception):
        pass

    def __init__(self, name, transform=None):
        """
        Args:
            name (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = SoundDataset.DATA_FOLDER / name
        if not self.root_dir.exists():
            raise SoundDataset.DatasetFolderNotFound(
                f'No folder called \'{name}\' found in the dataset '
                f'folder{SoundDataset.DATA_FOLDER}'
            )
        self.sounds_list = list(self.root_dir.rglob('*.wav'))
        self.classes = [f.stem for f in self.root_dir.glob('*/') if f.is_dir()]
        self.nb_classes = len(self.classes)
        self.transform = transform

    def __len__(self):
        return len(self.sounds_list)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        sound_path = self.sounds_list[idx]
        label = sound_path.parent.stem  # label is the folder name
        sound = Sound(path=sound_path, process=False)
        sample = {'sound': sound, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample



