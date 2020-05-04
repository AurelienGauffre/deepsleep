import os
from pathlib import Path

import librosa  # Librosa for audio
import librosa.display  # and the display module for visualization
import matplotlib.pyplot as plt
import numpy as np


class Sound:
    """
    Basic class to deal with sounds.
    """

    class SpectrogramNotComputed(Exception):
        pass

    def __init__(
        self,
        path: Path = None,
        sampling_rate: int = None,
        process: bool = True,
    ):
        self.path = path
        self.folder, self.filename = os.path.split(self.path)
        self.filebase, self.ext = os.path.splitext(self.filename)
        self.y, self.sampling_rate = librosa.load(self.path, sr=sampling_rate)
        self.spectrogram = None

        if process:
            self.spectrogram = self.compute_spectrogram()

    def compute_spectrogram(self, n_mels: int = 256):
        """
        Args:
            n_mels: vertical resolution of the spectrogram
        Returns:
            Melspectrogram on a log scale
        """
        return librosa.power_to_db(
            librosa.feature.melspectrogram(
                self.y, sr=self.sampling_rate, n_mels=n_mels
            ),
            ref=np.max,
        )  # convert to log scale (dB). Use the peak power (max) as ref

    def plot_spectrogram(self, ax=None):
        if self.spectrogram is None:
            raise Sound.SpectrogramNotComputed(
                'please call compute_spectrogram() to compute the '
                'sound spectrogram before trying to plot it'
            )
        ax = plt.gca() if ax is None else ax
        librosa.display.specshow(
            self.spectrogram,
            sr=self.sampling_rate,
            x_axis='time',
            y_axis='mel',
            ax=ax,
        )
        ax.set_title('mel power spectrogram')

    def to_samples(
        self,
        sample_length: float = 4,
        mode: str = 'duplicate',
        output_folder: str = None,
        suffix: str = 'sample',
    ):
        """
        Take the sound in self.y and split in into samples of size sample_size.

        Args:
            mode: Set to one of 3 values : 'drop','duplicate','keep' to select
            how to deal with last sample. Drop will drop the sample,
            keep will keep a sample from an unknown size, and 'duplicate'
            will add a small part from the previous sample to make the last
            one of equal size.
            sample_length: length of each sample in seconds.
            output_folder: Folder where samples are saved.
            suffix: String add at the end of the files.
        """

        output_folder = self.folder if output_folder is None else output_folder
        step_size = int(sample_length * self.sampling_rate)
        i = 0
        n = len(self.y)
        for i in range(n // step_size):
            librosa.output.write_wav(
                os.path.join(
                    output_folder, f'{self.filebase}_{suffix}_{i}.wav'
                ),
                self.y[i * step_size : (i + 1) * step_size],
                sr=self.sampling_rate,
                norm=False,
            )
        if n % step_size != 0:  # if the last sample is smaller than the others
            i = n // step_size
            if mode == 'drop':
                pass
            elif mode == 'duplicate':
                librosa.output.write_wav(
                    os.path.join(
                        output_folder, f'{self.filebase}_{suffix}_{i}.wav'
                    ),
                    self.y[n - step_size : n],
                    sr=self.sampling_rate,
                    norm=False,
                )
            elif mode == 'keep':
                librosa.output.write_wav(
                    os.path.join(
                        output_folder, f'{self.filebase}_{suffix}_{i}.wav'
                    ),
                    self.y[i * step_size : n],
                    sr=self.sampling_rate,
                    norm=False,
                )


if __name__ == '__main__':
    for file in os.listdir('../../Talk'):
        print(file)
        s1 = Sound(os.path.join('../../Talk/', file))
        # s1.to_samples(sample_length=4, mode='duplicate')
        s1.plot_spectrogram()
        plt.show()
