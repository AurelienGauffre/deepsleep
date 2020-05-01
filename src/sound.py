import os

import matplotlib.pyplot as plt
import numpy as np

import librosa  # Librosa for audio
import librosa.display  # and the display module for visualization


class Sound:
    """
    Basic class to deal with sounds
    """

    def __init__(self, path=None, sampling_rate=None, process=True):
        self.path = path
        self.folder, self.filename = os.path.split(self.path)
        self.filebase, self.ext = os.path.splitext(self.filename)
        self.y, self.sampling_rate = librosa.load(self.path, sr=sampling_rate)
        self.spectrogram = self.compute_spectrogram() if process else None

    def compute_spectrogram(self, n_mels=256):
        """

        Args:
            n_mels: vertical resolution of the spectrogram

        Returns:
            melspectrogram on a log scale

        """
        return librosa.power_to_db(
            librosa.feature.melspectrogram(
                self.y, sr=self.sampling_rate, n_mels=n_mels
            ),
            ref=np.max,
        )  # convert to log scale (dB). Use the peak power (max) as ref

    def plot_spectrogram(self):
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(
            self.spectrogram,
            sr=self.sampling_rate,
            x_axis='time',
            y_axis='mel',
        )
        plt.title('mel power spectrogram')
        plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()
        plt.show()

    def to_samples(self, sample_size=1, output_folder=None, suffix='_sample_'):
        """
        Take the sound in self.y and split in into samples of size sample_size
        Args:
            sample_size: length of each sample in logs
            output_folder: folder where samples are saved
            suffix: string add at the end of the files
        """
        output_folder = self.folder if output_folder is None else output_folder
        i, step_size = 0, sample_size * self.sampling_rate
        n = len(self.y)
        while i * step_size < n:
            librosa.output.write_wav(
                os.path.join(
                    output_folder, self.filebase + suffix + str(i) + '.wav'
                ),
                self.y[i * step_size : min(i * step_size + step_size, n)],
                sr=self.sampling_rate,
                norm=False,
            )
            i += 1


if __name__ == '__main__':
    s1 = Sound('../misc/trap.mp3')
    s1.to_samples(sample_size=4)
    s1.plot_spectrogram()
