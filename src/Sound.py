import os
import numpy as np
import librosa  # Librosa for audio
import librosa.display  # And the display module for visualization
import matplotlib.pyplot as plt


class Sound():
    def __init__(self, path=None, sr=None):
        self.path = path
        self.wave, self.sr = librosa.load(self.path, sr=sr)

        S = librosa.feature.melspectrogram(self.wave, sr=self.sr, n_mels=256)
        log_S = librosa.power_to_db(S, ref=np.max)  # Convert to log scale (dB). Use the peak power (max) as reference.
        self.spectrogram = log_S

    def plot_spectrogram(self):
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(self.spectrogram, sr=self.sr, x_axis='time', y_axis='mel')
        plt.title('mel power spectrogram')
        plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()
        plt.show()


s1 = Sound('../misc/trap.mp3')
s1.plot_spectrogram()
