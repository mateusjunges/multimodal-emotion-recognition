import os
from glob import glob
import pickle
import itertools
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from PIL import Image
import librosa
import sys
import argparse


emotions = {
    '02': 'Calmo',
    '03': 'Felicidade',
    '04': 'Tristeza',
    '05': 'Raiva',
    '06': 'Medo',
    '07': 'Desgosto',
    '08': 'Supresa'
}


def mel_spectrogram(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):
    mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2

    # Compute mel spectrogram
    mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)

    # Compute log-mel spectrogram
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

    return mel_spect

def get_ravdess_label(key):
    return emotions.get(key, 'Not Found')


def main(args):
    audio_file = args.audio_file
    sample_rate = 16000
    max_pad_len = 49100
    # Read audio file
    y, sr = librosa.core.load(audio_file, sr=sample_rate, offset=0.5)

    # Z-normalization
    y = zscore(y)

    # Padding or truncated signal
    if len(y) < max_pad_len:
        y_padded = np.zeros(max_pad_len)
        y_padded[:len(y)] = y
        y = y_padded
    elif len(y) > max_pad_len:
        y = np.asarray(y[:max_pad_len])

    # Add to signal list
    signal = y

    mel_spect = mel_spectrogram(signal)

    correct_class = audio_file.split('/')[-1]
    correct_class = correct_class[6:8]

    label = get_ravdess_label(correct_class)

    plt.figure(figsize=(20, 10))
    plt.imshow(mel_spect, origin='lower', aspect='auto', cmap='viridis')
    plt.title('Espectograma de um áudio com emoção {}'.format(label), fontsize=26)
    plt.tight_layout()
    plt.show()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_file', type=str, help="File to generate spectogram")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))


