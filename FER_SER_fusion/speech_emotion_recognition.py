import time
import os
import numpy as np
import wave
import librosa
from scipy.stats import zscore
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, TimeDistributed
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.layers import LSTM


class SpeechEmotionRecognition:

    def __init__(self, path_to_model=None, path_to_weights=None):

        # Load prediction model
        if path_to_model is not None:
            self._model = load_model(path_to_model)
            self._model.load_weights(path_to_weights)

        self._emotion = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Neutral',
            5: 'Sad',
            6: 'Surprise'
        }

    def get_emotion_label(self, key):
        return self._emotion.get(key, 'Emoção não encontrada')

    @staticmethod
    def mel_spectrogram(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128,
                        fmax=4000):

        # Compute spectogram
        mel_spect = np.abs(
            librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2

        # Compute mel spectrogram
        mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)

        # Compute log-mel spectrogram
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

        return np.asarray(mel_spect)

    @staticmethod
    def frame(y, win_step=64, win_size=128):

        # Number of frames
        nb_frames = 1 + int((y.shape[2] - win_size) / win_step)

        # Framming
        frames = np.zeros((y.shape[0], nb_frames, y.shape[1], win_size)).astype(np.float16)
        for t in range(nb_frames):
            frames[:, t, :, :] = np.copy(y[:, :, (t * win_step):(t * win_step + win_size)]).astype(np.float16)

        return frames

    @staticmethod
    def build_model():

        # Clear Keras session
        K.clear_session()

        # Define input
        input_shape = Input(shape=(5, 128, 128, 1))

        # First LFLB (local feature learning block)
        y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'))(input_shape)
        y = TimeDistributed(BatchNormalization())(y)
        y = TimeDistributed(Activation('elu'))(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))(
            y)
        y = TimeDistributed(Dropout(0.2))(y)

        # Second LFLB (local feature learning block)
        y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'))(y)
        y = TimeDistributed(BatchNormalization())(y)
        y = TimeDistributed(Activation('elu'))(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))(
            y)
        y = TimeDistributed(Dropout(0.2))(y)

        # Third LFLB (local feature learning block)
        y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'))(y)
        y = TimeDistributed(BatchNormalization())(y)
        y = TimeDistributed(Activation('elu'))(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))(
            y)
        y = TimeDistributed(Dropout(0.2))(y)

        # Fourth LFLB (local feature learning block)
        y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'))(y)
        y = TimeDistributed(BatchNormalization())(y)
        y = TimeDistributed(Activation('elu'))(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))(
            y)
        y = TimeDistributed(Dropout(0.2))(y)

        # Flat
        y = TimeDistributed(Flatten())(y)

        # LSTM layer
        y = LSTM(256, return_sequences=False, dropout=0.2)(y)

        # Fully connected
        y = Dense(7, activation='softmax')(y)

        model = Model(inputs=input_shape, outputs=y)

        return model

    def predict_emotion_from_file(self, filename, chunk_step=16000, chunk_size=49100, predict_proba=False,
                                  sample_rate=16000):

        # Read audio file
        y, sr = librosa.core.load(filename, sr=sample_rate, offset=0.5)

        if y.shape[0] < chunk_size:
            y_padded = np.zeros(chunk_size)
            y_padded[:y.shape[0]] = y
            y = y_padded

        # Split audio signals into chunks
        chunks = self.frame(y.reshape(1, 1, -1), chunk_step, chunk_size)

        # Reshape chunks
        chunks = chunks.reshape(chunks.shape[1], chunks.shape[-1])

        # Z-normalization
        y = np.asarray(list(map(zscore, chunks)))

        # Compute mel spectrogram
        mel_spect = np.asarray(list(map(self.mel_spectrogram, y)))

        # Time distributed Framing
        mel_spect_ts = self.frame(mel_spect)

        # Build X for time distributed CNN
        X = mel_spect_ts.reshape(mel_spect_ts.shape[0],
                                 mel_spect_ts.shape[1],
                                 mel_spect_ts.shape[2],
                                 mel_spect_ts.shape[3],
                                 1)

        # Predict emotion
        if predict_proba is True:
            predict = self._model.predict(X)
        else:
            predict = np.argmax(self._model.predict(X), axis=1)
            predict = [self._emotion.get(emotion) for emotion in predict]

        # Clear Keras session
        K.clear_session()

        # Predict timestamp
        timestamp = np.concatenate([[chunk_size], np.ones((len(predict) - 1)) * chunk_step]).cumsum()
        timestamp = np.round(timestamp / sample_rate)

        return [predict, timestamp]

    '''
    Export emotions predicted to csv format
    '''

    def prediction_to_csv(self, predictions, filename, mode='w'):

        # Write emotion in filename
        with open(filename, mode) as f:
            if mode == 'w':
                f.write("EMOTIONS" + '\n')
            for emotion in predictions:
                f.write(str(emotion) + '\n')
            f.close()
