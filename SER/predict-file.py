import time
import os
import numpy as np
import wave
import librosa
from scipy.stats import zscore
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, TimeDistributed
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.layers import LSTM
import sys
import argparse

emotions = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}


def get_emotion_label(key):
    return emotions.get(key, 'Emoção não encontrada')


def mel_spectrogram(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):

    # Compute spectogram
    mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2

    # Compute mel spectrogram
    mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)

    # Compute log-mel spectrogram
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

    return np.asarray(mel_spect)


def frame(y, win_step=64, win_size=128):

    # Number of frames
    nb_frames = 1 + int((y.shape[2] - win_size) / win_step)

    # Framming
    frames = np.zeros((y.shape[0], nb_frames, y.shape[1], win_size)).astype(np.float16)
    for t in range(nb_frames):
        frames[:, t, :, :] = np.copy(y[:, :, (t * win_step):(t * win_step + win_size)]).astype(np.float16)

    return frames


def build_model():
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


def main(args):
    print("Loading Model...")

    # model = load_model('SER/holdout/best_model.hdf5')
    model = build_model()
    print("Loading model weights...")
    cwd = os.getcwd()
    model.load_weights(cwd + '/holdout/model-weights.h5')

    audio_file_to_predict = args.file
    chunk_step = 16000
    chunk_size = 49100
    sample_rate = 16000

    if args.proba == 'yes':
        predict_proba = True
    else:
        predict_proba = False


    y, sr = librosa.core.load(audio_file_to_predict, sr=sample_rate, offset=0.5)

    if y.shape[0] < chunk_size:
        y_padded = np.zeros(chunk_size)
        y_padded[:y.shape[0]] = y
        y = y_padded

    # Split audio signals into chunks
    chunks = frame(y.reshape(1, 1, -1), chunk_step, chunk_size)

    # Reshape chunks
    chunks = chunks.reshape(chunks.shape[1], chunks.shape[-1])

    # Z-normalization
    y = np.asarray(list(map(zscore, chunks)))

    # Compute mel spectrogram
    mel_spect = np.asarray(list(map(mel_spectrogram, y)))

    # Time distributed Framing
    mel_spect_ts = frame(mel_spect)

    # Build X for time distributed CNN
    X = mel_spect_ts.reshape(mel_spect_ts.shape[0],
                             mel_spect_ts.shape[1],
                             mel_spect_ts.shape[2],
                             mel_spect_ts.shape[3],
                             1)

    predictions = None
    predict = []
    # Predict emotion
    if predict_proba is True:
        predictions = model.predict(X)
        predictions = {
            'Angry': "{}".format(predictions[0][0]),
            'Disgust': "{}".format(predictions[0][1]),
            'Fear': "{}".format(predictions[0][2]),
            'Happy': "{}".format(predictions[0][3]),
            'Neutral/Calm': "{}".format(predictions[0][4]),
            'Sad': "{}".format(predictions[0][5]),
            'Surprise': "{}".format(predictions[0][6])
        }
    else:
        predict = np.argmax(model.predict(X), axis=1)
        predict = [emotions.get(emotion) for emotion in predict]

    # Clear Keras session
    K.clear_session()

    print("\n\nPrediction results: \n")
    if predict_proba is True:
        print("Angry: {}".format(float(predictions.get('Angry')) * 100))
        print("Disgust: {}".format(float(predictions.get('Disgust')) * 100))
        print("Fear: {}".format(float(predictions.get('Fear')) * 100))
        print("Happy: {}".format(float(predictions.get('Happy')) * 100))
        print("Neutral/Calm: {}".format(float(predictions.get('Neutral/Calm')) * 100))
        print("Sad: {}".format(float(predictions.get('Sad')) * 100))
        print("Surprise: {}".format(float(predictions.get('Surprise')) * 100))
    else:
        print(predict[0])

    if predictions is not None:
        predict = np.argmax(model.predict(X), axis=1)
        print("\nMost present emotion:")
        predict = [emotions.get(emotion) for emotion in predict]
        print(predict[0])


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help="File that will be predicted")
    parser.add_argument('proba', type=str, help="Predict probabilities")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))