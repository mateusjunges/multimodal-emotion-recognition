import os
from glob import glob
import pickle
import numpy as np
from IPython.display import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, TimeDistributed, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from keras.utils import np_utils
from keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder
import warnings


warnings.filterwarnings('ignore')

base_path = 'tcc/'

X_train = pickle.load(open(base_path + 'features-from-extracted-audios/x_train.p', 'rb'))
y_train = pickle.load(open(base_path + 'features-from-extracted-audios/y_train.p', 'rb'))
y_test = pickle.load(open(base_path + 'features-from-extracted-audios/y_test.p', 'rb'))
X_test = pickle.load(open(base_path + 'features-from-extracted-audios/x_test.p', 'rb'))

# Encode Label from categorical to numerical
lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(np.ravel(y_train)))
y_test = np_utils.to_categorical(lb.transform(np.ravel(y_test)))

# Reshape for convolution
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] , X_train.shape[2], X_train.shape[3], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] , X_test.shape[2], X_test.shape[3], 1)

K.clear_session()

# Define two sets of inputs: MFCC and FBANK
input_y = Input(shape=X_train.shape[1:], name='Input_MELSPECT')

## First LFLB (local feature learning block)
y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'))(input_y)
y = TimeDistributed(BatchNormalization())(y)
y = TimeDistributed(Activation('elu'))(y)
y = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))(y)
y = TimeDistributed(Dropout(0.2))(y)

## Second LFLB (local feature learning block)
y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'))(y)
y = TimeDistributed(BatchNormalization())(y)
y = TimeDistributed(Activation('elu'))(y)
y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))(y)
y = TimeDistributed(Dropout(0.2))(y)

## Second LFLB (local feature learning block)
y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'))(y)
y = TimeDistributed(BatchNormalization())(y)
y = TimeDistributed(Activation('elu'))(y)
y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))(y)
y = TimeDistributed(Dropout(0.2))(y)

## Second LFLB (local feature learning block)
y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'))(y)
y = TimeDistributed(BatchNormalization())(y)
y = TimeDistributed(Activation('elu'))(y)
y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))(y)
y = TimeDistributed(Dropout(0.2))(y)

## Flat
y = TimeDistributed(Flatten())(y)

# Apply 2 LSTM layer and one FC
y = LSTM(256, return_sequences=False, dropout=0.2)(y)
y = Dense(y_train.shape[1], activation='softmax', name='FC')(y)

# Build final model
model = Model(inputs=input_y, outputs=y)

# Plot model graph
plot_model(model, show_shapes=True, show_layer_names=True, to_file=base_path + 'model.png')
Image(retina=True, filename=base_path + 'model.png')

# Compile model
model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.8), loss='categorical_crossentropy', metrics=['accuracy'])

# Save best model
best_model_save = ModelCheckpoint(base_path + 'best_model.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')

# Early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, verbose=1, mode='max')

# Fit model
history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping, best_model_save])

# Loss Curves
plt.figure(figsize=(25, 10))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b', linewidth=2.0)
plt.plot(history.history['val_loss'], 'r', linewidth=2.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=14)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=22)

# Accuracy Curves
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b', linewidth=2.0)
plt.plot(history.history['val_accuracy'], 'r', linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
plt.xlabel('Epochs ', fontsize=16)

plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=22)
plt.show()


model.save(base_path + 'model.h5')
model.save_weights(base_path + 'model-weights.h5')