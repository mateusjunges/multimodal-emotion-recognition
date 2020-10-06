from tensorflow.keras.models import load_model
import numpy as np
import argparse
import time
import csv
import cv2
import sys
import os

classes_fer = ["Angry", "Disgust", "Fearful", "Happy", "Sad", "Surprised", "Neutral"]

classes_ravdess = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]


def main(args):
    print('Testing...')
    t_inicio = time.time()

    # Load parameters
    image_file = args.image
    model_path = args.model

    # Load the pre-trained X-Ception model
    model = load_model(model_path)

    face = cv2.imread(image_file)
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (48, 48))
    output = face_resized.astype(np.float32)
    output = output / float(output.max())
    to_predict = np.reshape(output.flatten(), (1, 48, 48, 1))

    predictions = []

    prediction = model.predict(to_predict)
    print("\n\n\n\n\n\nPrediction results:\n")
    for key, value in enumerate(prediction[0]):
        print("{}: {}".format(classes_fer[key], value * 100))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Image to predict')
    parser.add_argument('model', type=str, help='Model path')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
