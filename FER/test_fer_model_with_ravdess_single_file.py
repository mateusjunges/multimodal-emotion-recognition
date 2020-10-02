import time
import os
import argparse
import sys
from tensorflow.keras.models import Model as ModelTF2
from tensorflow.keras.layers import Input as InputTF2
from tensorflow.keras.layers import Dropout as DropoutTF2
from tensorflow.keras.layers import Conv2D as Conv2DTF2
from tensorflow.keras.layers import MaxPooling2D as MaxPooling2DTF2
from tensorflow.keras.layers import BatchNormalization as BatchNormalizationTF2
from tensorflow.keras.layers import Flatten as FlattenTF2
from tensorflow.keras.layers import Dense as DenseTF2
from tensorflow.keras.regularizers import l2
import cv2
import numpy as np
from fer import FaceEmotionRecognition
import matplotlib.pyplot as plt

def main(args):
    print("Predicting emotions...")
    start = time.time()

    cropped_faces_folder = args.cropped_faces_folder
    output_folder = args.output_folder

    print("Loading FER model...")
    model = create_model()
    print("Model loaded! Loading weights...")
    model.load_weights('models/model1-fer-weights.h5')
    print("Model weights loaded!")

    FER = FaceEmotionRecognition()
    n_actors = len(os.listdir(cropped_faces_folder))

    print("Starting classification...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(output_folder + '/results.csv', mode='a+') as file:

        for folder in sorted(os.listdir(cropped_faces_folder)):
            done = 0

            n_images = len(next(os.walk(cropped_faces_folder + '/' + folder))[2])
            print(n_images)

            for path in sorted(os.listdir(cropped_faces_folder + '/' + folder)):
                image_path = cropped_faces_folder + '/' + folder + '/' + path
                image_name = os.path.splitext(path)[0]

                image = cv2.imread(image_path)
                correct_class = image_name[6:8]

                predictions = []
                angry, disgust, fear, happy, sad, surprise, neutral = [], [], [], [], [], [], []

                for face in FER.extract_features_from_face(FER.face_detector(image)):

                    to_predict = np.reshape(face.flatten(), (1, 48, 48, 1))
                    predicted_emotions = model.predict(to_predict)

                    angry.append(float('{:f}'.format(predicted_emotions[0][0])))
                    disgust.append(float('{:f}'.format(predicted_emotions[0][1])))
                    fear.append(float('{:f}'.format(predicted_emotions[0][2])))
                    happy.append(float('{:f}'.format(predicted_emotions[0][3])))
                    sad.append(float('{:f}'.format(predicted_emotions[0][4])))
                    surprise.append(float('{:f}'.format(predicted_emotions[0][5])))
                    neutral.append(float('{:f}'.format(predicted_emotions[0][6])))

                done += 1
                print("Processing {} of {}. Done {} of {} frames - {}%".format(folder, n_actors, done, n_images, round((done / n_images) * 100, 2)))

                if len(angry) and len(disgust) and len(fear) and len(happy) and len(sad) and len(surprise) and len(neutral):
                    # angry,disgust,fear,happy,sad,surprise,neutra,image_name,correct_class
                    results = "{},{},{},{},{},{},{},{},{}"\
                        .format(
                            (sum(angry) / len(angry)),
                            (sum(disgust) / len(disgust)),
                            (sum(fear) / len(fear)),
                            (sum(happy) / len(happy)),
                            (sum(sad) / len(sad)),
                            (sum(surprise) / len(surprise)),
                            (sum(neutral) / len(neutral)),
                            image_name,
                            correct_class
                        )

                    file.write(results)
                    file.write("\n")

    end = time.time()
    print("Classification finished. Total time: {}".format(end - start))


def create_model():
    num_features = 64
    width = height = 48
    input_shape = InputTF2(shape=(48, 48, 1))

    x = Conv2DTF2(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01))(input_shape)
    x = Conv2DTF2(num_features, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalizationTF2()(x)
    x = MaxPooling2DTF2(pool_size=(2, 2), strides=(2, 2))(x)
    x = DropoutTF2(0.5)(x)

    x = Conv2DTF2(2*num_features, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalizationTF2()(x)
    x = Conv2DTF2(2*num_features, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalizationTF2()(x)
    x = MaxPooling2DTF2(pool_size=(2, 2), strides=(2, 2))(x)
    x = DropoutTF2(0.5)(x)

    x = Conv2DTF2(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalizationTF2()(x)
    x = Conv2DTF2(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalizationTF2()(x)
    x = MaxPooling2DTF2(pool_size=(2, 2), strides=(2, 2))(x)
    x = DropoutTF2(0.5)(x)

    x = Conv2DTF2(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalizationTF2()(x)
    x = Conv2DTF2(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalizationTF2()(x)
    x = MaxPooling2DTF2(pool_size=(2, 2), strides=(2, 2))(x)
    x = DropoutTF2(0.5)(x)

    x = FlattenTF2()(x)

    x = DenseTF2(2*2*2*num_features, activation='relu')(x)
    x = DropoutTF2(0.4)(x)
    x = DenseTF2(2*2*num_features, activation='relu')(x)
    x = DropoutTF2(0.4)(x)
    x = DenseTF2(2*num_features, activation='relu')(x)
    x = DropoutTF2(0.5)(x)

    x = DenseTF2(7, activation='softmax')(x)

    return ModelTF2(inputs=input_shape, outputs=x)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('cropped_faces_folder', type=str, help='The path to the cropped faces folder')
    parser.add_argument('output_folder', type=str, help='The path where the .csv file will be saved')
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))