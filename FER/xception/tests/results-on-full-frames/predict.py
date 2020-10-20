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

def main():
    full_frames_path = "/home/mateus/Documents/datasets/ravdess-frames"
    model_path = "/home/mateus/Documents/TCC/tcc-v2-1/FER/xception/xception-model.h5"
    weights_path = "/home/mateus/Documents/TCC/tcc-v2-1/FER/xception/tests/final_xception.h5"

    model = load_model(model_path)
    model.load_weights(weights_path)

    actors = os.listdir(full_frames_path)
    done_actors = 0

    for actor in sorted(actors):
        done_actors += 1
        final_result = [
            [
                "Angry",
                "Disgust",
                "Fearful",
                "Happy",
                "Sad",
                "Surprised",
                "Neutral",
                "Video",
                "Frame",
                "Correct_Class",
                "Predicted_Class",
                "Correct_Class_Name",
                "Predicted_Class_Name"
            ]
        ]

        csv_name = "{}.csv".format(actor)

        done = 0

        images = os.listdir(full_frames_path + "/" + actor)
        for image in sorted(images):
            image_path = full_frames_path + "/" + actor + "/" + image
            video = image.split('_')[0]
            correct_class = image.split('-')[2]
            frame = image.split('-')[-1]

            face = cv2.imread(image_path)
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))

            output = face_resized.astype(np.float32)

            output = output / float(output.max())
            to_predict = np.reshape(output.flatten(), (1, 48, 48, 1))

            predictions = []

            prediction = model.predict(to_predict)
            prediction_result = np.argmax(prediction)

            for item in prediction[0]:
                predictions.append(float(item))

            predicted_class_name = classes_fer[prediction_result]
            correct_class_name = classes_ravdess[int(correct_class) - 1]
            predictions.append(video)
            predictions.append(frame)
            predictions.append(correct_class)
            predictions.append(prediction_result)
            predictions.append(correct_class_name)
            predictions.append(predicted_class_name)
            final_result.append(predictions)

            done += 1

            print("Processing actor {} of {} actors - Image {} of {} - Predicted: {} - Correct: {}".format(done_actors, len(actors), done, len(images), predicted_class_name, correct_class_name))

            with open("/home/mateus/Documents/TCC/tcc-v2-1/FER/xception/tests/results-on-full-frames/results-with-class-name/{}".format(csv_name), mode='a+') as file:
                writer = csv.writer(file)
                writer.writerows(final_result)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('images', type=str, help="Path to the cropped images")
    parser.add_argument('model', type=str, help="Path to the xception model")
    parser.add_argument('weights', type=str, help="Path to the xception weights")
    return parser.parse_args(argv)

if __name__ == '__main__':
    # main(parse_arguments(sys.argv[1:]))
    main()