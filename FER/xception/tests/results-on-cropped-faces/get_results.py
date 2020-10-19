import os
import csv
import numpy as np

base_path = '/home/mateus/Documents/TCC/tcc-v2-1/FER/xception/tests/results-on-cropped-faces/results'
csvs = sorted(os.listdir(base_path))

correct = 0
wrong = 0

classes_fer = ["Angry", "Disgust", "Fearful", "Happy", "Sad", "Surprised", "Neutral"]

classes_ravdess = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]

for _csv in sorted(csvs):

    path = '{}/{}'.format(base_path, _csv)

    with open(base_path + "/" + _csv, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        count = 0
        for row in reader:
            try:
                count += 1
                print(classes_fer[int(row[9])])
                print(classes_ravdess[int(row[10]) -1])
            except:
                print("Erro:")
                print("Count: ")
                print("FER")
                print("RAVDESS")
                exit()
            # print(row[10])
            # exit()
            #   0      1       2     3     4       5       6      7      8        9             10
            # Angry,Disgust,Fearful,Happy,Sad,Surprised,Neutral,Video,Frame,Correct_Class,Predicted_Class
            if classes_ravdess[int(row[9])] == classes_fer[int(row[10]) - 1]:
                correct += 1
            elif classes_ravdess[int(row[9])] == "Neutral" and classes_fer[int(row[10]) - 1] == "Calm":
                correct += 1
            elif classes_ravdess[int(row[9])] == "Calm" and classes_fer[int(row[10]) - 1] == "Neutral":
                correct += 1
            else:
                wrong += 1


print("Acertou: {}".format(correct))
print("Errou: {}".format(wrong))
print("RESULTADO FINAL: {}%".format(correct / (correct + wrong)))