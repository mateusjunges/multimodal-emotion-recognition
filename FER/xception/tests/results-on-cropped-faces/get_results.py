import os
import csv
import numpy as np

base_path = '/home/mateus/Documents/TCC/tcc-v2-1/FER/xception/tests/results-on-cropped-faces/results-with-class-name'
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
            correct_class = row[11]
            predicted_class = row[12]

            #  0     1       2      3     4     5       6        7      8        9             10              11                12
            # Angry,Disgust,Fearful,Happy,Sad,Surprised,Neutral,Video,Frame,Correct_Class,Predicted_Class,Correct_Class_Name,Predicted_Class_Name
            if correct_class == predicted_class:
                correct += 1
            elif correct_class == 'Calm' and predicted_class == 'Neutral':
                correct += 1
            elif predicted_class == 'Calm' and correct_class == 'Neutral':
                correct += 1
            else:
                wrong += 1


print("Acertou: {}".format(correct))
print("Errou: {}".format(wrong))
print("RESULTADO FINAL: {:.2f}%".format(correct / (correct + wrong) * 100))