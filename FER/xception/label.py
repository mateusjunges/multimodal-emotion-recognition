#python3 label.py  ~/Doutorado/BASES/RAVDESS/normalized final_xception.h5

from tensorflow.keras.models import load_model
import numpy as np
import argparse
import time
import csv
import cv2
import sys
import os

classes_fer = ["Angry","Disgust","Fearful","Happy","Sad","Surprised","Neutral"]

classes_ravdess = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]

def main(args):
	
	print('Testing...')
	t_inicio = time.time()

	#Load parameters
	images_folder = args.images
	model_path = args.model
	
	# Load the pre-trained X-Ception model
	model = load_model(model_path)

	actor = 0

	for folder in sorted(os.listdir(images_folder)):

		final_result = [["Angry", "Disgust", "Fearful", "Happy", "Sad", "Surprised", "Neutral", "Video", "Frame", "Correct_Class", "Predicted_Class"]]
		actor += 1
		csv_name = "results/Actor_{}.csv".format(actor)
		done = 0
		n_images = len(next(os.walk(images_folder + '/' + folder))[2])

		for path in sorted(os.listdir(images_folder + '/' + folder)):

			image_path = images_folder +'/'+ folder + '/' + path
			imgname = os.path.splitext(path)[0]

			video = imgname.split('_')[0]
			frame = imgname.split('-')[-1]
			classe = imgname.split('-')[2]
			
			nome_classe = classes_ravdess[int(classe)-1]
			if nome_classe != 'Neutral':		
				face = cv2.imread(image_path)
				face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
				face_resized = cv2.resize(face_gray, (48,48))
				
				#Cast type float
				output = face_resized.astype(np.float32)

				#Scale
				output = output / float(output.max())
				to_predict = np.reshape(output.flatten(), (1, 48, 48, 1))
				  
				predictions = []
				
				#Make Prediction
				prediction = model.predict(to_predict)
				prediction_result = np.argmax(prediction)

				for item in prediction[0]:
					predictions.append(float(item))
				
				predictions.append(video)
				predictions.append(frame)
				predictions.append(nome_classe)
				predictions.append(classes_fer[prediction_result])
				
				final_result.append(predictions)
				
				done = done + 1
				print("Processing Actor {}... {}/{} ({} %)".format(actor, done, (n_images), round((done/(n_images))*100, 2)))
		
			with open(csv_name, 'w', newline='') as file:
				writer = csv.writer(file)
				writer.writerows(final_result)


	t2 = time.time()
	print('Testing complete!')
	print("Tempo total: {}".format(t2-t_inicio))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('images', type=str, help='Images folder path')
    parser.add_argument('model', type=str, help='Model path')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
