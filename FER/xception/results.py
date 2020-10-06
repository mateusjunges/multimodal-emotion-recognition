import os
import csv
import numpy as np

actors = sorted(os.listdir('/home/mateus/Documents/UEPG/original-dataset/'))

acertou = 0
errou = 0

for actor in actors:

	path = '/home/mateus/Documents/TCC/tcc-v2-1/FER/xception/results/'.format(actor)
	videos = sorted(os.listdir(path))

	labels = ['Angry', 'Calm', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

	lista_videos = []
	for video in videos:
		if video.split('.')[0] not in lista_videos:
			lista_videos.append(video.split('.')[0])
	print(lista_videos)
	exit()


	for video in lista_videos:

		print(video)
		angry, calm, disgust, fearful, happy, neutral, sad, surprised = 0,0,0,0,0,0,0,0

		csv_file = "{}.csv".format(actor)
		print(csv_file)
		with open(csv_file, 'r') as arquivo_csv:

			leitor = csv.reader(arquivo_csv, delimiter=',')

			for linha in leitor:

				if video == linha[7]:

					true_label = linha[9]

					if linha[9] != 'Neutral':
						predicted = linha[10]
						if predicted == 'Angry':
							angry += 1
						elif predicted == 'Disgust':
							disgust += 1
						elif predicted == 'Fearful':
							fearful += 1
						elif predicted == 'Happy':
							happy += 1
						elif predicted == 'Neutral':
							calm += 1
						elif predicted == 'Sad':
							sad += 1
						elif predicted == 'Surprised':
							surprised += 1
						else:
							print('ERROR')
							print(predicted)
							break

		print(true_label)
		results = [angry, calm, disgust, fearful, happy, neutral, sad, surprised]
		prediction = np.argmax(results)
		print(labels[prediction])
		#print('{}, {}, {}, {}, {}, {}, {}, {}'.format(angry, calm, disgust, fearful, happy, neutral, sad, surprised))
		if true_label == labels[prediction]:
			print('ACERTOU')
			acertou += 1

		else:
			print('ERROU')
			errou += 1


print("Acertou: {}".format(acertou))
print("Errou: {}".format(errou))
print("RESULTADO FINAL: {}".format(acertou/(acertou+errou)))
