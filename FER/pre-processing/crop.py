# python3 crop.py ~/Doutorado/BASES/RAVDESS/frames ~/Doutorado/BASES/RAVDESS/normalized

import cv2
import glob
import os
import sys
import argparse
import dlib
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import timeit
import pickle
from scipy import misc
import time


def main(args):
	
	print('Extracting faces...')
	t_inicio = time.time()

	images_folder = args.image_folder
	output_folder = args.output_folder
	
	size = 155
	padding = 0.1

	actor = 0

	predictor_path = "dlib/shape_predictor_5_face_landmarks.dat"
	# Load all the models we need: a detector to find the faces, a shape predictor
	# to find face landmarks so we can precisely localize the face
	dlib_frontal_face_detector_file = open("dlib/dlib_frontal_face_detector.dat", "rb")
	detector = pickle.load(dlib_frontal_face_detector_file)
	#detector = dlib.get_frontal_face_detector()
	sp = dlib.shape_predictor(predictor_path)

	for folder in sorted(os.listdir(images_folder)):
		actor += 1
		done = 0
		n_images = len(next(os.walk(images_folder + '/' + folder))[2])

		for path in sorted(os.listdir(images_folder + '/' + folder)):

			image_path = images_folder +'/'+ folder + '/' + path
			imgname = os.path.splitext(path)[0]
			#print(image)
			img = dlib.load_rgb_image(image_path)
			# Ask the detector to find the bounding boxes of each face.
			dets = detector(img, 0)
			num_faces = len(dets)
			if (num_faces == 1):
				faces = dlib.full_object_detections()
				faces.append(sp(img, dets[0]))
				aligned = dlib.get_face_chips(img, faces, size, padding)
				#img_list.append(aligned[0])
				save_path = output_folder +'/'+ folder + '/' + "{}.png".format(imgname)
				directory = os.path.dirname(save_path)
				if not os.path.exists(directory):
					os.makedirs(directory)
				#print(save_path)
				final_image = cv2.cvtColor(aligned[0], cv2.COLOR_BGR2RGB)
				cv2.imwrite(save_path, final_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
				#misc.imsave(save_path, aligned[0])
			
			done = done + 1
			print("Processing Actor {}... {}/{} ({} %)".format(actor, done, (n_images), round((done/(n_images))*100, 2)))
		

	t2 = time.time()
	print('Extracting faces complete!')
	print("Tempo total: {}".format(t2-t_inicio))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('image_folder', type=str, help='Images to normalize')
    parser.add_argument('output_folder', type=str, help='Output')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
