# python frame_extraction.py ~/Doutorado/BASES/RAVDESS/videos ~/Doutorado/BASES/RAVDESS/frames

import argparse
import time
import sys
import os

from cv2 import VideoCapture
from PIL import Image
from cv2 import cvtColor
from cv2 import COLOR_BGR2RGB
from cv2 import destroyAllWindows

def main(args):

	print('Extracting frames...')
	t_inicio = time.time()
	final_path = []

	ravdess_folder = args.video_folder
	save_frames_to = args.output_folder

	for actor in sorted(os.listdir(ravdess_folder)): 
		actor_folder = ravdess_folder + '/' + actor
		for video in sorted(os.listdir(actor_folder)): 
			final_path.append(actor_folder + '/' + video)

	n_videos = len(final_path)
	done = 0

	for video_path in final_path:

		total_frames_for_video = 0
		i = 1
		print(video_path)
		actor = video_path.split('/')[-2]
		video = video_path.split('/')[-1].split('.')[0]
		
		if not os.path.exists(save_frames_to + '/' + actor):
			os.makedirs(save_frames_to + '/' + actor)

		cap = VideoCapture(video_path)

		while cap.isOpened():

			ret, frame = cap.read()
			if i > 25:
				if not ret:
					break

				img = Image.fromarray(cvtColor(frame, COLOR_BGR2RGB))
				img.save(save_frames_to + '/' + actor + '/' + video + '_frame-' + str(i) + '.jpg')

			i += 1
			total_frames_for_video += 1
			# print("Frame saved: {}".format(save_frames_to + '/' + actor + '/' + video + '_frame-' + str(i) + '.jpg'))

		done += 1
		print("Processing... {}/{} ({} %)".format(done, (n_videos), round((done / (n_videos)) * 100, 2)))

		remove_until = total_frames_for_video - 25

		if total_frames_for_video > 0:
			while total_frames_for_video > remove_until:
				os.remove(save_frames_to + '/' + actor + '/' + video + '_frame-' + str(total_frames_for_video) + '.jpg')
				total_frames_for_video -= 1



	t2 = time.time()
	print('Extracting frames complete!')
	print("Tempo total: {}".format(t2-t_inicio))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('video_folder', type=str, help='Input video folder')
    parser.add_argument('output_folder', type=str, help='Output')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
