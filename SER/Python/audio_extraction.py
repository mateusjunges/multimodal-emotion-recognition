import argparse
import sys
import subprocess
import time
import os


def extract_video_audio(path_to_video, save_to):
    if path_to_video.endswith('.mp4'):
        name = str.split(path_to_video, '/')[-1]
        name = str.split(name, '.')[0]

        command = "ffmpeg -i " + path_to_video + " -ab 160k -ac 2 -ar 44100 -vn " + save_to

        # Execute conversion:
        subprocess.call(command, shell=True)
        return save_to + 'audio.wav'


def main(args):
    ravdess_folder = args.videos_folder

    output_folder = args.output_folder

    start = time.time()

    final_path = []

    for actor in sorted(os.listdir(ravdess_folder)):
        actor_folder = ravdess_folder + '/' + actor
        for video in sorted(os.listdir(actor_folder)):
            final_path.append(actor_folder + '/' + video)

    n_videos = len(final_path)
    done = 0

    for video_path in final_path:

        i = 1

        actor = video_path.split('/')[-2]
        video = video_path.split('/')[-1].split('.')[0]
        filename = video.split('/')[-1].split('.')[0]
        output = output_folder + '/' + actor + '/' + filename + '.wav'

        if not os.path.exists(output_folder + '/' + actor):
            os.makedirs(output_folder + '/' + actor)

        extract_video_audio(video_path, output)
        done += 1
        print("Done {} of {} - {}%".format(done, n_videos, round((done / n_videos) * 100, 2)))

    print("\n\n\n")
    print("Extraction finished!")
    print("Total time: {}".format(time.time() - start))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('videos_folder', type=str, help='Path to the videos folder')
    parser.add_argument('output_folder', type=str, help='Audios output path')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))