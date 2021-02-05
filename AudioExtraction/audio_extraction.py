import subprocess


def extract_video_audio(path_to_video, save_to):
    if path_to_video.endswith('.mp4'):
        name = str.split(path_to_video, '/')[-1]
        name = str.split(name, '.')[0]

        command = "ffmpeg -i " + path_to_video + " -ab 16k -ac 2 -ar 44100 -vn " + save_to + "audio.wav"

        # Execute conversion:
        subprocess.call(command, shell=True)
        return save_to + 'audio.wav'
