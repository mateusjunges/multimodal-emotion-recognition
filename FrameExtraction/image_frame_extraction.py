from cv2 import VideoCapture
from cv2 import resize
from PIL import Image
from cv2 import cvtColor
from cv2 import COLOR_BGR2RGB
from cv2 import destroyAllWindows
import os


def extract_video_frames(path_to_video, save_frames_to, framerate=None):
    i = 1
    if not os.path.exists(save_frames_to):
        os.makedirs(save_frames_to)
    cap = VideoCapture(path_to_video)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if framerate is not None:
            if i % framerate == 0:
                img = Image.fromarray(cvtColor(frame, COLOR_BGR2RGB))
                img.save(save_frames_to + '/' + str(i) + '.jpg')
        else:
            img = Image.fromarray(cvtColor(frame, COLOR_BGR2RGB))
            img.save(save_frames_to + '/' + str(i) + '.jpg')
        i += 1
    cap.release()
    destroyAllWindows()
    return save_frames_to