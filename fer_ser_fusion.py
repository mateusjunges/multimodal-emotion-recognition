from __future__ import division
from AudioExtraction.audio_extraction import extract_video_audio
from FrameExtraction.image_frame_extraction import extract_video_frames
from FER_SER_fusion.speech_emotion_recognition import SpeechEmotionRecognition
from FER_SER_fusion.face_emotion_recognition import FaceEmotionRecognition
from keras.models import model_from_json
import os
import numpy as np
import cv2
import operator
import time


def analyse(video_path, output_file):
    for _file in os.listdir('extracted-frames'):
        os.remove('extracted-frames/' + _file)
    for _file in os.listdir('extracted-audios'):
        os.remove('extracted-audios/' + _file)

    step = 1
    sample_rate = 16000

    filename = video_path.split('/')[-1]
    filename_class = filename.split('.')[0]


    # Extrair o áudio do vídeo enviado:
    audio_path = extract_video_audio(video_path, 'extracted-audios/')

    # Extrair os frames do vídeo enviado:
    frames_path = extract_video_frames(video_path, 'extracted-frames')

    # Analise do áudio extraído:
    SER = SpeechEmotionRecognition('models/ser/best_model.hdf5', 'models/ser/model-weights.h5')

    emotions, timestamp = SER.predict_emotion_from_file(audio_path, predict_proba=True, chunk_step=step * sample_rate)

    emotion = np.argmax(emotions)

    predictions_from_audio_float = {
        'angry': emotions[0][0],
        'disgust': emotions[0][1],
        'fear': emotions[0][2],
        'happy': emotions[0][3],
        'neutral': emotions[0][4],
        'sad': emotions[0][5],
        'surprise': emotions[0][6]
    }

    # Análise dos frames extraídos
    with open('models/model1-fer.json', 'r') as file:
        json = file.read()
    model = model_from_json(json)
    # model = create_model1()
    model.load_weights('models/model1-fer-weights.h5')

    # Arrays pra salvar as probabilidades de cada emoção por frame analisado
    angry, disgust, fear, happy, sad, surprise, neutral = [], [], [], [], [], [], []

    for frame in os.listdir(frames_path):
        if frame not in ['.gitkeep']:
            FER = FaceEmotionRecognition()

            image_to_predict = frames_path + '/' + frame

            face = cv2.imread(image_to_predict)
            predictions_frame = []

            for face in FER.extract_features_from_face(FER.face_detector(face)):
                to_predict = np.reshape(face.flatten(), (1, 48, 48, 1))
                emotions = model.predict(to_predict)

                angry.append(float('{:f}'.format(emotions[0][0])))
                disgust.append(float('{:f}'.format(emotions[0][1])))
                fear.append(float('{:f}'.format(emotions[0][2])))
                happy.append(float('{:f}'.format(emotions[0][3])))
                sad.append(float('{:f}'.format(emotions[0][4])))
                surprise.append(float('{:f}'.format(emotions[0][5])))
                neutral.append(float('{:f}'.format(emotions[0][6])))

                prediction = np.argmax(emotions)
                predictions_frame.append(prediction)


    predictions_from_frames_float = {
        'angry': sum(angry) / len(angry),
        'disgust': sum(disgust) / len(disgust),
        'fear': sum(fear) / len(fear),
        'happy': sum(happy) / len(happy),
        'neutral': sum(neutral) / len(neutral),
        'sad': sum(sad) / len(sad),
        'surprise': sum(surprise) / len(surprise)
    }

    predictions_with_sum = {
        'angry': predictions_from_frames_float['angry'] + predictions_from_audio_float['angry'],
        'disgust': predictions_from_frames_float['disgust'] + predictions_from_audio_float['disgust'],
        'fear': predictions_from_frames_float['fear'] + predictions_from_audio_float['fear'],
        'happy': predictions_from_frames_float['happy'] + predictions_from_audio_float['happy'],
        'neutral': predictions_from_frames_float['neutral'] + predictions_from_audio_float['neutral'],
        'sad': predictions_from_frames_float['sad'] + predictions_from_audio_float['sad'],
        'surprise': predictions_from_frames_float['surprise'] + predictions_from_audio_float['surprise'],
    }

    predictions_with_product = {
        'angry': predictions_from_frames_float['angry'] * predictions_from_audio_float['angry'],
        'disgust': predictions_from_frames_float['disgust'] * predictions_from_audio_float['disgust'],
        'fear': predictions_from_frames_float['fear'] * predictions_from_audio_float['fear'],
        'happy': predictions_from_frames_float['happy'] * predictions_from_audio_float['happy'],
        'neutral': predictions_from_frames_float['neutral'] * predictions_from_audio_float['neutral'],
        'sad': predictions_from_frames_float['sad'] * predictions_from_audio_float['sad'],
        'surprise': predictions_from_frames_float['surprise'] * predictions_from_audio_float['surprise'],
    }

    class_sum = max(predictions_with_sum.items(), key=operator.itemgetter(1))[0]
    class_product = max(predictions_with_product.items(), key=operator.itemgetter(1))[0]

    # filename, s_angry, s_disgust, s_fear, s_happy, s_neutral, s_sad, s_surprise, p_angry, p_disgust, p_fear, p_happy, p_neutral, p_sad, p_surprise, class_sum, class_product
    append_to_file = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}"\
        .format(
            filename_class,
            predictions_with_sum['angry'],
            predictions_with_sum['disgust'],
            predictions_with_sum['fear'],
            predictions_with_sum['happy'],
            predictions_with_sum['neutral'],
            predictions_with_sum['sad'],
            predictions_with_sum['surprise'],
            predictions_with_product['angry'],
            predictions_with_product['disgust'],
            predictions_with_product['fear'],
            predictions_with_product['happy'],
            predictions_with_product['neutral'],
            predictions_with_product['sad'],
            predictions_with_product['surprise'],
            class_sum,
            class_product,
        )

    output_file.write(append_to_file)
    output_file.write("\n")


if __name__ == '__main__':
    print("Classification started")
    start = time.time()
    output_folder = "FER_SER_fusion"
    total_actors = len(os.listdir('/home/mateus/Downloads/Video_Speech'))

    current_actor = 0
    for actor in os.listdir('/home/mateus/Downloads/Video_Speech'):
        if actor.startswith('Actor'):
            current_actor += 1
            total_videos_per_actor = len(os.listdir('/home/mateus/Downloads/Video_Speech/' + actor))
            done_videos = 0
            for video  in os.listdir('/home/mateus/Downloads/Video_Speech/' + actor):
                done_videos += 1
                with open(output_folder + '/results-correct.csv', mode='a+') as file:
                    analyse('/home/mateus/Downloads/Video_Speech/'+actor+'/'+video, file)
                    percentage = round(((done_videos / total_videos_per_actor) * 100), 2)
                    print("Done {} of {} actors. Video {} of {}. ({}%)".format(current_actor, total_actors, done_videos, total_videos_per_actor, percentage))


    end = time.time()
    print("Classification finished. Total time: {}".format(end-start))