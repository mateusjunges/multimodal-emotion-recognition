from csv import reader

emotions = {
    '01': 'neutral',
    '02': 'neutral',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

correct_predictions_sum = 0
correct_predictions_product = 0
incorrect_predictions_sum = 0
incorrect_predictions_product = 0
total_predictions = 0

with open('/home/mateus/Documents/tcc/tcc-v2/FER_SER_fusion/results-correct.csv', mode='r') as file:

    content = reader(file)

    for row in content:
        filename = row[0]
        correct_class = filename[6:8]
        emotion = emotions.get(correct_class)

        # 0 01 - 01 - 03 - 02 - 01 - 01 - 21, - filename
        # 1 0.023219605284854138, s_angry
        # 2 0.001845027220878226, s_disgust
        # 3 0.03573768251602862, s_fear
        # 4 1.685802388898883, s_happy
        # 5 0.1698020594004547, s_neutral
        # 6 0.03874191665084039, s_sad
        # 7 0.044851281764548646, s_surprise
        # 8 3.8244897411760155e-06, p_angry
        # 9 3.703870002729608e-09, p_disgust
        # 10 4.996678691510553e-06, p_fear
        # 11 0.6859017698584774,p_happy
        # 12 3.0939170821766204e-08, p_neutral
        # 13 2.0472588278789776e-07, p_sad
        # 14 1.2711826554913103e-07, p_surprise
        # 15 happy, - sum
        # 16 happy - product

        emotion_sum = row[15]
        emotion_product = row[16]

        if emotion_sum == emotion:
            correct_predictions_sum += 1
        else:
            incorrect_predictions_sum += 1

        if emotion_product == emotion:
            correct_predictions_product += 1
        else:
            incorrect_predictions_product += 1

        total_predictions += 1


print("Total predictions: {}".format(total_predictions))
print("--------------------------------------------------------")
print("Correct predictions using 'sum': {} of {} ({}%)".format(correct_predictions_sum, total_predictions, round((correct_predictions_sum / total_predictions) * 100), 2))
print("Incorrect predictions using sum: {} of {} ({}%)".format(incorrect_predictions_sum, total_predictions, round((incorrect_predictions_sum / total_predictions) * 100), 2))
print("--------------------------------------------------------")
print("Correct predictions using 'product': {} of {} ({}%)".format(correct_predictions_product, total_predictions, round((correct_predictions_product / total_predictions) * 100), 2))
print("Incorrect predictions using 'product': {} of {} ({}%)".format(incorrect_predictions_product, total_predictions, round((incorrect_predictions_product / total_predictions) * 100), 2))