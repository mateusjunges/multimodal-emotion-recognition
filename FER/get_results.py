from csv import reader

#  05      07      06     03    04     08        01
# 0         1       2      3    4      5         6         7             8
# angry, disgust, fear, happy, sad, surprise, neutral, image_name, correct_class
# (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)

emotions = {
    '01': 6,
    '02': 6,
    '03': 3,
    '04': 4,
    '05': 0,
    '06': 2,
    '07': 1,
    '08': 5
}

correct_predictions = 0
incorrect_predictions = 0
total_predictions = 0

with open('results/results.csv', mode='r') as file:

    content = reader(file)

    for row in content:
        correct_class = row[8]
        key = emotions.get(correct_class)
        prediction_for_correct_class = row[key]

        predictions = row[:7]

        key_max_value = predictions.index(max(predictions))

        if key_max_value == emotions.get(correct_class):
            correct_predictions += 1
        else:
            incorrect_predictions += 1

        total_predictions += 1


print("Total predictions = {}".format(total_predictions))
print("Correct predictions = {} - ({}%)".format(correct_predictions, round((correct_predictions / total_predictions) * 100, 2)))
print("Incorrect predictions = {} - ({}%)".format(incorrect_predictions, round((incorrect_predictions / total_predictions) * 100, 2)))