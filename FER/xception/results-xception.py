import os
import sys
import argparse
from csv import reader


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_results', type=str, help='Path to csv results')
    return parser.parse_args(argv)


def main(args):
    csv_path = args.csv_results

    correct_predictions = 0
    incorrect_predictions = 0
    total_predictions = 0

    for csv in sorted(os.listdir(csv_path)):
        with open(csv_path + '/' + csv, mode='r') as file:

            content = reader(file, delimiter=',')
            next(content)

            for row in content:
                #   0      1       2      3    4      5        6      7     8        9             10
                # Angry,Disgust,Fearful,Happy,Sad,Surprised,Neutral,Video,Frame,Correct_Class,Predicted_Class

                correct_class = row[9]
                predicted_class = row[10]

                if correct_class == 'Calm' and predicted_class == 'Neutral':
                    correct_predictions += 1
                elif correct_class == predicted_class:
                    correct_predictions += 1
                else:
                    incorrect_predictions += 1
                total_predictions += 1

    print("Correct predictions: {}".format(correct_predictions))
    print("Incorrect predictions: {}".format(incorrect_predictions))
    print("Total predictions: {}".format(total_predictions))

    print("Final result: {}%".format(round((correct_predictions / total_predictions) * 100, 2)))


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))