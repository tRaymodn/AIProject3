import csv
import random
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


def main():
    with open('ai4i2020.csv') as file:
        sampledData = []
        failure = []
        notFailure = []
        reader = csv.reader(file)
        for row in reader:
            if row[8] == "1":
                print(row)
                failure.append(row)
            else:
                notFailure.append(row)

        notFailure_sampled = random.sample(notFailure, len(failure))
        for row in failure:
            sampledData.append(row)
        for row in notFailure_sampled:
            sampledData.append(row)
        print("----------------------")
        split = train_test_split(sampledData, test_size=0.3, random_state=42, shuffle=True)
        print(len(split[0]))
        print(split[0])
        print(len(split[1]))
        print(split[1])

        randomForest(split[0], split[1])


def randomForest(trainingInput, testingInput):

    clf = RandomForestClassifier(max_depth=2, random_state=0)

    for row in trainingInput:
        row[1] = 0

        if row[2] == 'L':
            row[2] = 0.5
        elif row[2] == 'M':
            row[2] = 0.3
        elif row[2] == 'H':
            row[2] = 0.2

    y = []
    for row in trainingInput:
        y.append(row[8])

    clf.fit(trainingInput, y)

    for row in testingInput:
        row[1] = 0

        if row[2] == 'L':
            row[2] = 0.5
        elif row[2] == 'M':
            row[2] = 0.3
        elif row[2] == 'H':
            row[2] = 0.2


    y_pred = clf.predict(trainingInput)

    y_true = []
    for row in trainingInput:
        y_true.append(row[8])

    print(y_true)
    print(y_pred)

    print(sklearn.metrics.f1_score(y_true, y_pred, average='weighted'))


if __name__ == "__main__":
    main();
