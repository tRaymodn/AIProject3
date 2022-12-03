import csv
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

def main():
    with open('ai4i2020.csv') as file:
        sampledData = []
        failure = []
        notFailure = []
        reader = csv.reader(file)
        for row in reader:
            row[1] = row[1][1:]
            if row[2] == "L":
                row[2] = 0.5
            elif row[2] == "M":
                row[2] = 0.3
            else:
                row[2] = 0.2
            if row[8] == "1":
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
        print(split[0])  # 70% training data
        print(len(split[1]))
        print(split[1])  # 30% testing data
        neuralNetwork(split[0], split[1])

def neuralNetwork(training_data, testing_data):
    for row in training_data:
        row[1] = row[1][1:]
        if row[2] == "L":
            row[2] = 0.5
        elif row[2] == "M":
            row[2] = 0.3
        else:
            row[2] = 0.2
    y = []
    for row in training_data:
        y.append(row[8])

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(training_data, y)

    intList = []
    for row in testing_data:
        newRow = []
        for att in row:
            newRow.append(float(att))
        intList.append(newRow)
    print(intList)
    y_true = []
    for row in testing_data:  # Create an array of ints correctly representing the machine failure for each data point
        y_true.append(row[8])

    y_pred = clf.predict(intList)
    print(y_pred)

    scores = cross_val_score(clf, intList, y_true, cv=5)
    print(scores)

if __name__ == "__main__":
    main()
