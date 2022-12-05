import csv
import random
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

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
        ##neuralNetwork(split[0], split[1])
        naiveBayes(split[0], split[1])
        ##randomForest(split[0], split[1])
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


def naiveBayes(trainingData, testingData):
    intList = []
    for row in trainingData:
        newRow = []
        for att in row:
            newRow.append(float(att))
        intList.append(newRow)
    gnb = GaussianNB()
    y_training_true = []
    for row in intList:
        y_training_true.append(row[8])
    gnb.fit(intList, y_training_true)
    testingIntList = []
    for row in testingData:
        newRow = []
        for att in row:
            newRow.append(float(att))
        testingIntList.append(newRow)
    y_pred = gnb.predict(testingIntList)

    y_testing_true = []
    pointsMislabeled = 0
    for row in testingData:
        y_testing_true.append(float(row[8]))

    i = 0
    while i < len(y_testing_true):
        if y_testing_true[i] != y_pred[i]:
            pointsMislabeled += 1
        i += 1
    print("prediction:", y_pred)
    print("true values are: ", y_testing_true)
    print("Number of mislabeled points out of a total %d points : %d" % (len(testingData), pointsMislabeled))
    print("Mislabeled point percentage: %", (pointsMislabeled/len(testingData)*100))


if __name__ == "__main__":
    main()
