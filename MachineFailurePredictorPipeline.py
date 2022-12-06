import csv
import random
import numpy as np
import sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

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
        ##naiveBayes(split[0], split[1])
        ##randomForest(split[0], split[1])
        ##supportVectorMachine(split[0], split[1])
        adaBoost(split[0], split[1])
def neuralNetwork(training_data, testing_data):
    for row in training_data:  # Convert letters into numerical data
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
    for row in testing_data:  # Convert all values in testing data into numerical, floating point numbers
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

def supportVectorMachine(trainingData, testingData):
    training_true_y = []
    for row in trainingData:  # Creating a table of true labels for the training data
        training_true_y.append(row[8])

    trainingFloatList = []  # Turn training data into all numerical floating point values
    for row in trainingData:
        newRow = []
        for att in row:
            newRow.append(float(att))
        trainingFloatList.append(newRow)

    y_testing_true = []
    for row in testingData:  # Creates a list of all the true labels for the testing data
        y_testing_true.append(float(row[8]))
    print(y_testing_true)

    clf = svm.SVC()
    clf.fit(trainingFloatList, training_true_y)
    pred = clf.predict(testingData)  # Make a prediction using the Support Vector Machine

    predFloat = []
    for value in pred:  # Transform the data from the prediction into floating point values
        predFloat.append(float(value))
    print(predFloat)


    i = 0
    mislabeledPoints = 0
    while i < len(y_testing_true):  # Count the number of mislabeled points in pred
        if y_testing_true[i] != predFloat[i]:
            mislabeledPoints += 1
        i += 1
    print("Total points: ", len(pred))
    print("Number of mislabeled points: ", mislabeledPoints)
    print("Mislabel percentage: ", (mislabeledPoints/len(pred)*100))
    """# get support vectors
    print(clf.support_vectors_)
    
    # get indices of support vectors
    print(clf.support_)
    # get number of support vectors for each class
    print(clf.n_support_)"""




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

def adaBoost(trainingData, testingData):
    y_training_true = []
    for row in trainingData:  # Create list of true labels values for training data
        y_training_true.append(row[8])

    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(trainingData, y_training_true)
    pred = clf.predict(testingData)
    print(pred)

    predFloat = []
    j = 0
    while j < len(pred):  # Transforms the prediction list into a list of floating point numbers
        predFloat.append(float(pred[j]))
        j += 1

    y_testing_true = []
    for row in testingData:  # Creates a list of all the true labels for the testing data
        y_testing_true.append(float(row[8]))
    print(y_testing_true)

    pointsMislabeled = 0
    i = 0
    while i < len(pred):  # Compares the predicted labels to the real labels and finds the number of mislabeled points
        if predFloat[i] != y_testing_true[i]:
            pointsMislabeled += 1
        i += 1

    print("There were ", len(pred), "data points")
    print("Number of points mislabeled: ", pointsMislabeled)
    print("Percentage of mislabeled points: ", (pointsMislabeled/len(pred))*100)
def naiveBayes(trainingData, testingData):
    intList = []  # Turn training data into all numerical floating point values
    for row in trainingData:
        newRow = []
        for att in row:
            newRow.append(float(att))
        intList.append(newRow)

    gnb = GaussianNB()  # Create Gaussian Naive Bayes object

    y_training_true = []
    for row in intList:  # Create list of true labels values for training data
        y_training_true.append(row[8])

    gnb.fit(intList, y_training_true)  # Fit the training data to the labels according to Gaussian Naive Bayes

    testingIntList = []
    for row in testingData:  # Transform all values in testingData to floating point numbers
        newRow = []
        for att in row:
            newRow.append(float(att))
        testingIntList.append(newRow)

    y_pred = gnb.predict(testingIntList)  # Use GNB to make a prediction for the classifiers of testingData

    y_testing_true = []
    for row in testingData:  # Creates a list of all the true labels for the testing data
        y_testing_true.append(float(row[8]))

    pointsMislabeled = 0
    i = 0
    while i < len(y_testing_true):  # Counts how many data points that the prediction got wrong
        if y_testing_true[i] != y_pred[i]:
            pointsMislabeled += 1
        i += 1

    print("prediction:", y_pred)
    print("true values are: ", y_testing_true)
    print("Number of mislabeled points out of a total %d points : %d" % (len(testingData), pointsMislabeled))
    print("Mislabeled point percentage: %", (pointsMislabeled/len(testingData)*100))


if __name__ == "__main__":
    main()
