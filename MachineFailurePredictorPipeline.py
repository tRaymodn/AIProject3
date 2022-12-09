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
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate

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
        trainTable = []
        trainNN = trainNeuralNetwork(split[0])
        trainNNinst = ['Neural Network']
        trainNNinst.append(trainNN[0].get_params())
        trainNNinst.append(trainNN[1])
        trainTable.append(trainNNinst)

        trainsvm = trainSupportVectorMachine(split[0])
        trainsvminst = ['Support Vector Machine']
        trainsvminst.append(trainsvm[0].get_params())
        trainsvminst.append(trainsvm[1])
        trainTable.append(trainsvminst)

        trainrf = trainRandomForest(split[0])
        trainrfinst = ['Random Forest']
        trainrfinst.append(trainrf[0].get_params())
        trainrfinst.append(trainrf[1])
        trainTable.append(trainrfinst)

        trainada = trainAdaBoost(split[0])
        trainadainst = ['Ada Boost']
        trainadainst.append(trainada[0].get_params())
        trainadainst.append(trainada[1])
        trainTable.append(trainadainst)

        trainnb = trainNaiveBayes(split[0])
        trainnbinst = ['Naive Bayes']
        trainnbinst.append(trainnb[0].get_params())
        trainnbinst.append(trainnb[1])
        trainTable.append(trainnbinst)
        print(tabulate(trainTable, headers=["ML Model", "Best Parameter Set", "Training Data F1 Score with 5-Fold Validation"]))


        testTable = []
        testNN = tstNeuralNetwork(split[1], trainNN)
        testnninst = ['Neural Network']
        testnninst.append(testNN[0].get_params())
        testnninst.append(testNN[1])
        testTable.append(testnninst)

        testsvm = tstSupportVectorMachine(split[1], trainsvm)
        testsvminst = ['Support Vector Machine']
        testsvminst.append(testsvm[0].get_params())
        testsvminst.append(testsvm[1])
        testTable.append(testsvminst)

        testrf = tstRandomForest(split[1], trainrf)
        testrfinst = ['Random Forest']
        testrfinst.append(testrf[0].get_params())
        testrfinst.append(testrf[1])
        testTable.append(testrfinst)

        testada = tstAdaBoost(split[1], trainada)
        testadainst = ['Ada Boost']
        testadainst.append(testada[0].get_params())
        testadainst.append(testada[1])
        testTable.append(testadainst)

        testnb = tstNaiveBayes(split[1], trainnb)
        testnbinst = ['Naive Bayes']
        testnbinst.append(testnb[0].get_params())
        testnbinst.append(testnb[1])
        testTable.append(testnbinst)

        print(tabulate(testTable, headers=["ML Model", "Best Parameter Set", "Testing Data F1 Score"]))


def trainNeuralNetwork(trainingData):
    trainingFloatList = []  # Turn training data into all numerical floating point values
    for row in trainingData:
        newRow = []
        for att in row:
            if att == "L":
                newRow.append(0.5)
            elif att == "M":
                newRow.append(0.3)
            elif att == "H":
                newRow.append(0.2)
            else:
                newRow.append(float(att))
        trainingFloatList.append(newRow)
    y_true = []
    for row in trainingData:  # Create an array of ints correctly representing the machine failure for each data point
        y_true.append(row[8])

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=2000)
    clf.fit(trainingFloatList, y_true)

    randomState = np.array([9, 8, 10])
    tol = np.array([0.0001, 0.00011, 0.000010])
    alpha = np.array([101.46, 101.45, 101.49, 101.6])
    grid = GridSearchCV(estimator=clf, param_grid={'tol': tol, 'alpha': alpha, 'random_state': randomState})
    grid.fit(trainingFloatList, y_true)
    print(grid)
    # summarize the results of the grid search
    print("Best grid score: ", grid.best_score_)
    print("Best estimator for random state: ", grid.best_estimator_.random_state)
    print("Best estimator for tolerance: ", grid.best_estimator_.tol)
    print("Best estimator for alpha: ", grid.best_estimator_.alpha)

    bestEstimate = MLPClassifier(solver='lbfgs', random_state=grid.best_estimator_.random_state,
                                 tol=grid.best_estimator_.tol,
                                 alpha=grid.best_estimator_.alpha,
                                 hidden_layer_sizes=(5, 2),
                                 max_iter=2000)
    scores = cross_val_score(bestEstimate, trainingFloatList, y_true, cv=5)
    print("5-fold cross validation scores on training data: ", scores)
    f1 = sum(scores) / len(scores)
    print("Average cross validation score on training data: ", f1)

    output = [bestEstimate,
              f1]
    return output
def tstNeuralNetwork(testingData, modelParams):
    testingIntList = []  # Turn training data into all numerical floating point values
    for row in testingData:
        newRow = []
        for att in row:
            newRow.append(float(att))
        testingIntList.append(newRow)

    y_testing_true = []
    for row in testingData:  # Create list of true labels values for testing data
        y_testing_true.append(row[8])

    clfNew = modelParams[0]
    clfNew.fit(testingData, y_testing_true)
    pred = clfNew.predict(testingIntList)
    f1 = f1_score(y_testing_true, pred, pos_label="1")
    print(pred)

    output = modelParams
    output.pop()
    output.append(f1)
    return output


def trainSupportVectorMachine(trainingData):
    training_true_y = []
    for row in trainingData:  # Creating a table of true labels for the training data
        training_true_y.append(row[8])

    trainingFloatList = []  # Turn training data into all numerical floating point values
    for row in trainingData:
        newRow = []
        for att in row:
            if att == "L":
                newRow.append(0.5)
            elif att == "M":
                newRow.append(0.3)
            elif att == "H":
                newRow.append(0.2)
            else:
                newRow.append(float(att))
        trainingFloatList.append(newRow)

    clf = svm.SVC()
    clf.fit(trainingFloatList, training_true_y)

    degree = np.array([1, 2, 3])
    tol = np.array([0.0001, 0.001, 0.01, 0.0005, 0.00005, 0.00001])
    gamma = np.array([0.0001, 0.00001, 0.000011, 0.000009])
    grid = GridSearchCV(estimator=clf, param_grid={'tol': tol, 'gamma': gamma, 'degree': degree})
    grid.fit(trainingFloatList, training_true_y)
    print(grid)
    # summarize the results of the grid search
    print("Best grid score: ", grid.best_score_)
    print("Best estimator for degree: ", grid.best_estimator_.degree)
    print("Best estimator for tol: ", grid.best_estimator_.tol)
    print("Best estimator for gamma: ", grid.best_estimator_.gamma)

    bestEstimate = svm.SVC(degree=grid.best_estimator_.degree,
                       tol=grid.best_estimator_.tol,
                       gamma=grid.best_estimator_.gamma,
                       )
    scores = cross_val_score(bestEstimate, trainingFloatList, training_true_y, cv=5)
    f1 = sum(scores) / len(scores)
    print("Average cross validation score on training data: ", f1)

    output = [bestEstimate,
              f1]
    return output


def tstSupportVectorMachine(testingData, modelParams):
    testingIntList = []  # Turn training data into all numerical floating point values
    for row in testingData:
        newRow = []
        for att in row:
            newRow.append(float(att))
        testingIntList.append(newRow)

    y_testing_true = []
    for row in testingData:  # Create list of true labels values for testing data
        y_testing_true.append(row[8])

    clfNew = modelParams[0]
    clfNew.fit(testingIntList, y_testing_true)
    pred = clfNew.predict(testingIntList)
    f1 = f1_score(y_testing_true, pred, pos_label="1")
    print(pred)

    output = modelParams
    output.pop()
    output.append(f1)
    return output

def trainRandomForest(trainingInput):

    clf = RandomForestClassifier()

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
    estimators = np.array([100, 110, 90])
    maxDepth = np.array([2, 3, 5, 7])
    maxFeatures = np.array([2, 3, 5, 7])
    grid = GridSearchCV(estimator=clf,
                        param_grid={'n_estimators': estimators, 'max_depth': maxDepth, 'max_features': maxFeatures})

    grid.fit(trainingInput, y)
    print(grid)
    # summarize the results of the grid search
    print("Best grid score: ", grid.best_score_)
    print("Best estimator for n_estimators: ", grid.best_estimator_.n_estimators)
    print("Best estimator for max_depth: ", grid.best_estimator_.max_depth)
    print("Best estimator for max_features: ", grid.best_estimator_.max_features)

    bestEstimate = RandomForestClassifier(n_estimators=grid.best_estimator_.n_estimators,
                                          max_depth=grid.best_estimator_.max_depth,
                                          max_features=grid.best_estimator_.max_features,
                           )
    scores = cross_val_score(bestEstimate, trainingInput, y, cv=5)
    f1 = sum(scores) / len(scores)
    print("Average cross validation score on training data: ", f1)

    output = [bestEstimate,
              f1]
    return output


def tstRandomForest(testingData, modelParams):
    for row in testingData:
        row[1] = 0

        if row[2] == 'L':
            row[2] = 0.5
        elif row[2] == 'M':
            row[2] = 0.3
        elif row[2] == 'H':
            row[2] = 0.2

    y_testing_true = []
    for row in testingData:  # Create list of true labels values for testing data
        y_testing_true.append(row[8])

    clfNew = modelParams[0]
    clfNew.fit(testingData, y_testing_true)
    pred = clfNew.predict(testingData)
    f1 = f1_score(y_testing_true, pred, pos_label="1")
    print(pred)

    output = modelParams
    output.pop()
    output.append(f1)
    return output


def trainAdaBoost(trainingData):
    y_training_true = []
    for row in trainingData:  # Create list of true labels values for training data
        y_training_true.append(row[8])
    print(y_training_true)

    clf = AdaBoostClassifier()
    clf.fit(trainingData, y_training_true)
    pred = clf.predict(trainingData)
    print(pred)

    learningRate = np.array([0.2, 0.5, 1, 2, 3])
    estimators = np.array([10, 9, 5, 6])
    randomState = np.array([2, 3, 5, 7, 1, 0])
    grid = GridSearchCV(estimator=clf,
                        param_grid={'n_estimators': estimators, 'random_state': randomState, 'learning_rate': learningRate})
    grid.fit(trainingData, y_training_true)
    print(grid)
    # summarize the results of the grid search
    print("Best grid score: ", grid.best_score_)
    print("Best estimator for n_estimators: ", grid.best_estimator_.n_estimators)
    print("Best estimator for random_state: ", grid.best_estimator_.random_state)
    print("Best estimator for learning_rate: ", grid.best_estimator_.learning_rate)

    bestEstimate = AdaBoostClassifier(n_estimators=grid.best_estimator_.n_estimators,
                           random_state=grid.best_estimator_.random_state,
                           learning_rate=grid.best_estimator_.learning_rate,
                           )
    scores = cross_val_score(bestEstimate, trainingData, y_training_true, cv=5)
    f1 = sum(scores) / len(scores)
    print("Average cross validation score on training data: ", f1)

    output = [bestEstimate,
              f1]
    return output


def tstAdaBoost(testingData, modelParams):
    testingIntList = []  # Turn training data into all numerical floating point values
    for row in testingData:
        newRow = []
        for att in row:
            newRow.append(float(att))
        testingIntList.append(newRow)

    y_testing_true = []
    for row in testingData:  # Create list of true labels values for testing data
        y_testing_true.append(row[8])

    clfNew = modelParams[0]
    clfNew.fit(testingIntList, y_testing_true)
    pred = clfNew.predict(testingIntList)
    f1 = f1_score(y_testing_true, pred, pos_label="1")
    print(pred)

    output = modelParams
    output.pop()
    output.append(f1)
    return output
def trainNaiveBayes(trainingData):
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

    varSmoothing = np.array([0.00000000010, 0.00000001, 0.000000000001])
    grid = GridSearchCV(estimator=gnb,
                        param_grid={'var_smoothing': varSmoothing})
    grid.fit(intList, y_training_true)
    print(grid)
    # summarize the results of the grid search
    print("Best grid score: ", grid.best_score_)
    print("Best estimator for var_smoothing: ", grid.best_estimator_.var_smoothing)

    bestEstimate = GaussianNB(var_smoothing=grid.best_estimator_.var_smoothing)
    scores = cross_val_score(bestEstimate, intList, y_training_true, cv=5)
    f1 = sum(scores) / len(scores)
    print("Average cross validation score on training data: ", f1)

    output = [bestEstimate,
              f1]
    return output


def tstNaiveBayes(testingData, modelParams):
    testingIntList = []  # Turn training data into all numerical floating point values
    for row in testingData:
        newRow = []
        for att in row:
            newRow.append(float(att))
        testingIntList.append(newRow)

    y_testing_true = []
    for row in testingData:  # Create list of true labels values for testing data
        y_testing_true.append(row[8])

    gnbNew = modelParams[0]
    gnbNew.fit(testingIntList, y_testing_true)
    pred = gnbNew.predict(testingIntList)
    f1 = f1_score(y_testing_true, pred, pos_label="1")
    print(pred)

    output = modelParams
    output.pop()
    output.append(f1)
    return output

if __name__ == "__main__":
    main()
