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
        #neuralNetwork(split[0], split[1])
        #naiveBayes(split[0], split[1])
        #randomForest(split[0], split[1])
        #supportVectorMachine(split[0], split[1])
        #adaBoost(split[0], split[1])
def neuralNetwork(training_data, testing_data):
    for row in training_data:  # Convert letters into numerical data
        row[1] = row[1][1:]
        if row[2] == "L":
            row[2] = 0.5
        elif row[2] == "M":
            row[2] = 0.3
        else:
            row[2] = 0.2

    trainingFloatList = []  # Turn training data into all numerical floating point values
    for row in training_data:
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
    for row in training_data:  # Create an array of ints correctly representing the machine failure for each data point
        y_true.append(row[8])

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=2000)
    clf.fit(trainingFloatList, y_true)

    y_pred = clf.predict(trainingFloatList)
    print("Prediction: ", y_pred)

    degree = np.array([1, 2, 3])
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

    testingIntList = []  # Turn training data into all numerical floating point values
    for row in testing_data:
        newRow = []
        for att in row:
            newRow.append(float(att))
        testingIntList.append(newRow)

    y_testing_true = []
    for row in testing_data:  # Create list of true labels values for testing data
        y_testing_true.append(row[8])

    clfNew = MLPClassifier(random_state=grid.best_estimator_.random_state,
                     tol=grid.best_estimator_.tol,
                     alpha=grid.best_estimator_.alpha)
    scores = cross_val_score(clfNew, testingIntList, y_testing_true, cv=5)
    print("5-fold cross validation scores on testing data: ", scores)
    f1 = sum(scores) / len(scores)
    print("Average cross validation score on testing data: ", f1)

    output = [['random_state', grid.best_estimator_.random_state],
              ['tol', grid.best_estimator_.tol],
              ['alpha', grid.best_estimator_.alpha],
              [f1]]
    return output

def supportVectorMachine(trainingData, testingData):
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

    y_testing_true = []
    for row in testingData:  # Creates a list of all the true labels for the testing data
        y_testing_true.append(float(row[8]))
    print(y_testing_true)

    clf = svm.SVC()
    clf.fit(trainingFloatList, training_true_y)
    pred = clf.predict(testingData)  # Make a prediction using the Support Vector Machine

    degree = np.array([1, 2, 3])
    tol = np.array([0.0001, 0.001, 0.01, 0.0005, 0.00005, 0.00001])
    gamma = np.array([0.0001, 0.00001, 0.000011, 0.000009])
    grid = GridSearchCV(estimator=clf, param_grid={'tol': tol, 'gamma': gamma, 'degree': degree})
    grid.fit(trainingFloatList, training_true_y)
    print(grid)
    # summarize the results of the grid search
    print("Best grid score: ", grid.best_score_)
    #print("Best estimator for coef0: ", grid.best_estimator_.coef0)
    #print("Best estimator for C: ", grid.best_estimator_.C)
    print("Best estimator for degree: ", grid.best_estimator_.degree)
    print("Best estimator for tolerance: ", grid.best_estimator_.tol)
    print("Best estimator for gamma: ", grid.best_estimator_.gamma)

    testingIntList = []  # Turn training data into all numerical floating point values
    for row in testingData:
        newRow = []
        for att in row:
            newRow.append(float(att))
        testingIntList.append(newRow)

    y_testing_true = []
    for row in testingData:  # Create list of true labels values for testing data
        y_testing_true.append(row[8])

    clfNew = svm.SVC(degree=grid.best_estimator_.degree,
                     tol=grid.best_estimator_.tol,
                     gamma=grid.best_estimator_.gamma)
    scores = cross_val_score(clfNew, testingIntList, y_testing_true, cv=5)
    print("5-fold cross validation scores on testing data: ", scores)
    f1 = sum(scores) / len(scores)
    print("Average cross validation score on testing data: ", f1)

    output = [['degree', grid.best_estimator_.degree],
              ['tolerance', grid.best_estimator_.tol],
              ['gamma', grid.best_estimator_.gamma],
              [f1]]
    return output


def randomForest(trainingInput, testingInput):

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

    estimators = np.array([100, 110, 90])
    maxDepth = np.array([2, 3, 5, 7])
    maxFeatures = np.array([2, 3, 5, 7])
    grid = GridSearchCV(estimator=clf, param_grid={'n_estimators': estimators, 'max_depth': maxDepth, 'max_features': maxFeatures})
    grid.fit(trainingInput, y_true)
    print(grid)
    # summarize the results of the grid search
    print("Best grid score: ", grid.best_score_)
    print("Best estimator for n_estimators: ", grid.best_estimator_.n_estimators)
    print("Best estimator for max_depth: ", grid.best_estimator_.max_depth)
    print("Best estimator for max_features: ", grid.best_estimator_.max_features)

    y_testing_true = []
    for row in testingInput:  # Create list of true labels values for testing data
        y_testing_true.append(row[8])

    clfNew = RandomForestClassifier(n_estimators=grid.best_estimator_.n_estimators,
                                    max_depth=grid.best_estimator_.max_depth,
                                    max_features=grid.best_estimator_.max_features)
    scores = cross_val_score(clfNew, testingInput, y_testing_true, cv=5)
    print("5-fold cross validation scores on testing data: ", scores)
    f1 = sum(scores) / len(scores)
    print("Average cross validation score on testing data: ", f1)

    output = [['n_estimators', grid.best_estimator_.n_estimators],
              ['max_depth', grid.best_estimator_.max_depth],
              ['max_features', grid.best_estimator_.max_features],
              [f1]]
    return output

def adaBoost(trainingData, testingData):
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
    print("Best estimator for random state: ", grid.best_estimator_.random_state)
    print("Best estimator for learning rate: ", grid.best_estimator_.learning_rate)

    testingIntList = []  # Turn training data into all numerical floating point values
    for row in testingData:
        newRow = []
        for att in row:
            newRow.append(float(att))
        testingIntList.append(newRow)

    y_testing_true = []
    for row in testingData:  # Create list of true labels values for testing data
        y_testing_true.append(row[8])

    clfNew = AdaBoostClassifier(learning_rate=grid.best_estimator_.learning_rate,
                                random_state=grid.best_estimator_.random_state,
                                n_estimators=grid.best_estimator_.n_estimators)
    scores = cross_val_score(clfNew, testingIntList, y_testing_true, cv=5)
    print("5-fold cross validation scores on testing data: ", scores)
    f1 = sum(scores) / len(scores)
    print("Average cross validation score on testing data: ", f1)

    output = [['learning_rate', grid.best_estimator_.learning_rate],
              ['n_estimators', grid.best_estimator_.n_estimators],
              ['random_state', grid.best_estimator_.random_state],
              [f1]]
    return output


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

    varSmoothing = np.array([0.00000000010, 0.00000001, 0.000000000001])
    grid = GridSearchCV(estimator=gnb,
                        param_grid={'var_smoothing': varSmoothing})
    grid.fit(intList, y_training_true)
    print(grid)
    # summarize the results of the grid search
    print("Best grid score: ", grid.best_score_)
    print("Best estimator for variable smoothing: ", grid.best_estimator_.var_smoothing)

    testingIntList = []  # Turn training data into all numerical floating point values
    for row in testingData:
        newRow = []
        for att in row:
            newRow.append(float(att))
        testingIntList.append(newRow)

    y_testing_true = []
    for row in testingData:  # Create list of true labels values for testing data
        y_testing_true.append(row[8])

    gnbNew = GaussianNB(var_smoothing=grid.best_estimator_.var_smoothing)
    scores = cross_val_score(gnbNew, testingIntList, y_testing_true, cv=5)
    print("5-fold cross validation scores on testing data: ", scores)
    f1 = sum(scores) / len(scores)
    print("Average cross validation score on testing data: ", f1)


    output = [['var_smoothing', grid.best_estimator_.var_smoothing], [f1]]
    return output


if __name__ == "__main__":
    main()
