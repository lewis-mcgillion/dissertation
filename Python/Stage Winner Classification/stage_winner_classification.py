import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import GradientBoostingClassifier
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
import logging

# disabling tensorflow error messages
# potential errors between keras and tensorflow versions??
tf.get_logger().setLevel(logging.ERROR)

random.seed(10)
np.random.seed(10)

#getting the training and testing data
def getTrainTest(data, year, stage, type, finishRange=10, encode=False):
    #previous years data and data from current year
    prevData = data.loc[(data["Year"] >= (year - 2)) & (data["Year"] < year) & (data["rank"]<=finishRange) & (data["Type"]==type)]
    currYearData = data.loc[(data["Year"]==year) & (data["Stage"]<stage)& (data["rank"]<=finishRange) & (data["Type"]==type)]

    #merge
    train = pd.concat([prevData, currYearData])

    #get test
    test = data.loc[(data["Year"]==year) & (data["Stage"]==stage) & (data["rank"]<=finishRange)]

    #split to x and y
    X_train = train[["Year", "Stage", "Type", "rank"]]
    X_train = X_train.sample(frac=1)

    #get y train
    y_train = train["rider"]

    #get x test and shuffle
    X_test = test[["Year", "Stage", "Type", "rank"]]
    X_test = X_test.sample(frac=1)

    #get y test
    y_test = test["rider"]


    if encode:
        #onehotencode ytrain and ytest
        enc = OneHotEncoder()
        enc.fit(pd.concat([y_train, y_test]).values.reshape(-1, 1))

        #encode y
        y_train = enc.transform(train[["rider"]]).toarray()
        y_test = enc.transform(test[["rider"]]).toarray()

        #return encoded variables and encoder object to decode later
        return y_train, y_test, enc

    return X_train, X_test, y_train, y_test

#returns average accuracy
def Eval_strict(yhat, y_test):
    #if the prediction is somewhere else in predicted range
    correct = 0
    for i, j in enumerate(yhat):
        if yhat[i] == y_test[i]:
            correct +=1

    accuracy = correct / len(yhat)
    return accuracy

#evaluation function calculates a prediction as correct if it is somewhere within the actual y values
def Eval(yhat, y_test):
    inPredRange = 0
    for i, j in enumerate(yhat):
        if yhat[i] in y_test:
            inPredRange +=1

    inPredRange /= len(yhat)
    return inPredRange

# #functions for each model to be called from main code
def NeuralNetwork(X_train, y_train, X_test):
    model = Sequential()
    model.add(Dense(500, input_dim=4, activation="relu"))
    model.add(Dense(500, activation="relu"))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    #compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=500, batch_size=16, verbose=0)
    y_hat = model.predict(X_test)
    return y_hat

def RandomForest(X_train, y_train, X_test):
    clf = RandomForestClassifier(n_estimators=100)
    #ravel changes y train from column vector to 1d array
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    return y_hat

#learning_rate=0.1, max_depth=20,
def GradientBoosting(X_train, y_train, X_test):
    GB = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=0)
    GB.fit(X_train, y_train)
    y_hat = GB.predict(X_test)
    return y_hat

def KNearestNeighbours(X_train, y_train, X_test):
    KNN = KNeighborsClassifier(n_neighbors=1)
    KNN.fit(X_train, y_train)
    y_hat = KNN.predict(X_test)
    return y_hat

def GuassianNaiveBayes(X_train, y_train, X_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_hat = gnb.predict(X_test)
    return y_hat


#importing data
data = pd.read_csv("tdf_stages_classification.csv", encoding="ISO-8859-1")
#selecting relevant columns
data = data[["Year", "Stage", "Type", "rank", "rider"]].dropna()


#testing models to ensure they work on training data

print("Training and testing each model on the same data to ensure they converge correctly")
print("Accuracy should be close to 100%")
year, stage = 2016, 1
#getting the current stage to be predicted and the type of stage it is
current = data.loc[(data["Year"] == year) & (data["Stage"] == stage) & (data["rank"] == 1)]
type = current["Type"].values[0]

#finish range is the number of places to be predicted
finishRange = 10
X_train, X_test, y_train, y_test = getTrainTest(data, year, stage, type, finishRange=finishRange, encode=False)
xData = pd.concat([X_train, X_test])
X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values

y_train_enc, y_test_enc, enc = getTrainTest(data, year, stage, type, finishRange=finishRange, encode=True)

#scaler for neural network
scaler = StandardScaler()
scaler.fit(xData)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)

ANN_y_hat_enc = NeuralNetwork(X_train_norm, y_train_enc, X_train_norm)
#reverse onehotencoding
ANN_y_hat = enc.inverse_transform(ANN_y_hat_enc)
print("Neural Network")
print(Eval_strict(ANN_y_hat, y_train))

RF_y_hat = RandomForest(X_train, y_train, X_train)
print("Random Forest")
print(Eval_strict(RF_y_hat, y_train))

GB_y_hat = GradientBoosting(X_train, y_train, X_train)
print("Gradient Boosting")
print(Eval_strict(GB_y_hat, y_train))

KNN_y_hat = KNearestNeighbours(X_train, y_train, X_train)
print("KNN")
print(Eval_strict(KNN_y_hat, y_train))

GNB_y_hat = GuassianNaiveBayes(X_train, y_train, X_train)
print("GNB")
print(Eval_strict(GNB_y_hat, y_train))

year = 2016
stages = range(1, 22)

ANNaccuracies = []
RFaccuracies = []
GBaccuracies = []
KNNaccuracies = []

for i in stages:
    print("Stage", i)

    #getting information about current stage
    current = data.loc[(data["Year"] == year) & (data["Stage"] == stage) & (data["rank"] == 1)]
    type = current["Type"].values[0]

    #getting train and test
    X_train, X_test, y_train, y_test = getTrainTest(data, year, i, type, finishRange=finishRange, encode=False)
    xData = pd.concat([X_train, X_test])
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values

    #encoded output data
    y_train_enc, y_test_enc, enc = getTrainTest(data, year, i, type, finishRange=finishRange, encode=True)

    #scaling input data
    scaler = StandardScaler()
    scaler.fit(xData)
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    ANN_y_hat_enc = NeuralNetwork(X_train_norm, y_train_enc, X_test_norm)
    ANN_y_hat = enc.inverse_transform(ANN_y_hat_enc)
    ANN_accuracy = Eval(ANN_y_hat, y_test)
    print(ANN_y_hat)
    print(ANN_accuracy)
    ANNaccuracies.append(ANN_accuracy)

    RF_y_hat = RandomForest(X_train, y_train, X_test)
    print("Random Forest")
    RFaccuracy = Eval(RF_y_hat, y_test)
    print(RF_y_hat)
    print(RFaccuracy)
    RFaccuracies.append(RFaccuracy)

    GB_y_hat = GradientBoosting(X_train, y_train, X_test)
    print("Gradient Boosting")
    GBaccuracy = Eval(GB_y_hat, y_test)
    print(GB_y_hat)
    print(GBaccuracy)
    GBaccuracies.append(GBaccuracy)

    KNN_y_hat = KNearestNeighbours(X_train, y_train, X_test)
    print("KNN")
    KNNaccuracy = Eval(KNN_y_hat, y_test)
    print(KNN_y_hat)
    print(KNNaccuracy)
    KNNaccuracies.append(KNNaccuracy)

print("Average Random Forest Accuracy:")
print(np.mean(RFaccuracies))

print("Average Neural Network Accuracy:")
print(np.mean(ANNaccuracies))

print("Average Gradient Boosting Accuracy:")
print(np.mean(GBaccuracies))

print("Average K Nearest Neighbours Accuracy:")
print(np.mean(KNNaccuracies))