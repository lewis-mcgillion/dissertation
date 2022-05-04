import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import linear_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import random
import matplotlib.pyplot as plt
import logging
import time

# disabling tensorflow error messages
# potential errors between keras and tensorflow versions??
tf.get_logger().setLevel(logging.ERROR)

random.seed(10)
np.random.seed(10)

#timer
start_time = time.time()

data = pd.read_csv("tdf_stages_classification.csv", encoding="ISO-8859-1")
#selecting relevant columns
wholeData = data[["Year", "Stage", "Type", "Rank", "Rider"]].dropna()
wholeData["Stage"] = pd.to_numeric(wholeData["Stage"])


#used to get testing
data = wholeData.loc[(data["Year"] == 2017)]
data = data[["Year", "Stage", "Type", "Rank", "Rider"]]

#used to find prev results
wholeData = wholeData[["Year", "Stage", "Type", "Rank", "Rider"]]

#stopping truncanation of output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

#gets previous results from stages where BOTH riders took part
def getPrevResults(data, rider1, rider2, year, stage):
    #getting results for specific rider including and before specified year
    previousRider1 = data.loc[(data['Rider'] == rider1) & (data['Year'] <= year)]
    previousRider2 = data.loc[(data["Rider"] == rider2) & (data['Year'] <= year)]

    #getting correct stages from current year
    currentYearRider1 = previousRider1.loc[(previousRider1["Stage"] <= stage) & (previousRider1["Year"] == year)]
    currentYearRider2 = previousRider2.loc[(previousRider2["Stage"] <= stage) & (previousRider2["Year"] == year)]

    #removing current year from previous results and then appending correct current year stages
    previousRider1 = previousRider1.loc[previousRider1['Year'] != year]
    previousRider2 = previousRider2.loc[previousRider2['Year'] != year]

    #appending correct current year results
    previousRider1 = previousRider1.append(currentYearRider1)
    previousRider2 = previousRider2.append(currentYearRider2)

    #the two arrays for previous results are now collected
    #inner join between year and stage to get only stages where both riders took part, in order to compare performances
    previousBothRiders = pd.merge(previousRider1, previousRider2, on=["Year", "Stage", "Type"])


    prevResults = previousBothRiders[["Year", "Stage", "Type"]]
    eachStageWinner = []

    #evaluating which rider finished first
    for i in previousBothRiders.index:
        #evaluates true if rider 1 wins
        if previousBothRiders['Rank_x'][i] < previousBothRiders['Rank_y'][i]:
            eachStageWinner.append(previousBothRiders['Rider_x'][i])
        else:
            eachStageWinner.append(previousBothRiders['Rider_y'][i])

    #adding the eachstagewinner array straight to the dataframe results in warning message
    #temp dataframe is created and concatenated instead
    tempdf = pd.DataFrame(eachStageWinner, columns=['firstFinisher'])

    #combine dataframes
    previousResults = pd.concat([prevResults, tempdf], axis=1)

    # converting y values (rider names) to binary values
    lb = preprocessing.LabelBinarizer()
    previousResults["firstFinisher"] = lb.fit_transform(previousResults["firstFinisher"])

    return previousResults


#model functions
def RandomForest(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=5, min_samples_leaf=4)
    #ravel changes y train from column vector to 1d array
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)

    #calculating and returning accuracy
    return accuracy_score(y_hat, y_test)

def NeuralNetwork(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Dense(5, input_dim=3, activation="softplus"))
    model.add(Dense(5, activation="softplus"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=250, batch_size=8, verbose=0)

    y_hat = np.round(model.predict(X_test))

    #calculating and returning accuracy
    return accuracy_score(y_hat, y_test)

def GradientBoosting(X_train, y_train, X_test, y_test):
    GB = GradientBoostingClassifier(learning_rate=0.01, min_samples_split=10, min_samples_leaf=4, max_depth=3, random_state=0)
    GB.fit(X_train, y_train)
    y_hat = GB.predict(X_test)
    return accuracy_score(y_hat, y_test)

def KNearestNeighbours(X_train, y_train, X_test, y_test):
    KNN = KNeighborsClassifier(n_neighbors=2)
    KNN.fit(X_train, y_train)
    y_hat = KNN.predict(X_test)
    return accuracy_score(y_hat, y_test)

def GuassianNaiveBayes(X_train, y_train, X_test, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_hat = gnb.predict(X_test)
    return accuracy_score(y_hat, y_test)

#arrays to store global accuracies
gbaccTotal = []
nnacctotal = []
rfacctotal = []
KNNacctotal = []
GNBacctotal = []

#stage 1 to 21
stages = range(1, 22)
#year

for i in stages:
    #current stage
    partData = data.loc[data["Stage"] == i]

    #shuffle
    dataArr = partData.sample(frac=1).values

    #arrays to store local accuracies
    rfAccuracies = []
    NNAccuracies = []
    gbAccuracies = []
    KNNaccuracies = []
    GNBaccuracies = []

    dataRange = len(dataArr) - 1

    #higher split number = less often predictions
    # range = length of data / split
    n_split = 5
    # only doing it for every nth in range, otherwise too computationally expensive
    for j in range(0, int(dataRange/n_split)):
        index = j*n_split
        #current and next rows in data array
        rider1 = dataArr[index]
        rider2 = dataArr[index + 1]

        #getting info from row
        rider1Name = rider1[4]
        rider2Name = rider2[4]
        year = rider1[0]
        stage = rider1[1]
        type = rider1[2]

        #gets both training and testing data
        trainTest = getPrevResults(wholeData, rider1Name, rider2Name, year, stage)

        # if this is the rider's first stage at the tour de france, there is no previous data to train on
        # therefore just skip this sample
        if len(trainTest)>1:
            xData = trainTest[["Year", "Stage", "Type"]]
            yData = trainTest["firstFinisher"].values

            #testing data is just the final row in the dataframes
            X_train = xData.iloc[0:-1].values
            X_test = xData.iloc[-1].values

            #X_test needs to be reshaped as it is only a single sample
            X_test = X_test.reshape(1, -1)

            y_train = yData[0:-1].ravel()
            y_test = yData[-1].ravel()

            #creating normalised train/test for use with neural net
            #output doesn't need to be normalised as it is already 0/1
            scaler = StandardScaler()
            scaler.fit(xData)
            xDataNorm = scaler.transform(xData)

            X_train_norm = xDataNorm[0:-1]
            X_test_norm = xDataNorm[-1]

            #reshape to 2D
            X_test_norm = np.reshape(X_test_norm, (-1, 3))

            # only making predictions when the training data contains more than 1 class
            # if there is only 1 class, that means one of the riders finished first in every sample
            # therefore there is no point trying to make predictions, as there is no evidence to point to the other rider finishing first
            if np.mean(y_train) % 1 != 0:

                #random forest
                rfAccuracy = RandomForest(X_train, y_train, X_test, y_test)
                rfAccuracies.append(rfAccuracy)

                #nerual network
                NNaccuracy = NeuralNetwork(X_train_norm, y_train, X_test_norm, y_test)
                NNAccuracies.append(NNaccuracy)

                #gradient boosting
                GBaccuracy = GradientBoosting(X_train, y_train, X_test, y_test)
                gbAccuracies.append(GBaccuracy)

                #k nearest neighbours
                KNNaccuracy = KNearestNeighbours(X_train, y_train, X_test, y_test)
                KNNaccuracies.append(KNNaccuracy)

                #guassian naive bayes
                GNBaccuracy = GuassianNaiveBayes(X_train, y_train, X_test, y_test)
                GNBaccuracies.append(GNBaccuracy)

    print("Accuracy for stage {} with Gradient Boosting: ".format(i))
    print(np.mean(gbAccuracies))
    gbaccTotal.append(np.mean(gbAccuracies))

    print("Accuracy for stage {} with Random Forest:".format(i))
    print(np.mean(rfAccuracies))
    rfacctotal.append(np.mean(rfAccuracies))

    print("Accuracy for stage {} with Neural Network:".format(i))
    print(np.mean(NNAccuracies))
    nnacctotal.append(np.mean(NNAccuracies))

    print("Accuracy for stage {} with KNN:".format(i))
    print(np.mean(KNNaccuracies))
    KNNacctotal.append(np.mean(KNNaccuracies))

    print("Accuracy for stage {} with Guassian Naive Bayes:".format(i))
    print(np.mean(GNBaccuracies))
    GNBacctotal.append(np.mean(GNBaccuracies))

print("\n\nAverage accuracy using Gradient Boosting:")
print(np.mean(gbaccTotal))

print("Average accuracy using Random Forest:")
print(np.mean(rfacctotal))

print("Average accuracy using Neural Network:")
print(np.mean(nnacctotal))

print("Average accuracy using KNN")
print(np.mean(KNNacctotal))

print("Average accuracy using GNB")
print(np.mean(GNBacctotal))

print("--- %s seconds ---" % (time.time() - start_time))

#dataframe of results
df = pd.DataFrame({'GB':gbaccTotal,
                   'RF': rfacctotal,
                   'KNN':KNNacctotal,
                   'GNB':GNBacctotal,
                   'ANN':nnacctotal})
#saving results
df.to_csv("head_to_head_results.csv")

#boxplot of results
df.boxplot()
plt.savefig('accuracies.png', format='png')
plt.show()

results = pd.read_csv("head_to_head_results.csv", encoding="ISO-8859-1")
ANNres = results["ANN"]
stages = np.array(list(range(1, 22)))
stages = stages.reshape(-1, 1)

regr = linear_model.LinearRegression()
regr.fit(stages, ANNres.values)
y_pred = regr.predict(stages)

plt.scatter(stages, ANNres.values)
plt.plot(stages, y_pred)
plt.show()