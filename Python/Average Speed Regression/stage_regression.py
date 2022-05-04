import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf
import logging

# disabling tensorflow error messages
# potential errors between keras and tensorflow versions??
tf.get_logger().setLevel(logging.ERROR)

#setting random seeds
random.seed(10)
np.random.seed(10)

def get_box_plot_data(labels, bp):
    rows_list = []

    for i in range(len(labels)):
        dict1 = {}
        dict1['label'] = labels[i]
        dict1['lower_whisker'] = bp['whiskers'][i*2].get_ydata()[1]
        dict1['lower_quartile'] = bp['boxes'][i].get_ydata()[1]
        dict1['median'] = bp['medians'][i].get_ydata()[1]
        dict1['upper_quartile'] = bp['boxes'][i].get_ydata()[2]
        dict1['upper_whisker'] = bp['whiskers'][(i*2)+1].get_ydata()[1]
        rows_list.append(dict1)

    return pd.DataFrame(rows_list)

data = pd.read_csv('tdf_stages_weather.csv', encoding="ISO-8859-1")

#only using relevant columns of the data
cleandata = data[["Year", "Distance", "Type", "Average_speed" ,"Temperature", "Precipitation", "Humidity", "Wind Speed","Relative Wind Speed"]]

#stopping truncanation of output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

#calculating correlation between all features
corrMatrix = cleandata.corr()
corrMatrix = corrMatrix[["Average_speed"]]

#only average speed column
print("Correlation Matrix:")
print(corrMatrix["Average_speed"])

#using different dataset as weather is irrelevant, this set contains more stages as well
data = pd.read_csv('tdf_stages_regression.csv', encoding="ISO-8859-1")
data = pd.read_csv('tdf_stages_weather.csv', encoding="ISO-8859-1")

#shuffling data
cleandata = data.sample(frac=1)

#only these features have a correlation to average speed so only they will be used
xData = cleandata[['Year', 'Distance', 'Type']].values
yData = cleandata['Average_speed'].values

#scaling data
scaler = StandardScaler()
scaler.fit(xData)
xDataNorm = scaler.transform(xData)

folds = 10
seed = 10
kfold = KFold(n_splits=folds, random_state=seed)


#neural network
def model():
    model = Sequential()
    model.add(Dense(500, input_dim=3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    #compile model
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

#test model to plot epochs
#to identify how many epochs are required to converge
testModel = model()
history = testModel.fit(xDataNorm, yData, validation_split=0.33, epochs=500, batch_size=16)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.show()

ANN = KerasRegressor(build_fn=model, epochs=250, batch_size=16, verbose=0)
ANNResults = cross_val_predict(ANN, xDataNorm, yData, cv=kfold)
ANNMAE = np.mean(abs(ANNResults - yData))
print("Artificial Neural Network MAE:", ANNMAE)

#Linear regression
lin = LinearRegression()
linResults = cross_val_predict(lin, xData, yData, cv=kfold)
linMAE = np.mean(abs(linResults - yData))
print("Linear Regression MAE:", linMAE)

#Ridge Regression
ridge = linear_model.Ridge(alpha=1)
ridgeResults = cross_val_predict(ridge, xData, yData, cv=kfold)
ridgeMAE = np.mean(abs(ridgeResults - yData))
print("Ridge Regression MAE:", ridgeMAE)

#Lasso regression
lasso = linear_model.Lasso(alpha=1)
lassoResults = cross_val_predict(lasso, xData, yData, cv=kfold)
lassoMAE = np.mean(abs(lassoResults - yData))
print("Lasso Regression MAE:", lassoMAE)

#Gradient Boosting
gradientBoost = GradientBoostingRegressor(random_state=1)
gradientBoostResults = cross_val_predict(gradientBoost, xData, yData, cv=kfold)
GBMAE = np.mean(abs(gradientBoostResults - yData))
print("Gradient Boosting MAE:", GBMAE)

#Decision Tree
dTree = tree.DecisionTreeRegressor()
dTreeResults = cross_val_predict(dTree, xData, yData, cv=kfold)
dTreeMAE = np.mean(abs(dTreeResults - yData))
print("Decision Tree MAE:", dTreeMAE)

#dataframe of results
df = pd.DataFrame({'Neural Network': [ANNMAE],
                    'Linear Regression': [linMAE],
                   'Ridge Regression': [ridgeMAE],
                   'Lasso Regression': [lassoMAE],
                   'Gradient Boosting': [GBMAE],
                   'Decision Tree': [dTreeMAE]})

#saving results
df.to_csv("average_speed_regression.csv")

#boxplot of each model
results = [abs(ANNResults - yData), abs(linResults - yData), abs(ridgeResults - yData), abs(lassoResults - yData), abs(gradientBoostResults - yData), abs(dTreeResults - yData)]
fig, ax = plt.subplots()
plot = ax.boxplot(results, labels=['ANN', 'Linear', 'Ridge', 'Lasso', 'GB', 'DT'])
plt.show()

print(get_box_plot_data(['ANN', 'Linear', 'Ridge', 'Lasso', 'GB', 'DT'], plot))


#further testing with Neural Network
#for each type of stage
print("Further testing with Neural Network")

X = cleandata[['Year', 'Distance', 'Type']].values
y = cleandata['Average_speed'].values

#global results for all predictions
globalRes = pd.DataFrame(columns=["Year", "Distance", "Type", "Error"])

#using k fold cross validation
for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    X_train_norm, X_test_norm = scaler.transform(X_train), scaler.transform(X_test)
    y_train, y_test = y[train_index], y[test_index]

    model = Sequential()
    model.add(Dense(500, input_dim=3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # compile model
    model.compile(loss='mean_absolute_error', optimizer='adam')
    model.fit(X_train_norm, y_train, epochs=250, batch_size=16, verbose=0)
    ANN_y_hat = model.predict(X_test_norm)

    y_test = y_test.reshape(y_test.shape[0], -1)

    errors = abs(ANN_y_hat - y_test)

    results = pd.DataFrame(data=X_test, columns=["Year", "Distance", "Type"])
    results["Error"] = errors

    globalRes = globalRes.append(results)

TT = globalRes.loc[globalRes["Type"] == 0]
flat = globalRes.loc[globalRes["Type"] == 1]
hilly = globalRes.loc[globalRes["Type"] == 2]
mountain = globalRes.loc[globalRes["Type"] == 3]

print("TT error",np.mean(TT["Error"].values))
print("Flat stage error",np.mean(flat["Error"].values))
print("Hilly stage error",np.mean(hilly["Error"].values))
print("Mountain stage error",np.mean(mountain["Error"].values))


allErrors = [TT["Error"].values, flat["Error"].values, hilly["Error"].values, mountain["Error"].values]
fig, ax = plt.subplots()
plot = ax.boxplot(allErrors, labels = ['TT', 'Flat', 'Hilly', 'Mountain'])
plt.show()

print(get_box_plot_data(['TT', 'Flat', 'Hilly', 'Mountain'], plot))

TT_average_speed = cleandata.loc[cleandata["Type"] == 0]
TT_average_speed = TT_average_speed["Average_speed"]

flat_average = cleandata.loc[cleandata["Type"] == 1]
flat_average = flat_average["Average_speed"]

hilly_average = cleandata.loc[cleandata["Type"] == 2]
hilly_average = hilly_average["Average_speed"]

mountain_average = cleandata.loc[cleandata["Type"] == 3]
mountain_average = mountain_average["Average_speed"]

all_average = [TT_average_speed, flat_average, hilly_average, mountain_average]
fig, ax = plt.subplots()
plot = ax.boxplot(all_average, labels=['TT', 'Flat', 'Hilly', 'Mountain'])
plt.show()


#testing on 2011 and 2015 tour de france
testdata = data

train_2011 = data.loc[cleandata["Year"]<2011]
test_2011 = data.loc[cleandata["Year"]==2011]

print(test_2011)
#shuffle
train_2011 = train_2011.sample(frac=1)

train_2011_X = train_2011[["Year", "Distance", "Type"]].values
train_2011_X = scaler.transform(train_2011_X)
train_2011_y = train_2011["Average_speed"].values
test_2011_X = test_2011[["Year", "Distance", "Type"]].values
test_2011_X = scaler.transform(test_2011_X)
test_2011_y = test_2011["Average_speed"].values
test_2011_y = np.vstack(test_2011_y)

test2011model = model()
test2011model.fit(train_2011_X, train_2011_y, epochs=250, batch_size=16, verbose=0)
pred_2011 = test2011model.predict(test_2011_X)

difference2011 = abs(pred_2011 - test_2011_y)
percent2011 = (difference2011 / test_2011_y) * 100

stages2011 = test_2011["Stage"].values
for i, j in enumerate(stages2011):
    print("Stage:")
    print(j)
    print("Actual:")
    print(test_2011_y[i])
    print("Predicted:")
    print(pred_2011[i])
    print("Difference:")
    print(difference2011[i])
    print("Percent Difference:")
    print(percent2011[i])
    print("")



#2015 testing
train_2015 = cleandata.loc[cleandata["Year"]<2015]
test_2015 = cleandata.loc[cleandata["Year"]==2015]

train_2015 = train_2015.sample(frac=1)

train_2015_X = train_2015[["Year", "Distance", "Type"]].values
train_2015_X = scaler.transform(train_2015_X)
train_2015_y = train_2015["Average_speed"].values
test_2015_X = test_2015[["Year", "Distance", "Type"]].values
test_2015_X = scaler.transform(test_2015_X)
test_2015_y = test_2015["Average_speed"].values
test_2015_y = np.vstack(test_2015_y)

test2015model = model()
test2015model.fit(train_2015_X, train_2015_y, epochs=250, batch_size=16, verbose = 0)
pred_2015 = test2015model.predict(test_2015_X)

difference2015 = abs(pred_2015 - test_2015_y)
percent2015 = (difference2015 / test_2015_y) * 100


stages2015 = test_2015["Stage"].values
for i, j in enumerate(stages2015):
    print("Stage:")
    print(j)
    print("Actual:")
    print(test_2015_y[i])
    print("Predicted:")
    print(pred_2015[i])
    print("Difference:")
    print(difference2015[i])
    print("Percent Difference:")
    print(percent2015[i])
    print("")

