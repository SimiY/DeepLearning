__author__ = 'Yan Kang'
# date: 24/04/2015

import csv
import numpy as np
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import matplotlib.pyplot as plt

# -------- Load in whole year every day close prices as TrainingPrices ------------
# -------- File path ------
filepath = './data/google.csv'
market_vibration = 5
# -------- Create Training set --------
ClosePrice = []
with open(filepath, 'rb') as csvfile:
    TableReader = csv.DictReader(csvfile)
    for row in TableReader:
        ClosePrice.append(float(row['Close']))
# print ClosePrice                                          # 251 items
# print ('ClosePrice length is ' + str(len(ClosePrice)))
TrainingPrices = ClosePrice[0:len(ClosePrice)-1]
# print TrainingPrices                                      # 250 items
# print ('TestPrices length is ' + str(len(TrainingPrices)))


# --------- Create time stamp for Training Prices -----------
TimeStamp = range(1, len(ClosePrice) + 1)
PredictTimeStamp = len(ClosePrice)                          # The last time stamp is the day we want to predict
# print TimeStamp                                           # TimeStamp 1-251

# --------- Create the input X based on the order M -------------
M = int(raw_input('Please input M (integer >0): '))
X = np.mat(TimeStamp).T
X_M = np.mat(np.zeros(shape=(len(ClosePrice), M+1)))
# print X
for i in range(0, M+1):
    X_temp = np.power(X, i)
    X_M[:, i] = X_temp
X = X_M                 # 251*(M+1) Matrix, first column elements are all "1"s, and then in order 1, order 2 ...order M
# print X

# --------- Reshape the input X into [0,1] -------------
X = X/X[len(ClosePrice)-1, M]  # scale pixel values to [0, 1], 251*(M+1)
X = X.astype(np.float32)
X_full = X                                                  # save the original X
X_last = X[-1, :]                                           # save the last X to make prediction
X = X[0:-1, :]
# print X

# --------- Reshape the label y(prices) into [-1, 1] -------------
Max = max(ClosePrice)
Min = min(ClosePrice)
Average = (Max + Min)/2
Distance = max(ClosePrice) - Average
y = np.subtract(ClosePrice, Average)
y = np.divide(y, Distance)                                  # y is in the range [-1, 1]
y = np.mat(y)
y = y.T
y = y.astype(np.float32)
y_full = y                                                  # save the original y
y_last = y[-1]                                              # save the true last y to compare prediction
y = y[0:-1, :]

# --------- Construct Neural Network ------------

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, M+1),  # 96x96 input pixels per batch
    hidden_num_units=100,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=1,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=40,  # we want to train this many epochs
    verbose=1,
    )

net1.fit(X, y)
y_predict = net1.predict(X)
row, col = y_predict.shape
y_predict = y_predict*Distance+Average
for i in range(0, row):
    y_predict[i] += np.random.normal(0, market_vibration)  # add price vibration into consideration
abs_error = np.absolute(y_predict-np.mat(ClosePrice[0:-1]).T)
relative_error = abs_error/np.mean(ClosePrice[0:-1])
print np.mean(abs_error)
print np.mean(relative_error)
print y_predict
print ClosePrice[0:-1]
print abs_error.shape
# -------------- plot ------------
plt.close("all")
fig = plt.figure()
err = np.squeeze(np.array(abs_error))
x1 = np.asarray(range(1, len(ClosePrice)))  # 1- 250
y1 = np.asarray(ClosePrice[0:-1])           # true prices
y2 = np.asarray(y_predict)                  # predict prices
plt.plot(x1, y1, 'ro-')
plt.plot(x1, y2, 'go-')
plt.errorbar(x1, y1, err, color='red')
plt.title('One layer neural network prediction curve')
plt.show()














