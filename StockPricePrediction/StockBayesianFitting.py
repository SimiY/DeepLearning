
__author__ = 'Yan Kang'

import csv
import numpy as np
import scipy as Sci
import scipy.linalg as la
import matplotlib.pyplot as plt

# ------------ Initialize -------------- #
Beta = 11.1
Alpha = 0.005
M = 4
Num_T = 200                                # !  How many inputs are used as training
Num_P = 10                                 # !  How many predictions to be made
# -------- load in every day close prices as TestPrices ------------ #
ClosePrice = []
with open('./data/google.csv', 'rb') as csvfile:
    TableReader = csv.DictReader(csvfile)
    for row in TableReader:
        ClosePrice.append(float(row['Close']))
# get the 0-19 prices as the first inputcurve_fitticurve_fitting.pyng.py
TestPrices = ClosePrice[-Num_T:]          # !
print TestPrices
# create time stamp for test prices
TimeStamp = range(1, len(TestPrices) + 1)
PredictTimeStamp = range(1, len(TestPrices) + 1 + Num_P)
print TimeStamp


# ================== Define some functions ======================= #
# ----------- Calculate phi according to input x ------------ #
def phi(x):
    exponent = np.mat(range(0, M + 1))
    phi_x = np.power(x, exponent).T
    return phi_x
# x = 2
# print phi(x)


# ---------- Calculate sum_phi_xn ------------- #
def sum_phi_xn(xn):
    sum_phi = np.mat(np.zeros([M+1, M+1]))
    for i in range(0, len(xn)):
        sum_phi = sum_phi + phi(xn[i])*phi(xn[i]).T
    return sum_phi
# print sum_phi_xn(TimeStamp)
# print sum_phi_n(TimeStamp)


# ---------- Calculate sum_phi_xn_tn ------------- #
def sum_phi_xn_tn(xn, tn):
    sum_phi = np.mat(np.zeros([M + 1, 1]))
    for i in range(0, len(xn)):
        sum_phi = sum_phi + phi(xn[i]) * tn[i]
    return sum_phi
# print sum_phi_xn_tn(TimeStamp,TestPrices)


# ---------- Calculate S inverse -------------- #
def s_inv(xn):
    s_inverse = Alpha * np.mat(np.identity(M + 1)) + Beta * sum_phi_xn(xn)
    return s_inverse
# print s_inv(TimeStamp,2)


# ------------- Calculate m(x) according to input x ---------------- #
def m_x(xn, tn, x):
    mx = Beta * phi(x).T * la.pinv(s_inv(xn)) * sum_phi_xn_tn(xn, tn)
    return mx
# mx = m_x(TimeStamp,TestPrices,2)
# print mx


# ------------- Calculate S2_x according to input x ---------------- #
def sigma2_x(xn, x):
    sigma2x = np.power(Beta, -1) + phi(x).T * la.pinv(s_inv(xn)) * phi(x)
    return sigma2x


# ================== Start prediction ============================ #
# ----------- Normalization ------------ #
# normalizing by subtracting the mean and dividing by the standard deviation
NormalTestPrices = []
for i in range(0, len(TestPrices)):
    middle = (TestPrices[i]-np.mean(TestPrices))/np.std(TestPrices)
    NormalTestPrices.append(middle)
# print NormalTestPrices


# ------------- Output the predicted values with de-normalization ----------- #
mu = np.mat(np.zeros([len(PredictTimeStamp), 1]))           # !
sigma = np.mat(np.zeros([len(PredictTimeStamp), 1]))        # !
x = np.mat(np.zeros([len(PredictTimeStamp), 1]))            # !
output = np.mat(np.zeros([len(PredictTimeStamp), 1]))       # !

for i in range(0, len(PredictTimeStamp)):                   # !
    mu[i] = m_x(TimeStamp, NormalTestPrices, i+1)
    sigma[i] = np.power(sigma2_x(TimeStamp, i+1), 0.5)
    x[i] = sigma[i] * np.random.randn() + mu[i]
    mu[i] = mu[i]*np.std(TestPrices)+np.mean(TestPrices)
    output[i] = x[i]*np.std(TestPrices)+np.mean(TestPrices)
# print mu
# print sigma
# print x
# print output


# ------------- Evaluation ----------------- #
abs_error = []
relative_error = []
for i in range(0, len(TestPrices)):
    abs_error.append(np.absolute(output[i]-TestPrices[i]))
    relative_error.append(abs_error[i]/TestPrices[i])
abs_error_mean = np.mean(abs_error)
relative_average_error = np.mean(relative_error)
s1 = 'Absolute error mean is: ' + repr(abs_error_mean)
s2 = 'Relative average error is: ' + repr(relative_average_error*100) + '%'
print s1
print s2


# ------------- Draw plots ----------------- #
yerr = np.squeeze(np.asarray(abs_error))
fig = plt.figure()
x1 = np.asarray(TimeStamp)
x2 = np.asarray(PredictTimeStamp)           # !
y1 = np.asarray(TestPrices)
y2 = np.asarray(output)
plt.plot(x1, y1, 'ro-', label='original')
plt.errorbar(x1, y1, yerr, color='red')
plt.plot(x2, y2, 'go-', label='predicted')  # !
plt.title('Bayesian curve fitting')
plt.show()