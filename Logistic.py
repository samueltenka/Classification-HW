'''.............................
   Creation 2015 by Samuel Tenka
   .............................'''
''' logistic classifier'''

import CSV
from math import exp, log
from numpy import zeros, array, vdot

regularizer = 1E-6
learning_rate = 1E-5
num_steps = 2000

def sigma(z):
    return 1.0/(1.0+exp(-z))
def prediction(x, w):
    return sigma(vdot(w,x))
def gradient(w, xts):
    return regularizer*w - sum((t-prediction(x, w))*x for x,t in xts)

def compute_weights(filename):
    training = CSV.CSV()
    training.read_from(filename)
    xts = [(array([1]+xt[:-1]), xt[-1]) for xt in training.rows]
    print('!')

    weights = zeros(3)
    for n in range(num_steps):
        if n%100==0: print(n)
        weights -= learning_rate*gradient(weights, xts)
    return weights

print(compute_weights('D1_train.csv'))
print(compute_weights('D2_train.csv'))
print(compute_weights('D3_train.csv'))
''' OUTPUT:
!
0
100
200
300
400
500
600
700
800
900
1000
1100
1200
1300
1400
1500
1600
1700
1800
1900
[-0.47255192  0.02103307  0.25201322]
!
0
100
200
300
400
500
600
700
800
900
1000
1100
1200
1300
1400
1500
1600
1700
1800
1900
[ 1.82111305  1.81886728 -2.56111293]
!
0
100
200
300
400
500
600
700
800
900
1000
1100
1200
1300
1400
1500
1600
1700
1800
1900
[-0.55017607  1.24441333 -0.47775587]'''
