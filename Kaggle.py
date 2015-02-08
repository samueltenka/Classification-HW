'''.............................
   Creation 2015 by Samuel Tenka
   .............................'''
''' perceptron classifier'''

import CSV
import Plot
from math import exp, log
from numpy import zeros, array, vdot

learning_rate = 1E-5
num_steps = 2000
feature_dim = 1+784

def features(x):
    return array([1]+x[1:]) # chop off ID, but prepend bias
def get_xts(filename):
    reader = CSV.CSV(); print('reading...')
    reader.read_from(filename); print('done!')
    return [(features(xt[:-1]), xt[-1]) for xt in reader.rows]
def predict(x, weights):
    return 1 if vdot(x, weights)>0 else 0
def compute_weights(xts):
    print('ha')
    count = 0
    weights = zeros(feature_dim)
    while True:
        for x,t in xts:
            if predict(x,weights)!=t:
                weights += learning_rate*x*(1 if t==1 else -1)
                count += 1
                if count>=num_steps: return weights

print('ha')
xts = get_xts('train_usps_small.csv')
weights = compute_weights(xts)
print(weights)
