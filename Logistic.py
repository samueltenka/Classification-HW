'''.............................
   Creation 2015 by Samuel Tenka
   .............................'''
''' logistic classifier'''

import CSV
import Plot
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


def get_xts(filename):
    reader = CSV.CSV()
    reader.read_from(filename)
    return [(array([1]+xt[:-1]), xt[-1]) for xt in reader.rows]
def compute_weights(xts):
    weights = zeros(3)
    for n in range(num_steps):
        if n%100==0: print(n)
        weights -= learning_rate*gradient(weights, xts)
    return weights
def generate_boundary(weights, x1_range):
    x1s = [(n-50)*x1_range/100.0 for n in range(100)]
    x2s = [(-weights[0]-weights[1]*x1)/weights[2] for x1 in x1s]
    return x1s, x2s

Ws = [array([-0.47255192,  0.02103307,  0.25201322]),
           array([ 1.82111305,  1.81886728, -2.56111293]),
           array([-0.55017607,  1.24441333, -0.47775587])]

for i,j in zip('123', [0,1,2]):
    xts = get_xts('D'+i+'_train.csv')
    weights = Ws[j] #compute_weights()
    x1s = [x1 for (o,x1,x2),t in xts]
    x1_range = 1.5*(max(x1s)-min(x1s))
    x1s, x2s = generate_boundary(weights, x1_range)
    classes = {0:[], 1:[]}
    for x,t in xts: classes[t].append(x)

    Plot.plot_scatter(classes[0], 0, label='class 0')
    Plot.plot_scatter(classes[1], 1, label='class 1')
    Plot.plot_line(x1s, x2s, label='decision boundary')
    Plot.save_plot('x1', 'x2', 'Logistic Decision Boundary for D'+i,
                   'log_db_'+i+'.png')
