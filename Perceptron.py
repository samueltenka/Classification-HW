'''.............................
   Creation 2015 by Samuel Tenka
   .............................'''
''' logistic classifier'''

import CSV
import Plot
from math import exp, log
from numpy import zeros, array, vdot

learning_rate = 1E-5
num_steps = 2000

def get_xts(filename):
    reader = CSV.CSV()
    reader.read_from(filename)
    return [(array([1]+xt[:-1]), xt[-1]) for xt in reader.rows]
def predict(x, weights):
    return 1 if vdot(x, weights)>0 else 0
def compute_weights(xts):
    count = 0
    weights = zeros(3)
    while True:
        for x,t in xts:
            if predict(x,weights)!=t:
                weights += learning_rate*x
                count += 1;
                if count>=num_steps: return weights

def generate_boundary(weights, xts):
    x1s = [x1 for (o,x1,x2),t in xts]  # find x1-range
    x1_range = 1.5*(max(x1s)-min(x1s)) # `    `  `
    x1s = [(n-50)*x1_range/100.0 for n in range(100)]
    x2s = [(-weights[0]-weights[1]*x1)/weights[2] for x1 in x1s]
    return x1s, x2s

for i,j in zip('123', [0,1,2]):
    xts = get_xts('D'+i+'_train.csv')
    weights = compute_weights(xts); print(weights)
    x1s, x2s = generate_boundary(weights, xts)
    classes = {0:[], 1:[]}
    for x,t in xts: classes[t].append(x)

    Plot.plot_scatter(classes[0], 0, label='class 0')
    Plot.plot_scatter(classes[1], 1, label='class 1')
    Plot.plot_line(x1s, x2s, label='decision boundary')
    Plot.save_plot('x1', 'x2', 'Logistic Decision Boundary for D'+i,
                   'perceptron_'+i+'.png')

    Errors = [x for x,t in xts if t!=predict(x, weights)]
    print('E', len(Errors))
