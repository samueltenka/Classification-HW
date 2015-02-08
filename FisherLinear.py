'''.............................
   Creation 2015 by Samuel Tenka
   .............................'''
''' Fisher linear classifier'''

import CSV
import Plot
from numpy import array, outer, dot, vdot
from numpy.linalg import inv


def get_xts(filename):
    reader = CSV.CSV()
    reader.read_from(filename)
    return [(array(xt[:-1]), xt[-1]) for xt in reader.rows] # no '1' in front
def compute_params(xts):
    C={0:[], 1:[]}
    for x,t in xts: C[t].append(array(x))
    mus = {t:sum(C[t])/len(C[t]) for t in [0,1]}
    S = sum(outer(x-mus[t], x-mus[t]) for x,t in xts)
    weights = dot(inv(S), (mus[1]-mus[0]))
    normed = weights / vdot(weights,weights)**0.5

    # projected:
    pxs = {t:[vdot(normed, x) for x in C[t]] for t in [0, 1]}
    pmus = {t:sum(pxs[t])/len(pxs[t]) for t in [0, 1]}
    pvars = {t:(sum((px-pmus[t])**2 for px in pxs[t])/len(pxs[t]))**0.5 for t in [0, 1]}
    offset = (pvars[1]*pmus[0]+pvars[0]*pmus[1])/(pvars[0]+pvars[1]) #vdot(weights, (mus[1]+mus[0])/2)
    return normed, offset

def generate_boundary(weights, offset, x1_range):
    x1s = [x1 for (x1,x2),t in xts]
    x1_range = 1.5*(max(x1s)-min(x1s))

    x1s = [(n-50)*x1_range/100.0 for n in range(100)]
    x2s = [(offset-weights[0]*x1)/weights[1] for x1 in x1s]
    return x1s, x2s


for i,j in zip('123', [0,1,2]):
    xts = get_xts('D'+i+'_train.csv')
    weights, offset = compute_params(xts); print(weights)
    x1s, x2s = generate_boundary(weights, offset, xts)
    classes = {0:[], 1:[]}
    for x,t in xts: classes[t].append(x)

    Plot.plot_scatter(classes[0], 0, label='class 0')
    Plot.plot_scatter(classes[1], 1, label='class 1')
    Plot.plot_line(x1s, x2s, label='decision boundary')
    Plot.save_plot('x1', 'x2', 'Logistic Decision Boundary for D'+i,
                   'fisher_'+i+'.png')

    pred = lambda x: vdot(weights, x)-offset > 0
    Errors = [x for x,t in xts if t!=pred(x)]
    print('E', len(Errors))
