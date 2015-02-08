'''.............................
   Creation 2015 by Samuel Tenka
   .............................'''
''' Fisher linear classifier'''

import CSV
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
    offset = vdot(weights, (mus[1]+mus[0])/2)
    return weights, offset*1.0

for i in '123':
    xts = get_xts('D'+i+'_train.csv')
    weights, offset = compute_params(xts)
    pred = lambda x: 1 if vdot(weights,x)-offset>0 else 0
    Errors = [x for x,t in xts if t!=pred(x)]
    print('E'+i+':', len(Errors))
