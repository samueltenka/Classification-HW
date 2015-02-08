import numpy, Plot, CSV
import Logistic, GaussQuadratic, FisherLinear, Perceptron

def get_xts(filename, bias=False):
    reader = CSV.CSV()
    reader.read_from(filename)
    return [(numpy.array(([1.1] if bias else [])+xt[:-1]), xt[-1]) for xt in reader.rows]

for i in '123':
    print('Generating', i, ' . . .')
    xts = get_xts('D'+i+'_train.csv')
    biased_xts = get_xts('D'+i+'_train.csv', bias=True)

    logistic_weights = Logistic.Ws['123'.find(i)] #Logistic.compute_weights(biased_xts)
    fweight, offset = FisherLinear.compute_params(xts)
    phis, mus, sigmas = GaussQuadratic.compute_params(xts)
    perceptron_weights = Perceptron.compute_weights(biased_xts)

    alpha = {'1':1.2, '2':2.5, '3':1.2}[i]
    logistic_boundaries = Logistic.generate_boundary(logistic_weights, biased_xts, alpha)
    gauss_boundaries = GaussQuadratic.generate_boundary(phis, mus, sigmas, xts, alpha)
    fisher_boundaries = FisherLinear.generate_boundary(fweight, offset, xts, alpha)
    perceptron_boundaries = Perceptron.generate_boundary(perceptron_weights, biased_xts, alpha)

    classes = {0:[], 1:[]}
    for x,t in xts: classes[t].append(x)
    Plot.plot_scatter(classes[0], 0, label='class 0')
    Plot.plot_scatter(classes[1], 1, label='class 1')

    Plot.plot_line(logistic_boundaries[0], logistic_boundaries[1],
                   label='logistic', color='brown')
    T = int(len(gauss_boundaries[0])/2)
    Plot.plot_line(gauss_boundaries[0][:T], gauss_boundaries[1][:T],
                   gauss_boundaries[0][T:], gauss_boundaries[1][T:],
                   label='gauss quadratic', color='red')
    Plot.plot_line(fisher_boundaries[0], fisher_boundaries[1],
                   label='fisher linear', color='orange')
    Plot.plot_line(perceptron_boundaries[0], perceptron_boundaries[1],
                   label='perceptron', color='magenta')

    Plot.save_plot('x1', 'x2',
                   'Comparison of Decision Boundaries for D'+i,
                   'total_'+i+'.png')
