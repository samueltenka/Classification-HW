import pylab

def plot_line(xs, ys, label=None):
    pylab.plot(xs, ys, label=label,
               color='red')

def plot_scatter(xys, c, label):
    xs = [x for o,x,y in xys]
    ys = [y for o,x,y in xys]
    pylab.scatter(xs, ys, label=label, s=2,
                  color=('green' if c else 'blue'))

def save_plot(xlabel, ylabel, title, filename):
    pylab.legend()

    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    pylab.title(title)
    pylab.grid(True)
    pylab.savefig(filename)

    pylab.clf()
