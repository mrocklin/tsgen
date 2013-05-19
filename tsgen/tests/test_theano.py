from tsgen.core import butcher_explicit
from tsgen.methods import rk4, euler, midpoint
from theano import tensor as T
from tsgen.util import theano_simplify
import theano
import numpy as np

sigma, beta, rho = 10., 8./3, 28.

def pack(*args):
    """ Pack variables into numpy object array

    Theano variables are iterable so asarray is confused when used normally
    """
    result = np.empty(len(args), dtype=object)
    for i, arg in enumerate(args):
        result[i] = arg
    return result

def lorenz_f(t, (x, y, z)):
    """ Lorenz Attractor '67 """
    return pack(sigma*(y-x), x*(rho-z) - y, -beta*z + x*y)

x, y, z = map(theano.tensor.scalar, 'xyz')
start_state = pack(x, y, z)

dt = theano.tensor.scalar('dt')

A, b = midpoint
xout, yout, zout = butcher_explicit(A, b, lorenz_f, dt, 0, start_state)

fgraph = theano.FunctionGraph([x, y, z, dt], [xout, yout, zout])
fgraph_simplified = theano_simplify(fgraph)

theano.printing.pydotprint(fgraph, outfile='lorenz.pdf', format='pdf')
theano.printing.pydotprint(fgraph_simplified, outfile='lorenz-simplified.pdf',
        format='pdf')
