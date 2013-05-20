def is_square(A):
    if not A:
        return True
    return len(A) == len(A[0])


import theano

def theano_simplify(fgraph):
    """ Simplify a Theano Computation """
    mode = theano.compile.get_default_mode()
    fgraph = fgraph.clone()
    mode.optimizer.optimize(fgraph)
    return fgraph
