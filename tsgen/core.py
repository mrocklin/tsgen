from util import is_square
from operator import mul

def butcher_explicit(A, b, f, h, t, y):
    """ Evolve forward with time stepping method

    A:  Butcher array (square lower triangular matrix)
    b:  Butcher vector
    f:  Derivative function :: (time, state -> direction)
    h:  Time step
    t:  Current time
    y:  state
    """
    assert is_square(A)
    assert len(A) == len(b) - 1

    ks = [h*f(t, y)]

    for i, row in enumerate(A):
        c = sum(row)
        time = t + c*h
        delta = sum(map(mul, row[:i+1], ks))
        ks.append(h * f(time, y + delta))

    return y + sum(map(mul, b, ks))

# (dt, t, state -> state), dt, time, time, state -> state
def evolve(step, dt, start, end, state):
    """ Repeat a stepping function through time (simple)

    step :: dt, time, state -> state
    """
    time = start
    while time < end:
        state = step(dt, time, state)
        time += dt
    return state
