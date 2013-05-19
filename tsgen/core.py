from util import is_square

def get_row(X, i):
    return X[i, :i]

def butcher_explicit(A, b, f, h, t, y):
    assert is_square(A)
    assert len(A) == len(b) - 1

    ks = [h*f(t, y)]

    for row in A:
        c = sum(row)
        time = t + c*h
        delta = sum(a*k for a, k in zip(row, ks))
        ks.append(h * f(time, y + delta))

    return y + sum(bi*k for bi, k in zip(b, ks))

# (dt, t, state -> state), dt, time, time, state -> state
def evolve(step, dt, start, end, state):
    """
    step :: dt, time, state -> state
    """
    time = start
    while time < end:
        state = step(dt, time, state)
        time += dt
    return state
