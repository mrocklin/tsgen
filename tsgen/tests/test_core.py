import numpy as np
from numpy import sin, cos, asarray
from tsgen.core import butcher_explicit, evolve
from functools import partial
from tsgen.methods import euler, midpoint, rk4

def circle_exact(t):
    R = 1
    speed = 1
    return asarray((R*cos(speed*t), (R*sin(speed*t))))

def circle_f(t, (x, y)):
    r = (x**2 + y**2)**.5
    if x == 0:
        return np.pi/2 if y > 0 else -np.pi/2
    theta = np.arctan(y/x)
    if x < 0:
        theta += np.pi
    return asarray((-r*sin(theta), (r*cos(theta))))

def test_euler_circle():
    A, b = euler
    h = .1
    xout, yout = butcher_explicit([], [1], circle_f, h, 0., (1., 0.))
    assert yout > 0
    assert xout == 1

def test_midpoint_circle():
    A, b = midpoint
    h = .1
    xout, yout = butcher_explicit(A, b, circle_f, h, 0., (1., 0.))
    assert yout > 0
    assert xout < 1

def test_rk4_circle():
    A, b = rk4
    h = .1
    xout, yout = butcher_explicit(A, b, circle_f, h, 0., (1., 0.))
    assert yout > 0
    assert xout < 1

def test_evolve():
    eps = 1e-15
    def step(h, t, x):
        return x+h

    assert abs(evolve(step, .1, 4, 6-10*eps, 0) - 2.0) < 100*eps


def test_evolve_circle_rk4():
    A, b = rk4
    step_rk4 = partial(butcher_explicit, A, b)
    step_circle = partial(step_rk4, circle_f)
    revolve = partial(evolve, step_circle, .1, 0, 2*np.pi)

    states = map(asarray, [(1., 0.), (np.sqrt(2), np.sqrt(2)), (-1., 0.), (0.,-1.)])
    for state in states:
        print state, revolve(state)
    assert all(np.allclose(revolve(state), state, atol=1e-1) for state in states)
