__author__ = 'aeoluseros'
#Basic Profiling: %prun and %run -p
import numpy as np
from numpy.linalg import eigvals
def run_experiement(niter=100):
    k = 100
    results = []
    for _ in xrange(niter):
        mat = np.random.randn(k, k)
        max_eigenvalue = np.abs(eigvals(mat)).max()
        results.append(max_eigenvalue)
    return results
some_results = run_experiement()
print 'Largest one we saw: %s' % np.max(some_results)


#run in terminal: python -m cProfile cprof_example.py
#or run in python console: %prun -l 7 -s cumulative run_experiement()
#or run in python console: %run -p -s cumulative cprof_example.py

#a small library called 'line_profiler' contains an IPython extension enabling
#a new magic function %lprun that computes a line-by-line-profiling of one or
#more functions.
#you can enable this extension by modifying your IPython configuration to include the following line:
#A list of dotted module names of IPython extension to load.
c.TerminalIPythonApp.extensions = ['line_profiler']

def add_and_sum(x, y):
    added = x + y
    summed = added.sum(axis=1)
    return summed

from numpy.random import randn
def call_function():
    x = randn(1000, 1000)
    y = randn(1000, 1000)
    return add_and_sum(x, y)












