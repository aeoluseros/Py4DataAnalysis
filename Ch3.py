__author__ = 'aeoluseros'

a = 5
a
import numpy as np
from numpy.random import randn
data = {i:randn() for i in xrange(7)}
data
print data  # if directly print dict, it would be much less readable.

x = 'foobar'
y = 'foo'
%timeit x.startswith(y)
%timeit x[:3] == y

#Advanced IPython Features
class Message:
    def __init__(self, msg):
        self.msg = msg

x = Message('I have a secret')
x

# add a simple __repr__ method to the above class to get a more helpful output
class Message:
    def __init__(self, msg):
        self.msg = msg
    def __repr__(self):
        return 'Message: %s' % self.msg

x = Message('I have a secret')
x





