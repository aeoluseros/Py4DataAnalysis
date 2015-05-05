__author__ = 'aeoluseros'
#%pylab inline
%matplotlib inline
#ipython qtconsole
#in qtconsole, run:  %pylab inline

#%"D:\Program Files\Anaconda\Lib\site-packages\pylab.py" inline
#%matplotlib inline

import matplotlib.pyplot as plt
from pylab import *
from numpy.random import randn
#plt.show()
img = plt.imread('./pydata-book/ch03/stinkbug.png')
plt.imshow(img)

plt.plot(randn(1000).cumsum())
