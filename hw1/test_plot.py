#!/usr/bin/env python
import numpy as np
import matplotlib.mlab as mlab
import matplotlib as mpl
#mpl.use('agg')
import matplotlib.pyplot as plt
#mpl.rcParams['savefig.dpi'] = 100

from pylab import *

mu, sigma = 100, 15
x = mu + sigma*np.random.randn(10000)
x_mod=[0, 0, 1, 3, 2, 1, 0, 2, 1, 0, 2, 2, 0, 0, 2, 0, 2, 0, 1, 1]
# the histogram of the data
n, bins, patches = plt.hist(x_mod, 50, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=1)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)

plt.show()
#plt.savefig('Test.png')

