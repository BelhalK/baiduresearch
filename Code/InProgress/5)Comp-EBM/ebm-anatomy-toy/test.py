import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
a = np.arange(0.0, 1.0, 0.01)
b = 0.01
s = 1+b+a*(1-b)

fig, ax = plt.subplots()
ax.plot(a, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

plt.show()
