import numpy as np

def np_bivariate_normal_pdf(domain, mean, variance):
  X = np.arange(-domain+mean, domain+mean, variance)
  Y = np.arange(-domain+mean, domain+mean, variance)
  X, Y = np.meshgrid(X, Y)
  R = np.sqrt(X**2 + Y**2)
  Z = ((1. / np.sqrt(2 * np.pi)) * np.exp(-.5*R**2))
  return X+mean, Y+mean, Z

import numpy as np
import numpy.random
import matplotlib.pyplot as plt

sigma = 2.5
mu = 5
x = np.array([])
y = np.array([])
gap_x = 0
x_sum = np.array([])
y_sum = np.array([])

for c in range (3):
    for a in range(0, 2):
        for b in range(0, 2):
            # Generate some test data
            dx = sigma * np.random.randn(10000)
            dy = sigma * np.random.randn(10000)
            x_sum = np.append(x_sum, dx)
            y_sum = np.append(y_sum, dy)
            x = np.append(x, dx + a * 10 + c * 20 + gap_x)
            y = np.append(y, dy + b * 8)
    gap_x += 15

heatmap, xedges, yedges = np.histogram2d(x, y, bins=100)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=plt.cm.Reds)
plt.show()

heatmap2, xedges2, yedges2 = np.histogram2d(x_sum, y_sum, bins=100)
extent = [xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]]
plt.clf()
plt.imshow(heatmap2.T, extent=extent, origin='lower', cmap=plt.cm.Reds)
plt.show()