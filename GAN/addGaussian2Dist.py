import numpy as np
import matplotlib.pyplot as plt

'''mu = 0
sigma = 0.1
mynorm = np.random.normal(mu, sigma, 1000)

count, bins, ignored = plt.hist(mynorm, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2*np.pi)) * np.exp(- (bins - mu) **2 / (2 * sigma **2)), linewidth=2, color='r')
plt.show()'''


for p in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    n = 100
    size = 1000
    mynegbin = np.random.negative_binomial(n, p, size)
    count, bins, ignored = plt.hist(mynegbin, 30, density=True)
    plt.show()

    var=10
    for i in range(len(mynegbin)):
        mynegbin[i] += np.random.normal(0, 0.1)

    count, bins, ignored = plt.hist(mynegbin, 30, density=True)
    plt.show()




