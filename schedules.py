"""Generates Figure 2 (left paper) of mirror descent paper.

Different annealing schedules for different Gaussian sequences.

To run this script, you need to install first particles:

    >> pip install particles

see also `here <https://github.com/nchopin/particles>`.
"""


from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

import particles
from particles import smc_samplers as ssps
from particles import distributions as dists

class GaussianBridge(ssps.TemperingBridge):
    def __init__(self, mu=None, sigma=None, dim=10):
        self.mu = mu
        self.sigma = sigma
        self.dim = dim
        self.prior = dists.IndepProd(*[dists.Normal() for _ in range(dim)])
        self.log_norm_cst = dim * (np.log(sigma) + dists.HALFLOG2PI)

    def logtarget(self, theta):
        return np.sum(stats.norm.logpdf(theta, loc=self.mu, scale=self.sigma), axis=1)
        # return (-(0.5/self.sigma**2) * np.sum(theta**2, axis=1) 
        #         - self.log_norm_cst)
        #
nruns = 1
N = 100
lc = 50
dim = 25
tau = 100.
mixsigs = np.ones(dim) * tau
mixsigs[:(dim//2)] = 1. / tau
sigs = [(1./ tau) * np.ones(dim), tau * np.ones(dim), mixsigs]
mus = np.ones(dim)
models = [GaussianBridge(mu=mus, sigma=sig, dim=dim) for sig in sigs]

fks = [ssps.AdaptiveTempering(model=mod, len_chain=lc) 
       for mod in models]

results = particles.multiSMC(fk=fks, nruns=nruns, N=N, verbose=True, nprocs=1)

## PLOTS
#######
plt.style.use('ggplot')
plt.rc('font', size=16)

plt.figure()
labels = ['smaller', 'bigger', 'both']
ls = ['solid', 'dashed', 'dotted']
colors = ['blue', 'green', 'red']
for i, r in enumerate(results):
    plt.plot(r['output'].X.shared['exponents'], 
             color=colors[i], ls=ls[i], lw=2, label=labels[i])
plt.legend(loc='lower right')
plt.xlabel(r'$n$')
plt.ylabel(r'$\lambda_n$')
plt.savefig('gaussian_temperatures.pdf')
