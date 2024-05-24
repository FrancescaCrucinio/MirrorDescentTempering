"""
Figure 2 (right panel) of mirror descent paper:
length of tempering sequence vs dimension.

To run this script, you need to install first particles:

    >> pip install particles

see also `here <https://github.com/nchopin/particles>`.

"""



from matplotlib import pyplot as plt

import particles
from particles import smc_samplers as ssps

import constant_rate_ais as lt  # TODO change name

dims = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600]
nruns = 3
N = 100

fks = {d: ssps.AdaptiveTempering(model=lt.GaussianBridge(dim=d), 
                          move=ssps.MCMCSequenceWF(mcmc=lt.ArrayDiagRandomWalk(),
                                                   len_chain=4 * d)) 
       for d in dims}

results = particles.multiSMC(fk=fks, nruns=nruns, N=N, verbose=True)

## PLOTS
#######
plt.style.use('ggplot')
plt.rc('font', size=16)

plt.figure()
plt.scatter([r['fk'] for r in results],
            [r['output'].t for r in results])
plt.xlim(left=1)
plt.ylim(bottom=0)
plt.xlabel('dim')
plt.ylabel('nr tempering steps')
plt.savefig('gaussian_nr_tempering_steps_vs_dim.pdf')

plt.figure()
rmax = [r for r in results if r['fk'] == max(dims)][0]
plt.plot(rmax['output'].X.shared['exponents'])
plt.xlabel('iteration t')
plt.ylabel('tempering exponent')
plt.savefig('gaussian_temperatures.pdf')
