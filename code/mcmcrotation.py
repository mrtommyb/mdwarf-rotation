import numpy as np
# import matplotlib.pyplot as plt

import george
from george import kernels

import pandas as pd

from astropy.stats import median_absolute_deviation as MAD
import glob
import scipy.optimize as op
import emcee

import tqdm
import h5py


def read_file(fn):
    df = pd.read_csv(fn, skiprows=39)
    df['time'] = df.t - df.t[0]
    df['flux'] = df.fdt_t_roll_2D / np.median(df.fdt_t_roll_2D) - 1.0
    df['ferr'] = np.ones_like(df.t) * MAD(df.flux) / 20.

    return df


def angus_kernel(theta):
    """
    use the kernel that Ruth Angus uses. Be sure to cite her
    """
    theta = np.exp(theta)
    A = theta[0]
    l = theta[1]
    G = theta[2]
    sigma = theta[4]
    P = theta[3]
    kernel = (A * kernels.ExpSquaredKernel(l) *
              kernels.ExpSine2Kernel(G, P) +
              kernels.WhiteKernel(sigma)
              )
    return kernel


def nll(p, args):
    yval, gp = args
    # Update the kernel parameters and compute the likelihood.
    gp.kernel[:] = p
    ll = gp.lnlikelihood(yval, quiet=True)

    # The scipy optimizer doesn't play well with infinities.
    return -ll if np.isfinite(ll) else 1e25


# And the gradient of the objective function.
def grad_nll(p, args):
    yval, gp = args
    # Update the kernel parameters and compute the likelihood.
    gp.kernel[:] = p
    return -gp.grad_lnlikelihood(yval, quiet=True)


def get_opt_params(theta, t, y, yerr):
    kernel = angus_kernel(theta)
    gp = george.GP(kernel, )  # solver=george.HODLRSolver)
    gp.compute(t, yerr=yerr)

    p0 = gp.kernel.vector
    results = op.minimize(nll, p0, jac=grad_nll,
                          args=[y, gp])

    gp.kernel[:] = results.x

    return gp


def get_giant_outliers(y):
    giant_outliers = np.abs(y) > (5 * MAD(y))
    return giant_outliers


def get_flares(t, y, gp, sig=2.5):
    # for some reason trying to predict at exactly the same time is a problem
    mup, covp = gp.predict(y, t + (t * 0.0001))
    stdp = np.sqrt(np.diag(covp))
    flares = y > mup + (stdp * 2.5)
    return flares


def lnprob(p, time, y, yerr):
    # Trivial improper prior: uniform in the log.
    # from DFM george user guide

    if np.any((-p < -20) + (p > 10)):
        return -np.inf

    lnprior = 0.0

    kernel = angus_kernel(p)
    gp = george.GP(kernel)
    gp.compute(time, yerr)

    return lnprior + gp.lnlikelihood(y, quiet=True)


if __name__ == '__main__':
    fns = glob.glob('../data/*.csv')
    fn = fns[4]
    df = read_file(fn)

    cutdays = True
    if cutdays:
        timemask = df.time < 15.
    else:
        timemask = np.ones_like(df.time, dtype='bool')

    theta_guess = [-10, 2.5 , 5, 1.3, -13]

    omask = get_giant_outliers(df.flux)
    mask = ~omask & timemask

    gp = get_opt_params(theta_guess,
                        df.loc[mask, 'time'],
                        df.loc[mask, 'flux'],
                        df.loc[mask, 'ferr'])

    flares = get_flares(df.time[mask],
                        df.flux[mask],
                        gp,
                        sig=2.5,
                        )
    notflares = ~flares

    # recalculate GP
    mask = ~omask & notflares & timemask
    gp = get_opt_params(gp.kernel[:],
                        df.loc[mask, 'time'],
                        df.loc[mask, 'flux'],
                        df.loc[mask, 'ferr'])

    nwalkers, ndim = 16, len(gp.kernel[:])
    outfile = 'chain.hdf5'
    steps = 30
    with h5py.File(outfile, u"w") as f:
        c_ds = f.create_dataset("chain",
                                (nwalkers, steps, ndim),
                                dtype=np.float64)
        lp_ds = f.create_dataset("lnprob",
                                 (nwalkers, steps),
                                 dtype=np.float64)


    print('starting mcmc with params {}'.format(gp.kernel[:]))



    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=[
                                        df.loc[mask, 'time'],
                                        df.loc[mask, 'flux'],
                                        df.loc[mask, 'ferr'],
                                    ])

    p0 = (np.repeat(gp.kernel.vector, nwalkers) +
          (np.random.random(size=ndim * nwalkers) * 1.e-4))
    p0 = p0.reshape(ndim, nwalkers).T
    print("Running burn-in")
    p0, _, _ = sampler.run_mcmc(p0, 2)

    print("Running production chain")

    for ind, (pos, lnprob, state) in tqdm.tqdm(enumerate(sampler.sample(
            p0, iterations=steps, storechain=False))):
        with h5py.File(outfile, u"a") as f:
            c_ds = f["chain"]
            lp_ds = f["lnprob"]
            c_ds[:, ind, :] = pos
            lp_ds[:, ind] = lnprob




    print("Mean acceptance fraction: {0:.3f}"
        .format(np.mean(sampler.acceptance_fraction)))





