import numpy as np
import re

# ANALYSIS CONSTANTS
NTHERM = 0  # Assume already thermalised
NBOOT = 200  # Number of bootstrap samples for error estimation
BINSIZE = 2  # Skip BINSIZE*ceil(tau_int) measuremet for autocorrelation

# Utility and analysis functions


# Get L, beta from the file name
def id_from_fname(fname):
    match = re.search(r"/L(\d+)/B(\d).(\d+)/", fname)
    if match:
        L = int(match.group(1))
        beta = float(match.group(2)) + float(match.group(3)) * 10 ** (
            -len(match.group(3))
        )

    return L, beta


def ms_autocorr_time(x, c=5.0):
    """
    Compute the Madras–Sokal integrated autocorrelation time and its analytic error.
    """
    x = np.asarray(x, np.float64)
    n = x.size
    if n < 8:
        return np.nan, np.nan

    x = x - np.mean(x)
    var = np.var(x)
    if not np.isfinite(var) or var <= 1e-15:
        return np.nan, np.nan

    # FFT-based autocovariance
    nfft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, nfft)
    acov = np.fft.irfft(fx * np.conjugate(fx), nfft)[:n]
    rho = np.real(acov / acov[0])
    rho[np.isnan(rho)] = 0.0
    rho = np.clip(rho, -1.0, 1.0)

    # Cut negative tail
    neg = np.where(rho < 0)[0]
    if len(neg) > 0:
        rho[neg[0] :] = 0.0

    tau = 0.5
    for _ in range(1000):
        W = int(max(1, np.floor(c * tau)))
        if W >= n:
            break
        new_tau = 0.5 + np.sum(rho[1 : W + 1])
        if abs(new_tau - tau) < 1e-5:
            tau = new_tau
            break
        tau = new_tau

    if not np.isfinite(tau) or tau <= 0.0:
        return np.nan, np.nan

    tau_err = tau * np.sqrt((4 * (2 * W + 1)) / n)
    return float(tau), float(tau_err)


def block(arr, size):
    N = len(arr)

    #return np.array([arr[i : i + size].mean() for i in range(0, N, size)])
    return np.array(arr[::size])


def bootstrap(arr, size):
    N = len(arr)
    k = np.random.randint(0, N, size=(size, N))

    return arr[k]


def analyse(arr, L, Ntherm=NTHERM, Nboot=NBOOT, bs_samples=False):
    arr_ = np.abs(arr[Ntherm:])
    mean = np.mean(arr_)

    tau, _ = ms_autocorr_time(arr_)
    blocksize = int(BINSIZE * np.ceil(tau))
    mean_bs = np.mean(bootstrap(block(arr_, blocksize), Nboot), axis=1)
    error = np.std(mean_bs)

    susc_mean = 6 * L**3 * (np.mean(arr_**2) - np.mean(arr_) ** 2)

    SUSCS = []
    samples = bootstrap(block(arr_, blocksize), Nboot)
    for sample in samples:
        SUSCS.append(6 * L**3 * (np.mean(sample**2) - np.mean(sample) ** 2))

    susc_err = np.std(SUSCS)

    binder_mean, binder_error = get_binder(block(arr[Ntherm:], blocksize))#int(np.ceil(tau))))

    if bs_samples:
        samples = bootstrap(block(arr[Ntherm:], blocksize), Nboot)
        return mean, error, susc_mean, susc_err, binder_mean, binder_error, tau, samples
    else:
        return mean, error, susc_mean, susc_err, binder_mean, binder_error, tau


def get_binder(arr, ntherm=NTHERM):

    arr_ = arr - np.mean(arr)
    mean = 1 - np.mean(arr_**4) / ( 3. * np.mean(arr_**2) ** 2)

    bs_arr = bootstrap(arr_, 200)

    error = np.std(
        1 - np.mean(bs_arr**4, axis=1) / (3. * np.mean(bs_arr**2, axis=1) ** 2)
    )

    return mean, error
