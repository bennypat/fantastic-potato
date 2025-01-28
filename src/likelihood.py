import numpy as np
from scipy.special import erf


def omega1(window, psf_kernel):
    """
    Calculate Omega 1: Weighted sum of observed intensities and PSF contributions.

    Parameters:
        window (np.ndarray): Section of the image centered on the pixel, same size as the PSF kernel.
        psf_kernel (np.ndarray): Gaussian PSF kernel.

    Returns:
        float: The value of Omega 1.
    """
    return 2 * np.sum(window * psf_kernel)


def omega2(psf_kernel):
    """
    Calculate Omega 2: Weighted sum of squared PSF contributions.

    Parameters:
        psf_kernel (np.ndarray): Gaussian PSF kernel.

    Returns:
        float: The value of Omega 2.
    """
    return np.sum(psf_kernel**2)


def omega3(i_min, i_max, sigma, omega1, omega2):
    """
    Calculate Omega 3: Normalization factor for intensity-marginalized likelihood.

    Parameters:
        i_min (float): Minimum target intensity.
        i_max (float): Maximum target intensity.
        sigma (float): Noise standard deviation.
        omega1 (float): Weighted sum of observed intensities and PSF contributions.
        omega2 (float): Weighted sum of squared PSF contributions.

    Returns:
        float: The value of Omega 3.
    """
    # Cap the argument of the exponential to prevent overflow
    max_exp_arg = 700
    exp_arg = (omega1**2) / (8 * sigma**2 * omega2)
    exp_term = np.exp(min(exp_arg, max_exp_arg))  # Safeguard against overflow

    return np.sqrt(np.pi * sigma**2) / (i_max - i_min) * np.sqrt(2 * omega2) * exp_term


def intensity_marginalized_ratio(omega1, omega2, omega3, sigma, I_min, I_max):
    """
    Compute the intensity marginalized likelihood ratio.

    Parameters:
        omega1 (float): Weighted sum of observed intensities and PSF contributions.
        omega2 (float): Weighted sum of squared PSF contributions.
        omega3 (float): Normalization factor for likelihood ratio.
        sigma (float): Noise standard deviation.
        I_min (float): Minimum target intensity.
        I_max (float): Maximum target intensity.

    Returns:
        float: Intensity-marginalized likelihood ratio.
    """
    term1 = (2 * I_max * omega2 - omega1) / np.sqrt(8 * sigma**2 * omega2)
    term2 = (2 * I_min * omega2 - omega1) / np.sqrt(8 * sigma**2 * omega2)
    likelihood_ratio = omega3 * (erf(term1) - erf(term2))
    return likelihood_ratio
