# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.special import erf

np.set_printoptions(precision=4)
rng = np.random.RandomState(42)


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


def image_likelihood(image, dark_frame, psf_kernel, noise_rate, i_min, i_max):
    pad_dim = psf_kernel.shape[0] // 2
    img_size = image.shape

    sub_image = image - dark_frame
    padded_image = np.pad(sub_image.astype(np.float64), pad_dim, mode="reflect")

    likelihood_img = np.zeros_like(sub_image.astype(np.float64))

    for idx in np.arange(0, img_size[0]):
        for jdx in np.arange(0, img_size[1]):
            pad_idx = idx + pad_dim
            pad_jdx = jdx + pad_dim
            pixel_window = padded_image[
                pad_idx - pad_dim : pad_idx + pad_dim + 1,
                pad_jdx - pad_dim : pad_jdx + pad_dim + 1,
            ]

            o1 = omega1(pixel_window, psf_kernel)
            o2 = omega2(psf_kernel)
            o3 = omega3(i_min, i_max, np.sqrt(noise_rate), o1, o2)
            likelihood_img[idx, jdx] = intensity_marginalized_ratio(
                o1, o2, o3, np.sqrt(noise_rate), i_min, i_max
            )

    return likelihood_img


def gaussian_psf(size, sigma):
    """
    Generates a 2D Gaussian kernel.

    Parameters:
        size (int): Kernel size (must be odd for symmetry).
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: Normalized 2D Gaussian kernel.
    """
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)  # Normalize to ensure sum = 1


def gen_images(noise_rate, img_size, num_images):
    """..."""
    fig, ax = plt.subplots()
    images = []
    artists = []
    for idx in range(num_images):
        image = rng.poisson(noise_rate, img_size)

        images.append(image)

        container1a = ax.imshow(
            image,
            vmin=0.5 * noise_rate,
            vmax=1.5 * noise_rate,
            origin="lower",
        )
        title = ax.text(
            0.5,
            1.01,
            f"Frame {idx}",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize="large",
        )

        artists.append([container1a, title])

    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=400)
    ani.save(filename="../figs/raw_images.gif", writer="pillow")

    return images


def insert_target(images, target_states, psf_kernel, snr, noise_rate):
    """..."""

    pad_dim = psf_kernel.shape[0] // 2

    fig, ax = plt.subplots()
    images_w_target = []
    artists = []
    target_snrs = []
    for idx, (image, state) in enumerate(zip(images, target_states)):
        x_t, _, y_t, _ = state
        padded_x_t, padded_y_t = x_t + pad_dim, y_t + pad_dim

        target_signal = rng.poisson(snr * np.sqrt(noise_rate)) * psf_kernel
        target_snrs.append(
            target_signal.max() / np.sqrt((snr * np.sqrt(noise_rate)) + noise_rate)
        )
        padded_image = np.pad(image.astype(np.float64), pad_dim, mode="reflect")

        padded_image[
            padded_x_t - pad_dim : padded_x_t + pad_dim + 1,
            padded_y_t - pad_dim : padded_y_t + pad_dim + 1,
        ] += target_signal

        new_image = padded_image[pad_dim:-pad_dim, pad_dim:-pad_dim]

        images_w_target.append(new_image)

        container1a = ax.imshow(
            new_image, vmin=0.5 * noise_rate, vmax=1.5 * noise_rate, origin="lower"
        )
        title = ax.text(
            0.5,
            1.01,
            f"Frame {idx}",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize="large",
        )

        artists.append([container1a, title])

    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=400)
    ani.save(filename="../figs/images_with_target.gif", writer="pillow")

    return images_w_target, target_snrs


def animate(images, file_name, vmin, vmax):
    """Given a list of images make a gif"""
    fig, ax = plt.subplots()
    artists = []
    for idx, image in enumerate(images):
        container1a = ax.imshow(
            image,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )
        title = ax.text(
            0.5,
            1.01,
            f"Frame {idx}",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize="large",
        )

        artists.append([container1a, title])

    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=400)
    ani.save(filename=file_name, writer="pillow")


# Define the CV dynamics model
def cv_dynamics(x, A):
    new_state = A @ x
    return new_state.astype(int)


def gen_states(x0, A, num_states):
    x = [x0]
    for idx in range(1, num_states, 1):
        xk = cv_dynamics(x[idx - 1], A)
        x.append(xk)

    return np.array(x)


def main():
    noise_rate = 50
    snr = 20  # 3 ~ 1.7, 10 ~ 6
    i_min = 0.5 * snr * np.sqrt(noise_rate)
    i_max = 2.0 * snr * np.sqrt(noise_rate)
    img_size = (128, 128)
    num_images = 30
    psf_size = 9
    sigma = 1.0
    psf_kernel = gaussian_psf(psf_size, sigma)

    # define target(s)
    dt = 1
    A = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

    x0a = np.array([5, 0, 10, 0])  # x, vx, y, vy
    xa = gen_states(x0a, A, 30)
    x0b = np.array([5, 2, 30, 0])  # x, vx, y, vy
    xb = gen_states(x0b, A, 30)

    # gen images
    print("Generating images")
    dark_frames = gen_images(noise_rate, img_size, num_images)

    print("Inserting target(s)")
    ims, snr_a = insert_target(dark_frames, xa, psf_kernel, snr, noise_rate)
    ims, snr_b = insert_target(ims, xb, psf_kernel, snr, noise_rate)
    print(f"{np.array(snr_a).mean()=}")
    print(f"{np.array(snr_b).mean()=}")

    print("Calculating likelihoods")
    prior = 0.5 * np.ones(img_size)
    ps = 0.99
    pb = 0.2
    likelihood_ims = []
    existence_ims = []
    for im, dark in zip(ims, dark_frames):
        pred_existence = pb * (1 - prior) + ps * prior
        lh_im = image_likelihood(im, dark, psf_kernel, noise_rate, i_min, i_max)
        likelihood_ims.append(lh_im)
        updated_existence = (lh_im * pred_existence) / (
            1 - pred_existence + lh_im * pred_existence
        )
        existence_ims.append(updated_existence)
        prior = updated_existence

    print("Animating likelihood images")
    animate(likelihood_ims, "../figs/likelihood_im.gif", vmin=0, vmax=10)
    animate(existence_ims, "../figs/target_existence.gif", vmin=0, vmax=1)

    print("Done")


if __name__ == "__main__":
    main()
