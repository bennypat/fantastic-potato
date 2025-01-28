import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import likelihood

rng = np.random.RandomState(42)


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
    ani.save(filename="./dev/images/raw_images.gif", writer="pillow")

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
        target_snrs.append(target_signal.max() / np.sqrt(noise_rate))
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
    ani.save(filename="./dev/images/images_with_target.gif", writer="pillow")

    return images_w_target, target_snrs


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

            o1 = likelihood.omega1(pixel_window, psf_kernel)
            o2 = likelihood.omega2(psf_kernel)
            o3 = likelihood.omega3(i_min, i_max, np.sqrt(noise_rate), o1, o2)
            likelihood_img[idx, jdx] = likelihood.intensity_marginalized_ratio(
                o1, o2, o3, np.sqrt(noise_rate), i_min, i_max
            )

    return likelihood_img


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
