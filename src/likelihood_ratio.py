import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.special import erf

rng = np.random.RandomState(42)
noise_rate = 100
img_size = (64, 64)
snr = 20
i_max_factor = 2
x0, y0 = 5, 5
state = np.array([x0, y0, 1, 0]).reshape((-1, 1))
nt = int(img_size[0] * 0.7)

# Define PSF parameters
psf_size = 9  # Kernel size (odd)
pad_dim = psf_size // 2
sigma = 0.75  # 625e-9 / (2 * np.pi) * 10  # wavelength * f / 2 * pi * D
pixel_size = 10e-6  # 10 um


def cv_dynamics(x, dt, img_size):
    # Update the target's state using constant velocity model
    new_state = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]) @ x

    # Ensure the position is within bounds
    new_state[0] = np.clip(np.round(new_state[0]), 0, img_size[0] - 1)
    new_state[1] = np.clip(np.round(new_state[1]), 0, img_size[1] - 1)
    return new_state.astype(int)


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


# loop over time
# --------------
print("Generating list of images")

# Generate the PSF kernel (only once)
psf_kernel = gaussian_psf(psf_size, sigma)
print(f"{psf_kernel.max()=}")

fig, ax = plt.subplots()

artists = []
images = []
target_states = []
target_snrs = []
for i in range(nt):
    image = rng.poisson(noise_rate, img_size)
    padded_image = np.pad(image.astype(np.float64), pad_dim, mode="reflect")

    new_state = cv_dynamics(state, 1, img_size)
    # Get the target position
    x_t, y_t = new_state[0, 0], new_state[1, 0]
    target_states.append((x_t, y_t))
    padded_x_t, padded_y_t = x_t + pad_dim, y_t + pad_dim

    target_signal = rng.poisson(snr * np.sqrt(noise_rate)) * psf_kernel
    target_snrs.append(target_signal.max() / np.sqrt(noise_rate))

    padded_image[
        padded_x_t - pad_dim : padded_x_t + pad_dim + 1,
        padded_y_t - pad_dim : padded_y_t + pad_dim + 1,
    ] += target_signal

    image = padded_image[pad_dim:-pad_dim, pad_dim:-pad_dim]

    images.append(image)
    container1a = ax.imshow(
        image, vmin=1.5 * np.sqrt(noise_rate), vmax=0.9 * snr * np.sqrt(noise_rate)
    )
    container1b = ax.plot(y_t, x_t, "r+")[0]
    title = ax.text(
        0.5,
        1.01,
        f"Frame {i}",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize="large",
    )

    artists.append([container1a, container1b, title])
    state = new_state

print(f"{np.array(target_snrs).mean()=}")
ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=400)
ani.save(filename="./figs/test_psf_target_motion.gif", writer="pillow")

print("Calculating likelihood ratios")
# Second loop: Compute the likelihood ratios for each generated image
likelihood_imgs = []
for idx, image in enumerate(images):
    # Calculate the likelihood ratio for this frame using the previously defined function
    x_t, y_t = target_states[idx]

    padded_image = np.pad(image.astype(np.float64), pad_dim, mode="reflect")

    likelihood_img = np.zeros_like(image.astype(np.float64))

    for idx in np.arange(0, img_size[0]):
        for jdx in np.arange(0, img_size[1]):
            # padded_x_t, padded_y_t = x_t + pad_dim, y_t + pad_dim
            pad_idx = idx + pad_dim
            pad_jdx = jdx + pad_dim
            pixel_window = padded_image[
                pad_idx - pad_dim : pad_idx + pad_dim + 1,
                pad_jdx - pad_dim : pad_jdx + pad_dim + 1,
            ]

            i_min = np.sqrt(noise_rate)
            i_max = i_max_factor * snr * np.sqrt(noise_rate)
            o1 = omega1(pixel_window, psf_kernel)
            o2 = omega2(psf_kernel)
            o3 = omega3(i_min, i_max, np.sqrt(noise_rate), o1, o2)
            likelihood_img[idx, jdx] = intensity_marginalized_ratio(
                o1, o2, o3, np.sqrt(noise_rate), i_min, i_max
            )

    likelihood_imgs.append(likelihood_img)

print("Animating likelihood images")
vmax = np.max(likelihood_imgs)
vmin = np.min(likelihood_imgs)
fig, ax = plt.subplots(1, 2)
lr_artists = []
for idx, likelihood_map in enumerate(likelihood_imgs):
    container1a = ax[0].imshow(
        images[idx],
        vmin=0.5 * np.sqrt(noise_rate),
        vmax=1.2 * snr * np.sqrt(noise_rate),
    )
    container1b = ax[0].plot(target_states[idx][1], target_states[idx][0], "r+")[0]
    title1 = ax[0].text(
        0.5,
        1.01,
        f"Frame {idx}",
        ha="center",
        va="bottom",
        transform=ax[0].transAxes,
        fontsize="large",
    )

    container2a = ax[1].imshow(likelihood_map, vmin=0.5 * vmin, vmax=1.2 * vmax)
    # container2b = ax[1].plot(target_states[idx][1], target_states[idx][0], "r+")[0]

    max_idxs = np.unravel_index(
        np.argpartition(likelihood_map.flatten(), -4)[-4:], likelihood_map.shape
    )

    # detections = np.argwhere(likelihood_map > 10**23)
    # print(f"{max_idxs=}")
    # max_idx = np.unravel_index(np.argmax(likelihood_map), likelihood_map.shape)

    # container2c = ax[1].plot(
    #     detections[:, 1], detections[:, 0], "gs", markerfacecolor="none"
    # )[0]
    title2 = ax[1].text(
        0.5,
        1.01,
        f"Frame {idx} Likelihood Map",
        ha="center",
        va="bottom",
        transform=ax[1].transAxes,
        fontsize="large",
    )

    lr_artists.append(
        [
            container1a,
            container1b,
            title1,
            container2a,
            # container2b,
            # container2c,
            title2,
        ]
    )

ani = animation.ArtistAnimation(fig=fig, artists=lr_artists, interval=400)
ani.save(filename="./figs/likelihood_map_animation.gif", writer="pillow")

# Compute and plot the CDF for each frame
fig = plt.figure(figsize=(10, 6))
ax = fig.gca()

tgt_likelihoods = []
for idx, likelihood_map in enumerate(likelihood_imgs):
    tgt_x, tgt_y = target_states[idx]
    tgt_likelihoods.append(likelihood_map[tgt_x, tgt_y])

    # Flatten the likelihood map
    flattened_likelihoods = likelihood_map.flatten()

    # Sort the flattened likelihoods
    sorted_likelihoods = np.sort(flattened_likelihoods)

    # Compute the CDF
    cdf = np.arange(1, len(sorted_likelihoods) + 1) / len(sorted_likelihoods)

    # Plot the CDF
    ax.plot(sorted_likelihoods, cdf, label=f"Frame {idx}")


# Add labels, legend, and grid
ax.set_xscale("log")
ax.xaxis.set_label_text("Likelihood")
ax.yaxis.set_label_text("Cumulative Proportion (CDF)")
ax.set_title("CDF of Likelihoods for Each Frame")
ax.grid(True)
fig.savefig("./figs/CDF of likelihoods")

tgt_likelihoods.sort()
cdf = np.arange(len(tgt_likelihoods)) / len(tgt_likelihoods)
fig = plt.figure(figsize=(10, 6))
ax = fig.gca()
ax.plot(tgt_likelihoods, cdf)
ax.set_xscale("log")
ax.xaxis.set_label_text("Likelihood")
ax.yaxis.set_label_text("Cumulative Proportion (CDF)")
ax.set_title("CDF of Target's Likelihoods for Each Frame")
ax.grid(True)
fig.savefig("./figs/CDF of target's likelihoods")

print("done")
