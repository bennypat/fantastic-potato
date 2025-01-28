# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import images
import target

np.set_printoptions(precision=4)


def main():
    noise_rate = 50
    snr = 10  # 3 ~ 1.7, 10 ~ 6
    i_min = 0.5 * snr * np.sqrt(noise_rate)
    i_max = 2.0 * snr * np.sqrt(noise_rate)
    img_size = (128, 128)
    num_images = 30
    psf_size = 9
    sigma = 0.5
    psf_kernel = images.gaussian_psf(psf_size, sigma)

    # define target(s)
    dt = 1
    A = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

    x0a = np.array([5, 1, 10, 0])  # x, vx, y, vy
    xa = target.gen_states(x0a, A, 30)
    x0b = np.array([5, 2, 30, 0])  # x, vx, y, vy
    xb = target.gen_states(x0b, A, 30)

    # gen images
    print("Generating images")
    dark_frames = images.gen_images(noise_rate, img_size, num_images)

    print("Inserting target(s)")
    ims, snr_a = images.insert_target(dark_frames, xa, psf_kernel, snr, noise_rate)
    ims, snr_b = images.insert_target(ims, xb, psf_kernel, snr, noise_rate)
    print(f"{np.array(snr_a).mean()=}")
    print(f"{np.array(snr_b).mean()=}")

    print("Calculating likelihoods")
    likelihood_ims = []
    for im, dark in zip(ims, dark_frames):
        lh_im = images.image_likelihood(im, dark, psf_kernel, noise_rate, i_min, i_max)
        # sorted_lh_im_idxs = np.argsort(lh_im.flatten())
        # sorted_lh_im = lh_im.flatten()[sorted_lh_im_idxs]
        # print(f"Top 5 likelihood values: {sorted_lh_im[-5:]}")
        likelihood_ims.append(lh_im)

    print("Animating likelihood images")
    images.animate(likelihood_ims, "./dev/images/lh_images.gif", vmin=0, vmax=10)

    print("Making detection gif")
    fig, ax = plt.subplots(1, 2)
    artists = []
    for idx, likelihood_map in enumerate(likelihood_ims):
        container1a = ax[0].imshow(
            ims[idx],
            vmin=0.5 * noise_rate,
            vmax=1.5 * noise_rate,
            origin="lower",
        )

        title1 = ax[0].text(
            0.5,
            1.01,
            f"Frame {idx}",
            ha="center",
            va="bottom",
            transform=ax[0].transAxes,
            fontsize="large",
        )

        container2a = ax[1].imshow(
            likelihood_map,
            vmin=0,
            vmax=10,
            origin="lower",
        )

        print(f"{np.sort(likelihood_map.flatten())[-5:]}")
        detections = np.argwhere(likelihood_map > 1)

        if detections.size:
            container2c = ax[1].plot(
                detections[:, 1], detections[:, 0], "gs", markerfacecolor="none"
            )[0]
        title2 = ax[1].text(
            0.5,
            1.01,
            f"Frame {idx} Likelihood Image with Detections",
            ha="center",
            va="bottom",
            transform=ax[1].transAxes,
            fontsize="medium",
        )
        if detections.size:
            artists.append(
                [
                    container1a,
                    title1,
                    container2a,
                    container2c,
                    title2,
                ]
            )
        else:
            artists.append(
                [
                    container1a,
                    title1,
                    container2a,
                    title2,
                ]
            )

    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=400)
    ani.save(filename="./figs/lh_image_with_detections.gif", writer="pillow")

    print("Done")


if __name__ == "__main__":
    main()
