import os

import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import scenepic as sp


def _fft2(image):
    image_freq = np.fft.fft2(np.fft.ifftshift(image))
    image_freq = np.fft.fftshift(image_freq)
    return image_freq


def _ifft2(image_freq):
    image = np.fft.ifft2(np.fft.ifftshift(image_freq))
    image = np.fft.fftshift(image)
    return image.real


def _save(path, image, normalize=False):
    if normalize:
        image = (image - image.min()) / (image.max() - image.min())

    image = (image * 255).astype(np.uint8)
    cv2.imwrite(path, image)


def _main():
    script_dir = os.path.dirname(__file__)
    image = cv2.imread(os.path.join(script_dir, "..", "data", "cat.jpg"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255

    image_freq = _fft2(image)

    _save("image.png", image)
    _save("image_freq.png", np.log(np.abs(image_freq)), True)

    image_freq_flat = image_freq.reshape(-1)
    order = np.argsort(np.abs(image_freq_flat))[::-1]
    i_vals = order // image_freq.shape[1]
    j_vals = order % image_freq.shape[1]

    width, height = 1280, 720
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    canvas = FigureCanvas(fig)
    basis_ax = fig.add_subplot(121)
    recon_ax = fig.add_subplot(122)

    max_num_freqs = 20000
    num_frames = 60

    stops = np.exp(np.linspace(0, np.log(max_num_freqs), num_frames)).astype(np.int32)
    stops[-1] = max_num_freqs
    with sp.VideoWriter("recon2d.mp4", (width, height), framerate=2) as writer:
        for start, end in zip(stops[:-1], stops[1:]):
            if start == end:
                continue

            subset = np.zeros_like(image_freq)
            subset[i_vals[start:end],
                   j_vals[start:end]] = image_freq[i_vals[start:end],
                                                   j_vals[start:end]]
            subset[-i_vals[start:end],
                   -j_vals[start:end]] = image_freq[-i_vals[start:end],
                                                    -j_vals[start:end]]

            basis_ax.cla()
            basis_ax.imshow(_ifft2(subset))
            basis_ax.set_title("Basis {} to {}".format(start, end))

            subset[i_vals[:start],
                   j_vals[:start]] = image_freq[i_vals[:start],
                                                j_vals[:start]]
            subset[-i_vals[:start],
                   -j_vals[:start]] = image_freq[-i_vals[:start],
                                                 -j_vals[:start]]

            recon_ax.cla()
            recon_ax.imshow(_ifft2(subset), cmap="gray")
            recon_ax.set_title("Reconstruction")
            fig.tight_layout()
            canvas.draw()
            buf = canvas.buffer_rgba()
            pixels = np.asarray(buf)[..., :3]
            pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            cv2.imshow("progress", pixels)
            writer.frame[:] = pixels
            writer.write_frame()

            cv2.waitKey(1)


if __name__ == "__main__":
    _main()
