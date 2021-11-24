"""Product various figures and images for 1D signal stuff."""

import numpy as np


def _multifreq(t):
    return 2 + np.sin(t*np.pi) + 0.5*np.sin(2*t*np.pi) - 0.2*np.cos(5*t*np.pi)


def _main():
    num_samples = 64
    t = np.linspace(0, 2, num_samples)
    y = _multifreq(t)
    y_freq = np.fft.fft(y)
    num_freqs = 4
    order = np.argsort(np.abs(y_freq[:num_samples // 2]))[::-1]
    freqs = np.fft.fftfreq(num_samples)
    basis = []
    recon = []
    for i in range(num_freqs):
        subset = np.zeros_like(y_freq)
        f = order[i]
        print(freqs[f] * num_samples, "hz", y_freq[f].real / num_samples,
              y_freq[f].imag / num_samples, y_freq[-f].real / num_samples,
              y_freq[-f].imag / num_samples)

        subset[f] = y_freq[f]
        subset[-f] = y_freq[-f]
        basis.append(np.fft.ifft(subset).real)
        for f in order[:i]:
            subset[f] = y_freq[f]
            subset[-f] = y_freq[-f]

        recon.append(np.fft.ifft(subset).real)

    with open("fourier_plots.tsv", "w") as file:
        file.write("\t".join(["t", "f(t)"] +
                             ["basis{}".format(i) for i in range(num_freqs)] +
                             ["recon{}".format(i) for i in range(num_freqs)]))
        file.write("\n")
        for i in range(num_samples):
            values = [t[i], y[i]]
            for f in range(num_freqs):
                values.append(basis[f][i].real)

            for f in range(num_freqs):
                values.append(recon[f][i].real)

            file.write("\t".join([str(val) for val in values]))
            file.write("\n")


if __name__ == "__main__":
    _main()
