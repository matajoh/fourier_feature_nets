import numpy as np


def func(t):
    return np.sin(t) + 0.5*np.sin(2*t) - 0.2*np.cos(5*t) + 2


def _main():
    num_samples = 64
    t = np.linspace(-np.pi, np.pi, num_samples)
    y = func(t)
    Y = np.fft.fft(y)
    num_freqs = 4
    order = np.argsort(np.abs(Y[:num_samples // 2]))[::-1]
    freqs = np.fft.fftfreq(num_samples)
    basis = []
    recon = []
    for i in range(num_freqs):
        subset = np.zeros_like(Y)
        f = order[i]
        if f > 0:
            print(freqs[f] * num_samples, "hz", Y[f].real, Y[f].imag, Y[64-f].real, Y[64-f].imag)

        subset[f] = Y[f]
        basis.append(np.fft.ifft(subset).real)
        for f in order[:i]:
            subset[f] = Y[f]

        if subset[0]:
            offset = subset[0] / num_samples
            subset[0] = 0

        recon.append(2 * np.fft.ifft(subset).real + offset)

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
