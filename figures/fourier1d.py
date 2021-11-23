import numpy as np
import fourier_feature_nets as ffn
import torch


def func(t):
    return np.sin(t) + 0.5*np.sin(2*t) - 0.2*np.cos(5*t) + 2

num_samples = 32
sample_rate = 8
num_layers = 1
num_channels = 64


dataset = ffn.SignalDataset.create(func, num_samples, sample_rate)
model = ffn.MLP(1, 1, num_layers, num_channels)
model.layers[-1].bias.data = dataset.train_y.mean()
torch.nn.init.xavier_normal_(model.layers[-1].weight, dataset.train_y.std())


def _main():
    num_samples = 64
    t = np.linspace(0, 2, num_samples)
    y = func(t*np.pi)
    Y = np.fft.fft(y)
    num_freqs = 4
    order = np.argsort(np.abs(Y[:num_samples // 2]))[::-1]
    freqs = np.fft.fftfreq(num_samples)
    basis = []
    recon = []
    for i in range(num_freqs):
        subset = np.zeros_like(Y)
        f = order[i]
        print(freqs[f] * num_samples, "hz", Y[f].real / num_samples,
              Y[f].imag / num_samples, Y[-f].real / num_samples,
              Y[-f].imag / num_samples)

        subset[f] = Y[f]
        subset[-f] = Y[-f]
        basis.append(np.fft.ifft(subset).real)
        for f in order[:i]:
            subset[f] = Y[f]
            subset[-f] = Y[-f]

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
