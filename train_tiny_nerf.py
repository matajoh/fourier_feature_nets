from nerf import GaussianFourierMLP, OcTree, RaySamplingDataset
import torch
import torch.optim
from torch.utils.data import DataLoader


def _train(dataset: RaySamplingDataset, model: GaussianFourierMLP,
           batch_size: int, learning_rate: float, num_epochs: int,
           device: str):
    optim = torch.optim.Adam(model.parameters(), 5e-4)
    model = model.to(device)

    data_loader = DataLoader(dataset, batch_size, True)
    max_dist = torch.full((batch_size, 1), 1e10, dtype=torch.float32)
    max_dist = max_dist.to(device)
    left_trans = torch.ones_like(max_dist)
    for i in range(num_epochs):
        print("Epoch", i)

        for positions, _, deltas, targets in data_loader:
            positions = positions.to(device)
            deltas = deltas.to(device)
            targets = targets.to(device)

            rgb_o = model(positions)
            rgb, opacity = torch.split(rgb_o, [3, 1], -1)

            deltas = torch.cat([deltas, max_dist], axis=-1)
            alpha = 1 - torch.exp(-(opacity * deltas))
            ones = torch.ones_like(alpha)
            trans = torch.minimum(ones, 1 - alpha + 1e-10)
            _, trans = trans.split([1, dataset.num_samples - 1], dim=-1)
            trans = torch.cat([left_trans, trans], -1)
            weights = alpha * torch.cumprod(trans, -1)
            outputs = weights * rgb
            outputs = outputs.sum(-2)

            loss = (outputs - targets).square().sum()
            loss.backward()
            optim.step()


def _main():
    pass


if __name__ == "__main__":
    _main()
