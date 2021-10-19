"""Script which trains 2D Fourier networks to predict image pixels."""

import argparse

import cv2
import nerf
import numpy as np
import torch
import torch.optim


def _parse_args():
    parser = argparse.ArgumentParser("NeRF2D Image Trainer")
    parser.add_argument("image_path", help="Path to an image file")
    parser.add_argument("nerf_model", choices=["mlp", "basic",
                                               "positional", "gaussian"])
    parser.add_argument("--image-size", type=int, default=512,
                        help="Size of the square input image")
    parser.add_argument("--color-space", choices=["YCrCb", "RGB"],
                        default="RGB")
    parser.add_argument("--num-channels", type=int, default=256,
                        help="Number of channels in the MLP")
    parser.add_argument("--num-frequencies", type=int, default=256,
                        help="Number of frequencies used for encoding")
    parser.add_argument("--pos-sigma", type=float, default=6,
                        help="Value of sigma for the positional model")
    parser.add_argument("--gauss-sigma", type=float, default=10,
                        help="Value of sigma for the gaussian model")
    parser.add_argument("--num-steps", type=int, default=2000)
    return parser.parse_args()


def _main():
    args = _parse_args()

    print("Creating dataset...")
    dataset = nerf.PixelDataset.create(args.image_path,
                                       args.color_space,
                                       args.image_size)
    frame = np.zeros((512, 1024, 3), np.uint8)
    frame[:, :512] = dataset.image

    dataset = dataset.to("cuda")

    if args.nerf_model == "mlp":
        model = nerf.MLP(2, 3, args.num_channels,
                         output_act=torch.sigmoid)
    elif args.nerf_model == "basic":
        model = nerf.BasicFourierMLP(2, 3, args.num_channels,
                                     output_act=torch.sigmoid)
    elif args.nerf_model == "positional":
        model = nerf.PositionalFourierMLP(2, 3, args.pos_sigma,
                                          args.num_channels,
                                          args.num_frequencies,
                                          output_act=torch.sigmoid)
    elif args.nerf_model == "gaussian":
        model = nerf.GaussianFourierMLP(2, 3, args.gauss_sigma,
                                        args.num_channels,
                                        args.num_frequencies,
                                        output_act=torch.sigmoid)
    else:
        raise NotImplementedError("Unsupported model: {}".format(args.nerf_model))

    model = model.to("cuda")
    optim = torch.optim.Adam(model.parameters(), 1e-3)
    for step in range(args.num_steps):
        if step % 25 == 0:
            with torch.no_grad():
                output = model(dataset.val_uv)
                print("step", step, dataset.psnr(output))
                frame[:, 512:] = dataset.to_image(output)
                cv2.imshow("progress", frame)
                cv2.waitKey(1)

        optim.zero_grad()
        output = model(dataset.train_uv)
        train_loss = torch.square(output - dataset.train_color).sum()
        train_loss.backward()
        optim.step()

    with torch.no_grad():
        output = model(dataset.val_uv)
        print("final", dataset.psnr(output))
        frame[:, 512:] = dataset.to_image(output)
        cv2.imshow("progress", frame)

        uvs = nerf.PixelDataset.generate_uvs(1024, "cpu")
        model = model.to("cpu")
        output = model(uvs)
        image = dataset.to_image(output, 1024)
        cv2.imshow("super-resolution", image)
        cv2.waitKey()


if __name__ == "__main__":
    _main()
