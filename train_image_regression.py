"""Script which trains 2D Fourier networks to predict image pixels."""

import argparse
import os

import cv2
import nerf
import numpy as np
import torch
import torch.optim


try:
    from azureml.core import Run
except ImportError:
    Run = None
    print("Unable to import AzureML, running as local experiment")


def _parse_args():
    parser = argparse.ArgumentParser("NeRF2D Image Trainer")
    parser.add_argument("image_path", help="Path to an image file")
    parser.add_argument("nerf_model", choices=["mlp", "basic",
                                               "positional", "gaussian"])
    parser.add_argument("results_dir", help="Path to the results directory")
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
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate for the optimizer")
    parser.add_argument("--report-interval", type=int, default=50,
                        help="Frequency of logging")
    return parser.parse_args()


def _main():
    args = _parse_args()

    if Run:
        run = Run.get_context()
    else:
        run = None

    is_offline = run is None or run.id.startswith("OfflineRun")

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    print("Creating dataset...")
    dataset = nerf.PixelDataset.create(args.image_path,
                                       args.color_space,
                                       args.image_size)
    if dataset is None:
        print("Dataset unavailable, exiting.")
        exit(1)

    frame = np.zeros((args.image_size, 2*args.image_size, 3), np.uint8)
    frame[:, :args.image_size] = dataset.image

    dataset = dataset.to("cuda")

    if args.nerf_model == "mlp":
        model = nerf.MLP(2, 3,
                         num_channels=args.num_channels,
                         output_act=True)
    elif args.nerf_model == "basic":
        model = nerf.BasicFourierMLP(2, 3,
                                     num_channels=args.num_channels,
                                     output_act=True)
    elif args.nerf_model == "positional":
        model = nerf.PositionalFourierMLP(2, 3,
                                          sigma=args.pos_sigma,
                                          num_channels=args.num_channels,
                                          num_frequencies=args.num_frequencies,
                                          output_act=True)
    elif args.nerf_model == "gaussian":
        model = nerf.GaussianFourierMLP(2, 3,
                                        sigma=args.gauss_sigma,
                                        num_channels=args.num_channels,
                                        num_frequencies=args.num_frequencies,
                                        output_act=True)
    else:
        raise NotImplementedError("Unsupported model: {}".format(args.nerf_model))

    model = model.to("cuda")
    optim = torch.optim.Adam(model.parameters(), args.learning_rate)
    for step in range(args.num_steps):
        if step % args.report_interval == 0:
            with torch.no_grad():
                output = model(dataset.val_uv)
                psnr = dataset.psnr(output)
                print("step", step, dataset.psnr(output))
                frame[:, args.image_size:] = dataset.to_image(output)
                image_path = os.path.join(args.results_dir,
                                          "val{:05}.png".format(step))
                cv2.imwrite(image_path, frame)
                if run:
                    run.log("psnr", psnr)

                if is_offline:
                    cv2.imshow("progress", frame)
                    cv2.waitKey(1)

        optim.zero_grad()
        output = model(dataset.train_uv)
        train_loss = torch.square(output - dataset.train_color).mean()
        train_loss.backward()
        optim.step()

    with torch.no_grad():
        output = model(dataset.val_uv)
        print("final", dataset.psnr(output))
        frame[:, args.image_size:] = dataset.to_image(output)
        image_path = os.path.join(args.results_dir,
                                  "val{:05}.png".format(args.num_steps))
        cv2.imwrite(image_path, frame)

        uvs = nerf.PixelDataset.generate_uvs(args.image_size * 2, "cpu")
        model = model.to("cpu")
        output = model(uvs)
        image = dataset.to_image(output, args.image_size * 2)
        final_path = os.path.join(args.results_dir, "superres.png")
        cv2.imwrite(final_path, image)
        if run:
            run.log("psnr", psnr)

        if is_offline:
            cv2.imshow("progress", frame)
            cv2.imshow("super-resolution", image)
            cv2.waitKey()

    model_path = os.path.join(args.results_dir, "model.pkl")
    model.save(model_path)


if __name__ == "__main__":
    _main()
