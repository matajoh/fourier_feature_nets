"""Script which trains 2D Fourier networks to predict image pixels."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os

import cv2
import fourier_feature_nets as ffn
import numpy as np
import scenepic as sp
import torch
import torch.optim


try:
    from azureml.core import Run
except ImportError:
    Run = None
    print("Unable to import AzureML, running as local experiment")


def _parse_args():
    parser = ArgumentParser("NeRF2D Image Trainer",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("image_path", help="Path to an image file")
    parser.add_argument("nerf_model", choices=["mlp", "basic",
                                               "positional", "gaussian"])
    parser.add_argument("results_dir", help="Path to the results directory")
    parser.add_argument("--activations", action="store_true",
                        help="Produce activation visualizations")
    parser.add_argument("--vertical", action="store_true",
                        help="Whether to stack the images vertically")
    parser.add_argument("--omit-gt", action="store_true",
                        help="whether to omit the GT image from the display")
    parser.add_argument("--image-size", type=int, default=512,
                        help="Size of the square input image")
    parser.add_argument("--color-space", choices=["YCrCb", "RGB"],
                        default="RGB", help="Color space to use for learning")
    parser.add_argument("--num-channels", type=int, default=256,
                        help="Number of channels in the MLP")
    parser.add_argument("--embedding_size", type=int, default=256,
                        help="Embedding size used for encoding")
    parser.add_argument("--pos-max-log-scale", type=float, default=6,
                        help="Max log scale for the positional encoding")
    parser.add_argument("--gauss-sigma", type=float, default=10,
                        help="Standard deviation for the gaussian encoding")
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate for the optimizer")
    parser.add_argument("--report-interval", type=int, default=50,
                        help="Frequency of logging")
    parser.add_argument("--make-video", action="store_true",
                        help="Whether to produce an MP4 of training.")
    parser.add_argument("--decay-rate", type=float, default=0.1,
                        help="Decay rate for the learning rate.")
    parser.add_argument("--decay-steps", type=int, default=2500,
                        help="Interval over which the rate should decay")
    parser.add_argument("--device", default="cuda",
                        help="Pytorch compute device")
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
    dataset = ffn.PixelDataset.create(args.image_path,
                                      args.color_space,
                                      args.image_size)
    if dataset is None:
        print("Dataset unavailable, exiting.")
        exit(1)

    dataset = dataset.to(args.device)

    if args.nerf_model == "mlp":
        model = ffn.MLP(2, 3, num_channels=args.num_channels)
    elif args.nerf_model == "basic":
        model = ffn.BasicFourierMLP(2, 3, num_channels=args.num_channels)
    elif args.nerf_model == "positional":
        model = ffn.PositionalFourierMLP(2, 3,
                                         max_log_scale=args.pos_max_log_scale,
                                         num_channels=args.num_channels,
                                         embedding_size=args.embedding_size)
    elif args.nerf_model == "gaussian":
        model = ffn.GaussianFourierMLP(2, 3,
                                       sigma=args.gauss_sigma,
                                       num_channels=args.num_channels,
                                       embedding_size=args.embedding_size)
    else:
        raise NotImplementedError("Unsupported model: {}".format(args.nerf_model))

    if args.omit_gt and not args.activations:
        width = args.image_size
        height = args.image_size
    elif args.vertical:
        width = args.image_size
        height = 2 * args.image_size
    else:
        width = 2 * args.image_size
        height = args.image_size

    frame = np.zeros((height, width, 3), np.uint8)

    if not args.omit_gt:
        if args.vertical:
            frame[:args.image_size, :] = dataset.image
        else:
            frame[:, :args.image_size] = dataset.image

    if args.make_video:
        writer = sp.VideoWriter(os.path.join(args.results_dir, "training.mp4"),
                                (width, height), framerate=5)
        writer.font_scale = 2
        writer.start()
    else:
        writer = None

    model = model.to(args.device)
    optim = torch.optim.Adam(model.parameters(), args.learning_rate)
    for step in range(args.num_steps + 1):
        if step % args.report_interval == 0 or step == args.num_steps:
            with torch.no_grad():
                model.eval()
                batch_rows = args.image_size // 4
                output = []
                for i in range(4):
                    start = i * batch_rows
                    end = start + batch_rows
                    output.append(model(dataset.val_uv[start:end]))

                output = torch.sigmoid(torch.cat(output))
                psnr_val = dataset.psnr(output)
                print("step", step, "val:", psnr_val, "lr:",
                      optim.param_groups[0]["lr"])
                image = dataset.to_image(output, args.image_size)
                if args.omit_gt and not args.activations:
                    frame[:] = image
                elif args.vertical:
                    frame[args.image_size:, :] = image
                else:
                    frame[:, args.image_size:] = image

                if args.activations:
                    act_image = dataset.to_act_image(model, args.image_size)
                    if args.vertical:
                        frame[:args.image_size, :] = act_image
                    else:
                        frame[:, :args.image_size] = act_image

                model.train()
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                image_path = os.path.join(args.results_dir,
                                          "val{:05}.png".format(step))
                cv2.imwrite(image_path, bgr_frame)

                if writer:
                    if not args.activations:
                        writer.text = "{:04d}: {:.02f}".format(step, psnr_val)

                    writer.frame[:] = bgr_frame
                    writer.write_frame()

                if run:
                    run.log("psnr_val", psnr_val)

                if is_offline:
                    cv2.imshow("progress", bgr_frame)
                    cv2.waitKey(1)

        ffn.exponential_lr_decay(optim, args.learning_rate, step,
                                 args.decay_rate, args.decay_steps)
        optim.zero_grad()
        output = torch.sigmoid(model(dataset.train_uv))
        train_loss = 0.5 * torch.square(output - dataset.train_color).mean()
        train_loss.backward()
        optim.step()

    with torch.no_grad():
        uvs = ffn.PixelDataset.generate_uvs(args.image_size * 2, "cpu")
        model.to("cpu")
        model.eval()
        output = torch.sigmoid(model(uvs))
        image = dataset.to_image(output, args.image_size * 2)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        path = os.path.join(args.results_dir, "superres.png")
        cv2.imwrite(path, image)

    model_path = os.path.join(args.results_dir, "model.pt")
    model.save(model_path)

    if writer:
        writer.stop()


if __name__ == "__main__":
    _main()
