import argparse
from nerf2d.models import PositionalNeRF2d

import cv2
from nerf2d import PositionalNeRF2d
import numpy as np
import torch
import torch.optim


def _main():
    print("Creating dataset...")
    pixels = cv2.imread("D:\\Temp\\imogen.jpg")
    frame = np.zeros((512, 1024, 3), np.uint8)
    frame[:, :512] = pixels

    pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2YCrCb) / 255
    train_uv = []
    train_color = []
    val_uv = []
    val_color = []
    for row in range(512):
        u = ((row + 0.5) / 256) + 1
        for col in range(512):
            v = ((col + 0.5) / 256) + 1
            color = pixels[row, col].tolist()
            val_uv.append((u, v))
            val_color.append(color)
            if col % 2 or row % 2:
                train_uv.append((u, v))
                train_color.append(color)

    train_uv = torch.FloatTensor(train_uv).to("cuda")
    train_color = torch.FloatTensor(train_color).to("cuda")
    val_uv = torch.FloatTensor(val_uv).to("cuda")
    val_color = torch.FloatTensor(val_color).to("cuda")
    model = PositionalNeRF2d().to("cuda")

    optim = torch.optim.Adam(model.parameters(), 1e-3)
    for step in range(2000):
        if step % 25 == 0:
            with torch.no_grad():
                output = model(val_uv)
                pixels = (output * 255).reshape(512, 512, 3).cpu().numpy().astype(np.uint8)
                frame[:, 512:] = cv2.cvtColor(pixels, cv2.COLOR_YCrCb2BGR)
                cv2.imshow("progress", frame)
                cv2.waitKey(1)
                val_loss = torch.square(output - val_color).sum()

            print("step", step, val_loss.item())

        optim.zero_grad()
        output = model(train_uv)
        train_loss = torch.square(output - train_color).sum()
        train_loss.backward()
        optim.step()

    with torch.no_grad():
        output = model(val_uv)
        pixels = (output * 255).reshape(512, 512, 3).cpu().numpy().astype(np.uint8)
        frame[:, 512:] = cv2.cvtColor(pixels, cv2.COLOR_YCrCb2BGR)
        val_loss = torch.square(output - val_color).sum()
        print("final", val_loss.item())
        cv2.imshow("progress", frame)
        cv2.waitKey()



if __name__ == "__main__":
    _main()
