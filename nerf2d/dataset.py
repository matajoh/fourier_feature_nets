
from collections import namedtuple
import math

import cv2
import numpy as np
import torch


PixelData = namedtuple("DataTensors", ["uv", "color"])

class PixelDataset:
    def __init__(self, size: int, color_space: str, train_data: PixelData, val_data: PixelData):
        self.size = size
        self.color_space = color_space
        self.image = self.to_image(val_data.color)
        self.train_uv, self.train_color = train_data
        self.val_uv, self.val_color = val_data

    @staticmethod
    def create(path: str, color_space: str, size=512) -> "PixelDataset":
        pixels = cv2.imread(path)
        if pixels.shape[0] > pixels.shape[1]:
            start = (pixels.shape[0] - pixels.shape[1]) // 2
            end = start + pixels.shape[1]
            pixels = pixels[start:end, :]
        elif pixels.shape[1] > pixels.shape[0]:
            start = (pixels.shape[1] - pixels.shape[0]) // 2
            end = start + pixels.shape[0]
            pixels = pixels[:, start:end]
        
        if pixels.shape[0] != size:
            sigma = pixels.shape[0] / size
            pixels = cv2.GaussianBlur(pixels, (0, 0), sigma)
            pixels = cv2.resize(pixels, (size, size), cv2.INTER_NEAREST)

        if color_space == "YCrCb":
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2YCrCb) / 255
        elif color_space == "RGB":
            pixels = pixels / 255
        else:
            raise NotImplementedError("Unsupported color space: {}".format(color_space))

        train_uv = []
        train_color = []
        val_uv = []
        val_color = []
        for row in range(size):
            u = (2 * (row + 0.5) / size) + 1
            for col in range(size):
                v = (2 * (col + 0.5) / size) + 1
                color = pixels[row, col].tolist()
                val_uv.append((u, v))
                val_color.append(color)
                if col % 2 or row % 2:
                    train_uv.append((u, v))
                    train_color.append(color)

        train_data = PixelData(torch.FloatTensor(train_uv), torch.FloatTensor(train_color))
        val_data = PixelData(torch.FloatTensor(val_uv), torch.FloatTensor(val_color))
        return PixelDataset(size, color_space, train_data, val_data)
    
    def to(self, *args) -> "PixelDataset":
        train_data = PixelData(self.train_uv.to(*args), self.train_color.to(*args))
        val_data = PixelData(self.val_uv.to(*args), self.val_color.to(*args))
        return PixelDataset(self.size, self.color_space, train_data, val_data)

    def to_image(self, colors: torch.Tensor, size = 0) -> np.ndarray:
        if size == 0:
            size = self.size

        pixels = (colors * 255).reshape(size, size, 3).cpu().numpy().astype(np.uint8)
        if self.color_space == "YCrCb":
            pixels = cv2.cvtColor(pixels, cv2.COLOR_YCrCb2BGR)

        return pixels

    @staticmethod
    def generate_uvs(size: int, device) -> torch.Tensor:
        uvs = []
        for row in range(size):
            u = (2 * (row + 0.5) / size) + 1
            for col in range(size):
                v = (2 * (col + 0.5) / size) + 1
                uvs.append((u, v))
        
        return torch.FloatTensor(uvs).to(device=device)

    def psnr(self, outputs: torch.Tensor) -> float:
        mse = torch.square(255 * (outputs - self.val_color)).mean().item()
        return 20 * math.log10(255) - 10 * math.log10(mse)
