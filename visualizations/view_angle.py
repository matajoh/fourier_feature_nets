import os
import cv2
import numpy as np
import fourier_feature_nets as ffn


def _camera_angle(source: ffn.CameraInfo, dest: ffn.CameraInfo) -> float:
    angle0 = source.position / np.linalg.norm(source.position, keepdims=True)
    angle1 = dest.position / np.linalg.norm(dest.position, keepdims=True)
    return (angle0 * angle1).sum()


def _main():
    output_dir = os.path.join("results","view_angle")
    os.makedirs(output_dir, exist_ok=True)
    dataset = ffn.ImageDataset.load("trex_400.npz", "train", 256, True, True)
    model = ffn.load_model("trex_800_nerf.pt")
    image = dataset.images[1].astype(np.float32) / 255
    image = image[..., :3] * image[..., 3:]
    image = (image * 255).astype(np.uint8)
    row = 310
    col = 137
    camera = dataset.cameras[1]
    index = dataset.sampler.rays_per_camera + row * 400 + col
    rays = dataset.sampler.sample([index], None)
    raycaster = ffn.Raycaster(model)
    render = raycaster.render(rays, True)
    start = dataset.sampler.starts[index].numpy()
    direction = dataset.sampler.directions[index].numpy()
    depth = render.depth[0].item()
    
    position = start + direction * depth
    source_cam = dataset.cameras[1]
    index = 0
    for camera, image in zip(dataset.cameras, dataset.images):
        angle = _camera_angle(source_cam, camera)
        if angle < 0.5:
            continue

        print(index, angle)
        image = image.astype(np.float32) / 255
        image = image[..., :3] * image[..., 3:]
        image = (image * 255).astype(np.uint8)

        col, row = camera.project(position[np.newaxis])[0]
        col = int(col - 16)
        row = int(row - 16)
        patch = image[row:row+32, col:col+32]
        patch = cv2.resize(patch, (128, 128), cv2.INTER_NEAREST)

        frame = np.zeros((400, 800, 3), np.uint8)
        frame[:, :400] = image
        frame[136:264, 536:664] = patch

        frame = cv2.rectangle(frame, (col, row), (col+32, row+32),
                              (255, 255, 255), 2)
        frame = cv2.rectangle(frame, (536, 136), (664, 264),
                              (255, 255, 255), 2)
        frame = cv2.line(frame, (col+32, row), (536, 136),
                         (255, 255, 255), 2)
        frame = cv2.line(frame, (col+32, row+32), (536, 264),
                         (255, 255, 255), 2)

        path = "frame_{:04d}.png".format(index)
        path = os.path.join(output_dir, path)
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        index += 1


if __name__ == "__main__":
    _main()
