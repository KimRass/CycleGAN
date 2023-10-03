# References:
    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/image_pool.py

import torch
from torchvision.utils import make_grid
from einops import rearrange
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import timedelta
from time import time
import random

import config


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def denorm(tensor, mean, std):
    tensor *= torch.Tensor(std)[None, :, None, None]
    tensor += torch.Tensor(mean)[None, :, None, None]
    return tensor


def _batched_image_to_grid(image, n_cols):
    b, _, h, w = image.shape
    assert b % n_cols == 0,\
        "The batch size should be a multiple of `n_cols` argument"
    pad = max(2, int(max(h, w) * 0.02))
    grid = make_grid(tensor=image, nrow=n_cols, normalize=False, padding=pad)
    grid = grid.clone().permute((1, 2, 0)).detach().cpu().numpy()

    grid *= 255
    grid = np.clip(a=grid, a_min=0, a_max=255).astype("uint8")

    for k in range(n_cols + 1):
        grid[:, (pad + h) * k: (pad + h) * k + pad, :] = 255
    for k in range(b // n_cols + 1):
        grid[(pad + h) * k: (pad + h) * k + pad, :, :] = 255
    return grid


def images_to_grid(x, y, x_mean, x_std, y_mean, y_std):
    x = x.detach().cpu()
    y = y.detach().cpu()

    x = denorm(x, mean=x_mean, std=x_std)
    y = denorm(y, mean=y_mean, std=y_std)

    images = [x, y]
    gen_image = rearrange(
        torch.cat(images, dim=0), pattern="(n m) c h w -> (m n) c h w", n=len(images),
    )
    grid = _batched_image_to_grid(gen_image, n_cols=len(images))
    return grid


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def save_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _to_pil(image).save(str(path), quality=100)


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad = True


class ImageBuffer(object):
# "To reduce model oscillation we update the discriminators using a history of generated images rather than the
# ones produced by the latest generators. We keep an image buffer that stores the 50 previously created images."
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size

        self.images = list()
        self._cnt = 0

    def __call__(self, image):
        images_to_return = list()
        for unbatched_image in image:
            if self._cnt < self.buffer_size:
                self.images.append(unbatched_image)
                self._cnt += 1
                images_to_return.append(unbatched_image)
            else: # buffer가 가득 찼다면
                if random.random() > 0.5: # 50%의 확률로
                    idx = random.randrange(len(self.images))
                    images_to_return.append(self.images[idx]) # buffer에서 하나의 이미지를 빼고
                    self.images[idx] = unbatched_image # 새로운 이미지를 저장합니다.
                else:
                    images_to_return.append(unbatched_image)
        new_image = torch.stack(images_to_return, dim=0)
        return new_image
