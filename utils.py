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
import os
import random
from collections import OrderedDict


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def denorm(tensor, mean, std):
    tensor *= torch.Tensor(std)[None, :, None, None]
    tensor += torch.Tensor(mean)[None, :, None, None]
    return tensor


def _batched_image_to_grid(image, n_cols):
    b, _, h, w = image.shape
    assert b % n_cols == 0,\
        "The batch size should be a multiple of `n_cols` argument"
    pad = max(2, int(max(h, w) * 0.016))
    grid = make_grid(tensor=image, nrow=n_cols, normalize=False, padding=pad)
    grid = grid.clone().permute((1, 2, 0)).detach().cpu().numpy()
    grid *= 255
    grid = np.clip(a=grid, a_min=0, a_max=255).astype("uint8")

    for k in range(n_cols + 1):
        grid[:, (pad + h) * k: (pad + h) * k + pad, :] = 255
    for k in range(b // n_cols + 1):
        grid[(pad + h) * k: (pad + h) * k + pad, :, :] = 255
    return grid


def image_to_grid(x, y, x_mean, x_std, y_mean, y_std, n_cols):
    x = x.detach().cpu()
    y = y.detach().cpu()

    x = denorm(x, mean=x_mean, std=x_std)
    y = denorm(y, mean=y_mean, std=y_std)

    images = [x, y]
    gen_image = rearrange(
        torch.cat(images, dim=0), pattern="(n m) c h w -> (m n) c h w", n=len(images),
    )
    grid = _batched_image_to_grid(gen_image, n_cols=n_cols)
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


def set_requires_grad(models, grad):
    for model in models:
        for p in model.parameters():
            p.requires_grad = grad


class ImageBuffer(object):
# "To reduce model oscillation we update the discriminators using a history of generated images rather than the
# ones produced by the latest generators. We keep an image buffer that stores the 50 previously created images."
    def __init__(self, buffer_size, stored_images=list()):
        self.buffer_size = buffer_size

        self.stored_images = stored_images
        self._cnt = len(stored_images)

    def __call__(self, image):
        images_to_return = list()
        for unbatched_image in image:
            if self._cnt < self.buffer_size:
                self.stored_images.append(unbatched_image)
                self._cnt += 1
                images_to_return.append(unbatched_image)
            else: # buffer가 가득 찼다면
                if random.random() > 0.5: # 50%의 확률로
                    idx = random.randrange(len(self.stored_images))
                    images_to_return.append(self.stored_images[idx].clone()) # buffer에서 하나의 이미지를 빼고
                    self.stored_images[idx] = unbatched_image # 새로운 이미지를 저장합니다.
                else: # 다른 50%의 확률로
                    images_to_return.append(unbatched_image) # 입력 받은 이미지를 그대로 출력합니다.
        new_image = torch.stack(images_to_return, dim=0)
        return new_image


def _modify_state_dict(state_dict, keyword="_orig_mod."):
    new_state_dict = OrderedDict()
    for old_key in list(state_dict.keys()):
        if old_key and old_key.startswith(keyword):
            new_key = old_key[len(keyword):]
        else:
            new_key = old_key
        new_state_dict[new_key] = state_dict[old_key]
    return new_state_dict
