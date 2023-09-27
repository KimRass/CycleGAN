import torch
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from pathlib import Path
from tqdm.auto import tqdm
import argparse

import config
from utils import save_image
from model import Generator


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    gen = Generator().to(config.DEVICE)
    ckpt = torch.load(args.ckpt_path, map_location=config.DEVICE)
    gen.load_state_dict(ckpt)

    # ### Generate images
    # gen.eval()
    # with torch.no_grad():
    #     for idx in tqdm(range(1, args.n_images + 1)):
    #         noise = torch.randn(9, 512, 1, 1, device=DEVICE)
    #         fake_image = gen(noise, img_size=args.img_size, alpha=1)

    #         fake_image = fake_image.detach().cpu()
    #         grid = make_grid(
    #             fake_image, nrow=3, padding=4, normalize=True, value_range=(-1, 1), pad_value=1,
    #         )
    #         grid = TF.to_pil_image(grid)
    #         save_path = Path(__file__).parent/\
    #             f"""generated_images/{args.img_size}×{args.img_size}_{idx}.jpg"""
    #         save_image(grid, path=save_path)
