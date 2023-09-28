import torch
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.auto import tqdm
import argparse

import config
from utils import images_to_grid, save_image
from model import Generator
from dataset import UnpairedImageDataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ds_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--gen_x_ckpt_path", type=str, required=True)
    parser.add_argument("--gen_y_ckpt_path", type=str, required=True)
    parser.add_argument("--n_workers", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    PARENT_DIR = Path(__file__).parent

    args = get_args()

    gen_x = Generator().to(config.DEVICE)
    gen_x_ckpt = torch.load(args.gen_x_ckpt_path, map_location=config.DEVICE)
    gen_x.load_state_dict(gen_x_ckpt)
    gen_x.eval()

    gen_y = Generator().to(config.DEVICE)
    gen_y_ckpt = torch.load(args.gen_y_ckpt_path, map_location=config.DEVICE)
    gen_y.load_state_dict(gen_y_ckpt)
    gen_y.eval()

    test_ds = UnpairedImageDataset(
        data_dir=args.data_dir,
        x_mean=config.X_MEAN,
        x_std=config.X_STD,
        y_mean=config.Y_MEAN,
        y_std=config.Y_STD,
        split="test",
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=False,
        drop_last=False,
    )
    # ### Generate images
    for idx, (real_x, real_y) in enumerate(tqdm(test_dl), start=1):
        with torch.no_grad():
            fake_y = gen_x(real_x)
        forward_grid = images_to_grid(
            x=real_x,
            y=fake_y,
            x_mean=config.X_MEAN,
            x_std=config.X_STD,
            y_mean=config.Y_MEAN,
            y_std=config.Y_STD,
        )
        save_image(
            forward_grid, path=f"{PARENT_DIR}/generated_images/{args.ds_name}/{idx}_forward.jpg",
        )

        with torch.no_grad():
            fake_x = gen_y(real_y)
        backward_grid = images_to_grid(
            x=real_y,
            y=fake_x,
            x_mean=config.X_MEAN,
            x_std=config.X_STD,
            y_mean=config.Y_MEAN,
            y_std=config.Y_STD,
        )
        save_image(
            backward_grid, path=f"{PARENT_DIR}/generated_images/{args.ds_name}/{idx}_backward.jpg",
        )
