import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.auto import tqdm
import argparse

import config
from utils import image_to_grid, save_image, _modify_state_dict
from model import Generator
from data import OneSideImageDataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ds_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--x_or_y", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)

    args = parser.parse_args()
    return args


def _df_name_to_ls(ds_name):
    return ds_name.split("2")


def _ls_to_str(ls):
    return "_to_".join(ls)


def get_dir_name(ds_name, x_or_y):
    ls = _df_name_to_ls(ds_name)
    if x_or_y == "y":
        ls = reversed(ls)
    dir_name = _ls_to_str(ls)
    return dir_name


if __name__ == "__main__":
    PARENT_DIR = Path(__file__).parent.resolve()

    args = get_args()

    DIR_NAME = get_dir_name(ds_name=args.ds_name, x_or_y=args.x_or_y)

    G = Generator().to(config.DEVICE)
    state_dict = torch.load(args.ckpt_path, map_location=config.DEVICE)
    state_dict = _modify_state_dict(state_dict)
    G.load_state_dict(state_dict)

    test_ds = OneSideImageDataset(
        data_dir=args.data_dir,
        x_or_y=args.x_or_y,
        mean=config.X_MEAN if args.x_or_y == "x" else config.Y_MEAN,
        std=config.X_STD if args.x_or_y == "x" else config.Y_STD,
        split="test",
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.n_cpus,
        pin_memory=False,
        drop_last=False,
    )

    # ### Generate images
    G.eval()
    for idx, real in enumerate(tqdm(test_dl), start=1):
        with torch.no_grad():
            fake = G(real)
        grid = image_to_grid(
            x=real,
            y=fake,
            x_mean=config.X_MEAN,
            x_std=config.X_STD,
            y_mean=config.Y_MEAN,
            y_std=config.Y_STD,
            n_cols=2,
        )
        save_image(grid, path=f"{PARENT_DIR}/generated_images/{DIR_NAME}/{idx}.jpg")
