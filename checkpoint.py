import argparse
import torch
from pathlib import Path

from utils import get_device, _modify_state_dict


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()
    return args


def save_gens(ckpt_path, save_dir, device):
    state_dict = torch.load(ckpt_path, map_location=device)
    gen_x_state_dict = _modify_state_dict(state_dict["Gx"])
    gen_y_state_dict = _modify_state_dict(state_dict["Gy"])
    torch.save(gen_x_state_dict, str(f"{Path(save_dir)/Path(ckpt_path).stem}_Gx.pth"))
    torch.save(gen_y_state_dict, str(f"{Path(save_dir)/Path(ckpt_path).stem}_Gy.pth"))


if __name__ == "__main__":
    args = get_args()

    DEVICE = get_device()
    save_gens(ckpt_path=args.ckpt_path, save_dir=args.save_dir, device=DEVICE)
