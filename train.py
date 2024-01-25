# References:
    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from pathlib import Path
import argparse
from time import time
import wandb

import config
from model import Generator, Discriminator
from data import UnpairedImageDataset, OneSideImageDataset
from utils import (
    set_seed,
    image_to_grid,
    save_image,
    get_elapsed_time,
    set_requires_grad,
    ImageBuffer,
    _modify_state_dict,
)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ds_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_cpus", type=int, required=True)
    parser.add_argument("--test_batch_size", type=int, required=True)
    parser.add_argument("--run_id", type=str, required=False)

    args = parser.parse_args()
    return args


def get_dls(data_dir, train_batch_size, test_batch_size, n_cpus, fixed_pairs):
    train_ds = UnpairedImageDataset(
        data_dir=data_dir,
        x_mean=config.X_MEAN,
        x_std=config.X_STD,
        y_mean=config.Y_MEAN,
        y_std=config.Y_STD,
        split="train",
        fixed_pairs=fixed_pairs,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=n_cpus,
        pin_memory=True,
        drop_last=True,
    )

    x_test_ds = OneSideImageDataset(
        data_dir=data_dir,
        x_or_y="x",
        mean=config.X_MEAN,
        std=config.X_STD,
        split="test",
    )
    x_test_dl = DataLoader(
        x_test_ds,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=n_cpus,
        pin_memory=True,
        drop_last=False,
    )

    y_test_ds = OneSideImageDataset(
        data_dir=data_dir,
        x_or_y="y",
        mean=config.Y_MEAN,
        std=config.Y_STD,
        split="test",
    )
    y_test_dl = DataLoader(
        y_test_ds,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=n_cpus,
        pin_memory=True,
        drop_last=False,
    )
    return train_dl, x_test_dl, y_test_dl


def get_models(device):
    Dx = Discriminator().to(device)
    Dy = Discriminator().to(device)
    Gx = Generator().to(device)
    Gy = Generator().to(device)

    Dx = torch.compile(Dx)
    Dy = torch.compile(Dy)
    Gx = torch.compile(Gx)
    Gy = torch.compile(Gy)
    return Dx, Dy, Gx, Gy


def get_optims(Dx, Dy, Gx, Gy):
    # "We use the Adam solver."
    # 논문에는 learning rate에 관한 얘기만 나오지만 공식 저장소를 그대로 따라 `betas=(0.5, 0.999)`를 설정했습니다. 
    D_optim = Adam(
        list(Dx.parameters()) + list(Dy.parameters()), lr=config.LR, betas=(config.BETA1, config.BETA2),
    )
    G_optim = Adam(
        list(Gx.parameters()) + list(Gy.parameters()), lr=config.LR, betas=(config.BETA1, config.BETA2),
    )
    return D_optim, G_optim


def get_G_losses(Dx, Dy, Gx, Gy, real_x, real_y, real_gt):
    with torch.autocast(device_type=config.DEVICE.type, dtype=torch.float16, enabled=True):
        fake_y = Gx(real_x)
        fake_y_pred = Dy(fake_y)
        Gx_gan_loss = config.GAN_CRIT(fake_y_pred, real_gt)

        fake_x = Gy(real_y)
        fake_x_pred = Dx(fake_x)
        Gy_gan_loss = config.GAN_CRIT(fake_x_pred, real_gt)

        Gx_id_loss = config.ID_CRIT(Gx(real_y), real_y)
        Gy_id_loss = config.ID_CRIT(Gy(real_x), real_x)

        fake_fake_x = Gy(fake_y)
        forward_cycle_loss = config.CYCLE_CRIT(fake_fake_x, real_x)

        fake_fake_y = Gx(fake_x)
        backward_cycle_loss = config.CYCLE_CRIT(fake_fake_y, real_y)
    return (
        fake_x,
        fake_y,
        Gx_gan_loss,
        Gy_gan_loss,
        Gx_id_loss,
        Gy_id_loss,
        forward_cycle_loss,
        backward_cycle_loss,
    )


def get_D_losses(
    Dx, Dy, real_x, real_y, real_gt, fake_gt, fake_x, fake_y, x_img_buffer, y_img_buffer,
):
    with torch.autocast(device_type=config.DEVICE.type, dtype=torch.float16, enabled=True):
        real_y_pred = Dy(real_y)
        real_Dy_loss = config.GAN_CRIT(real_y_pred, real_gt)
        past_fake_y = y_img_buffer(fake_y)
        fake_y_pred = Dy(past_fake_y.detach())
        fake_Dy_loss = config.GAN_CRIT(fake_y_pred, fake_gt)
        # "We divide the objective by 2 while optimizing D, which slows down the rate at which D learns,
        # relative to the rate of G."
        Dy_loss = (real_Dy_loss + fake_Dy_loss) / 2

        real_x_pred = Dx(real_x)
        real_Dx_loss = config.GAN_CRIT(real_x_pred, real_gt)
        past_fake_x = x_img_buffer(fake_x)
        fake_x_pred = Dx(past_fake_x.detach())
        fake_Dx_loss = config.GAN_CRIT(fake_x_pred, fake_gt)
        # "We divide the objective by 2 while optimizing D, which slows down the rate at which D learns,
        # relative to the rate of G."
        Dx_loss = (real_Dx_loss + fake_Dx_loss) / 2
    return Dy_loss, Dx_loss


def _get_lr(epoch):
    # "We keep the same learning rate for the first 100 epochs and linearly decay the rate to zero
    # over the next 100 epochs."
    if epoch < config.N_EPOCHS_BEFORE_DECAY:
        lr = config.LR
    else:
        lr = - config.LR / (config.N_EPOCHS - config.N_EPOCHS_BEFORE_DECAY + 1) * (epoch - config.N_EPOCHS - 1)
    return lr


def update_lrs(
    D_optim,
    G_optim,
    epoch,
):
    lr = _get_lr(epoch)
    D_optim.param_groups[0]["lr"] = lr
    G_optim.param_groups[0]["lr"] = lr


def generate_samples(Gx, Gy, real_x, real_y):
    Gx.eval()
    with torch.no_grad():
        fake_y = Gx(real_x)
    forward_grid = image_to_grid(
        x=real_x,
        y=fake_y,
        x_mean=config.X_MEAN,
        x_std=config.X_STD,
        y_mean=config.Y_MEAN,
        y_std=config.Y_STD,
    )
    Gx.train()

    Gy.eval()
    with torch.no_grad():
        fake_x = Gy(real_y)
    backward_grid = image_to_grid(
        x=real_y,
        y=fake_x,
        x_mean=config.X_MEAN,
        x_std=config.X_STD,
        y_mean=config.Y_MEAN,
        y_std=config.Y_STD,
    )
    Gy.train()
    return forward_grid, backward_grid


def save_G(G, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    state_dict = _modify_state_dict(G.state_dict())
    torch.save(state_dict, str(save_path))


def save_checkpoint(
    epoch, Dx, Dy, Gx, Gy, D_optim, G_optim, scaler, x_img_buffer, y_img_buffer, save_path,
):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        "epoch": epoch,
        "Dx": Dx.state_dict(),
        "Dy": Dy.state_dict(),
        "Gx": Gx.state_dict(),
        "Gy": Gy.state_dict(),
        "D_optimizer": D_optim.state_dict(),
        "G_optimizer": G_optim.state_dict(),
        "scaler": scaler.state_dict(),
        "stored_x_images": x_img_buffer.stored_images,
        "stored_y_images": y_img_buffer.stored_images,
    }
    torch.save(state_dict, str(save_path))
    wandb.save(str(save_path), base_path=Path(save_path).parent)


def train_single_step(
    real_x, real_y, real_gt, fake_gt, Dx, Dy, Gx, Gy, G_optim, D_optim, scaler,
):
    real_x = real_x.to(config.DEVICE)
    real_y = real_y.to(config.DEVICE)

    ### Train Gx and Gy.
    (
        fake_x,
        fake_y,
        Gx_gan_loss,
        Gy_gan_loss,
        Gx_id_loss,
        Gy_id_loss,
        forward_cycle_loss,
        backward_cycle_loss,
    ) = get_G_losses(
        Dx=Dx,
        Dy=Dy,
        Gx=Gx,
        Gy=Gy,
        real_x=real_x,
        real_y=real_y,
        real_gt=real_gt,
    )
    G_loss = Gx_gan_loss + Gy_gan_loss
    G_loss += config.ID_LAMB * (Gx_id_loss + Gy_id_loss)
    G_loss += config.CYCLE_LAMB * (forward_cycle_loss +  backward_cycle_loss)

    set_requires_grad(models=[Dx, Dy], grad=False) # Freeze Ds

    G_optim.zero_grad()
    scaler.scale(G_loss).backward()
    scaler.step(G_optim)

    set_requires_grad(models=[Dx, Dy], grad=True)

    ### Train Dx and Dy.
    Dy_loss, Dx_loss = get_D_losses(
        Dx=Dx,
        Dy=Dy,
        real_x=real_x,
        real_y=real_y,
        real_gt=real_gt,
        fake_gt=fake_gt,
        fake_x=fake_x,
        fake_y=fake_y,
        x_img_buffer=x_img_buffer,
        y_img_buffer=y_img_buffer,
    )

    D_optim.zero_grad()
    scaler.scale(Dy_loss).backward()
    scaler.scale(Dx_loss).backward()
    scaler.step(D_optim)

    scaler.update()
    return (
        Gx_gan_loss,
        Gy_gan_loss,
        Gx_id_loss,
        Gy_id_loss,
        forward_cycle_loss,
        backward_cycle_loss,
        Dy_loss,
        Dx_loss,
    )


if __name__ == "__main__":
    set_seed(config.SEED)

    PARENT_DIR = Path(__file__).resolve().parent
    SAMPLES_DIR = PARENT_DIR/"samples"
    CKPTS_DIR = PARENT_DIR/"checkpoints"

    args = get_args()

    run = wandb.init(project="CycleGAN", resume=args.run_id)
    if args.run_id is None:
        args.run_id = wandb.run.name
    wandb.config.update(
        {
            "seed": config.SEED, "fixed_pairs": config.FIXED_PAIRS, "ds_name": args.ds_name,
        },
        allow_val_change=True,
    )
    print(wandb.config)

    REAL_GT = torch.ones(size=(config.TRAIN_BATCH_SIZE, 1), device=config.DEVICE)
    FAKE_GT = torch.zeros(size=(config.TRAIN_BATCH_SIZE, 1), device=config.DEVICE)

    train_dl, x_test_dl, y_test_dl = get_dls(
        data_dir=args.data_dir,
        train_batch_size=config.TRAIN_BATCH_SIZE,
        test_batch_size=args.test_batch_size,
        n_cpus=args.n_cpus,
        fixed_pairs=config.FIXED_PAIRS,
    )
    TEST_REAL_X = next(iter(x_test_dl)).to(config.DEVICE)
    TEST_REAL_Y = next(iter(y_test_dl)).to(config.DEVICE)

    Dx, Dy, Gx, Gy = get_models(device=config.DEVICE)

    D_optim, G_optim = get_optims(Dx=Dx, Dy=Dy, Gx=Gx, Gy=Gy)

    scaler = GradScaler()

    ### Train.
    x_img_buffer = ImageBuffer(buffer_size=config.BUFFER_SIZE)
    y_img_buffer = ImageBuffer(buffer_size=config.BUFFER_SIZE)

    ### Resume
    DS_NAME_DIR = CKPTS_DIR/args.ds_name
    CKPT_PATH = DS_NAME_DIR/"checkpoint.tar"
    if wandb.run.resumed:
        state_dict = torch.load(str(CKPT_PATH), map_location=config.DEVICE)
        Dx.load_state_dict(state_dict["Dx"])
        Dy.load_state_dict(state_dict["Dy"])
        Gx.load_state_dict(state_dict["Gx"])
        Gy.load_state_dict(state_dict["Gy"])
        D_optim.load_state_dict(state_dict["D_optimizer"])
        G_optim.load_state_dict(state_dict["G_optimizer"])
        scaler.load_state_dict(state_dict["scaler"])
        init_epoch = state_dict["epoch"]
        x_img_buffer.stored_images = state_dict["stored_x_images"]
        y_img_buffer.stored_images = state_dict["stored_y_images"]
        print(f"Resuming from epoch {init_epoch + 1}...")
    else:
        init_epoch = 0

    for epoch in range(init_epoch + 1, config.N_EPOCHS + 1):
        update_lrs(
            D_optim=D_optim,
            G_optim=G_optim,
            epoch=epoch,
        )

        accum_Dy_loss = 0
        accum_Dx_loss = 0
        accum_Gx_gan_loss = 0
        accum_Gy_gan_loss = 0
        accum_Gx_id_loss = 0
        accum_Gy_id_loss = 0
        accum_forward_cycle_loss = 0
        accum_backward_cycle_loss = 0

        start_time = time()
        for step, (real_x, real_y) in enumerate(train_dl, start=1):
            (
                Gx_gan_loss,
                Gy_gan_loss,
                Gx_id_loss,
                Gy_id_loss,
                forward_cycle_loss,
                backward_cycle_loss,
                Dy_loss,
                Dx_loss,
            ) = train_single_step(
                real_x=real_x,
                real_y=real_y,
                real_gt=REAL_GT,
                fake_gt=FAKE_GT,
                Dx=Dx,
                Dy=Dy,
                Gx=Gx,
                Gy=Gy,
                G_optim=G_optim,
                D_optim=D_optim,
                scaler=scaler,
            )
            accum_Gx_gan_loss += Gx_gan_loss.item()
            accum_Gy_gan_loss += Gy_gan_loss.item()
            accum_Gx_id_loss += Gx_id_loss.item()
            accum_Gy_id_loss += Gy_id_loss.item()
            accum_forward_cycle_loss += forward_cycle_loss.item()
            accum_backward_cycle_loss += backward_cycle_loss.item()
            accum_Dy_loss += Dy_loss.item()
            accum_Dx_loss += Dx_loss.item()

        accum_Gx_gan_loss /= len(train_dl)
        accum_Gy_gan_loss /= len(train_dl)
        accum_Gx_id_loss /= len(train_dl)
        accum_Gy_id_loss /= len(train_dl)
        accum_forward_cycle_loss /= len(train_dl)
        accum_backward_cycle_loss /= len(train_dl)
        accum_Dy_loss /= len(train_dl)
        accum_Dx_loss /= len(train_dl)

        msg = f"[ {get_elapsed_time(start_time)} ]"
        msg += f"[ {epoch}/{config.N_EPOCHS} ]"
        msg += f"[ Dy: {accum_Dy_loss:.3f} ]"
        msg += f"[ Dx: {accum_Dx_loss:.3f} ]"
        msg += f"[ Gx GAN: {accum_Gx_gan_loss:.3f} ]"
        msg += f"[ Gy GAN: {accum_Gy_gan_loss:.3f} ]"
        msg += f"[ Gx id: {accum_Gx_id_loss:.3f} ]"
        msg += f"[ Gy id: {accum_Gy_id_loss:.3f} ]"
        msg += f"[ Forward cycle: {accum_forward_cycle_loss:.3f} ]"
        msg += f"[ Backward cycle: {accum_backward_cycle_loss:.3f} ]"
        print(msg)

        wandb.log(
            {
                "Learning rate": D_optim.param_groups[0]["lr"],
                "Dy loss": accum_Dy_loss,
                "Dx loss": accum_Dx_loss,
                "Gx GAN loss": accum_Gx_gan_loss,
                "Gy GAN loss": accum_Gy_gan_loss,
                "Gx identity loss": accum_Gx_id_loss,
                "Gy identity loss": accum_Gy_id_loss,
                "Forward cycle loss": accum_forward_cycle_loss,
                "Backward cycle loss": accum_backward_cycle_loss,
            },
            step=epoch,
        )

        ### Generate samples.
        if epoch % config.GEN_SAMPLES_EVERY == 0:
            forward_grid, backward_grid = generate_samples(
                Gx=Gx, Gy=Gy, real_x=TEST_REAL_X, real_y=TEST_REAL_Y,
            )
            forward_save_path = f"{SAMPLES_DIR}/{args.ds_name}/forward_epoch_{epoch}.jpg"
            backward_save_path = f"{SAMPLES_DIR}/{args.ds_name}/backward_epoch_{epoch}.jpg"
            save_image(forward_grid, path=forward_save_path)
            save_image(backward_grid, path=backward_save_path)
            wandb.log(
                {
                    "Generated images from test set (forward)": wandb.Image(forward_save_path),
                    "Generated images from test set (backward)": wandb.Image(backward_save_path),
                },
                step=epoch,
            )

        ### Save checkpoint.
        if epoch % config.SAVE_GENS_EVERY == 0:
            save_G(G=Gx, save_path=DS_NAME_DIR/f"Gx_epoch_{epoch}.pth")
            save_G(G=Gy, save_path=DS_NAME_DIR/f"Gy_epoch_{epoch}.pth")

        save_checkpoint(
            epoch=epoch,
            Dx=Dx,
            Dy=Dy,
            Gx=Gx,
            Gy=Gy,
            D_optim=D_optim,
            G_optim=G_optim,
            scaler=scaler,
            x_img_buffer=x_img_buffer,
            y_img_buffer=y_img_buffer,
            save_path=CKPT_PATH,
        )
