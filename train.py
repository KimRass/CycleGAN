import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from pathlib import Path
import argparse
import math

import config
from model import Generator, Discriminator
from monet2photo import Monet2PhotoDataset
from utils import images_to_grid, save_image


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ds_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    # "We use the Adam solver with a batch size of 1."
    parser.add_argument("--n_workers", type=int, required=True)
    parser.add_argument("--test_batch_size", type=int, required=True)
    # "We train our networks from scratch, with a learning rate of 0.0002."
    parser.add_argument("--lr", type=float, required=False, default=0.0002)
    parser.add_argument("--train_batch_size", type=int, required=False, default=1)
    parser.add_argument("--resume_from", type=str, required=False)

    args = parser.parse_args()
    return args


def select_ds(ds_name):
    if ds_name == "monet2photo":
        ds = Monet2PhotoDataset
        x_mean = config.MONET_MEAN
        x_std = config.MONET_STD
        y_mean = config.PHOTO_MEAN
        y_std = config.PHOTO_STD
    return ds, x_mean, x_std, y_mean, y_std


def get_dl(ds_name, data_dir, train_batch_size, test_batch_size, n_workers):
    ds, x_mean, x_std, y_mean, y_std = select_ds(ds_name)
    train_ds = ds(
        data_dir=data_dir,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        split="train",
    )
    test_ds = ds(
        data_dir=data_dir,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        split="test",
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_dl, test_dl


def get_models(device):
    disc_x = Discriminator().to(device)
    disc_y = Discriminator().to(device)
    gen_x = Generator().to(device)
    gen_y = Generator().to(device)
    return disc_x, disc_y, gen_x, gen_y


def get_optims(disc_x, disc_y, gen_x, gen_y, lr):
    # "We use the Adam solver."
    disc_x_optim = Adam(params=disc_x.parameters(), lr=lr)
    disc_y_optim = Adam(params=disc_y.parameters(), lr=lr)
    gen_x_optim = Adam(params=gen_x.parameters(), lr=lr)
    gen_y_optim = Adam(params=gen_y.parameters(), lr=lr)
    return disc_x_optim, disc_y_optim, gen_x_optim, gen_y_optim


def get_disc_losses(disc_x, disc_y, gen_x, gen_y, real_gt, fake_gt, gan_crit):
    with torch.autocast(device_type=config.DEVICE.type, dtype=torch.float16, enabled=True):
        real_y_pred = disc_y(real_y)
        real_disc_y_loss = gan_crit(real_y_pred, real_gt)
        fake_y = gen_x(real_x)
        fake_y_pred = disc_y(fake_y)
        fake_disc_y_loss = gan_crit(fake_y_pred, fake_gt)
        # "We divide the objective by 2 while optimizing D, which slows down the rate at
        # which D learns, relative to the rate of G."
        disc_y_loss = (real_disc_y_loss + fake_disc_y_loss) / 2

        real_x_pred = disc_x(real_x)
        real_disc_x_loss = gan_crit(real_x_pred, real_gt)
        fake_x = gen_y(real_y)
        fake_x_pred = disc_x(fake_x)
        fake_disc_x_loss = gan_crit(fake_x_pred, fake_gt)
        disc_x_loss = (real_disc_x_loss + fake_disc_x_loss) / 2
    return disc_x_loss, disc_y_loss


def get_gen_losses(disc_x, disc_y, gen_x, gen_y, real_gt, gan_crit, cycle_crit, identity_crit):
    with torch.autocast(device_type=config.DEVICE.type, dtype=torch.float16, enabled=True):
        fake_y = gen_x(real_x)
        fake_y_pred = disc_y(fake_y)
        gen_x_gan_loss = gan_crit(fake_y_pred, real_gt)

        fake_x = gen_y(real_y)
        fake_x_pred = disc_x(fake_x)
        gen_y_gan_loss = gan_crit(fake_x_pred, real_gt)

        gen_x_identity_loss = identity_crit(gen_x(real_x), real_x)
        gen_y_identity_loss = identity_crit(gen_y(real_y), real_y)

        fake_x = gen_y(fake_y) # G → F
        forward_cycle_loss = cycle_crit(fake_x, real_x)

        fake_y = gen_x(fake_x) # G → F
        backward_cycle_loss = cycle_crit(fake_y, real_y)
    return (
        gen_x_gan_loss,
        gen_y_gan_loss,
        gen_x_identity_loss,
        gen_y_identity_loss,
        forward_cycle_loss,
        backward_cycle_loss,
    )


def _get_lr(epoch, max_lr, warmup_epochs, n_epochs):
    # "We keep the same learning rate for the first 100 epochs and linearly decay the rate to zero
    # over the next 100 epochs."
    if epoch < warmup_epochs:
        lr = max_lr
    else:
        lr = - max_lr / (n_epochs - warmup_epochs + 1) * (epoch - n_epochs - 1)
    return lr


def update_lrs(
    disc_x_optim,
    disc_y_optim,
    gen_x_optim,
    gen_y_optim,
    epoch,
    max_lr,
    warmup_epochs,
    n_epochs,
):
    lr = _get_lr(epoch=epoch, max_lr=max_lr, warmup_epochs=warmup_epochs, n_epochs=n_epochs)
    disc_x_optim.param_groups[0]["lr"] = lr
    disc_y_optim.param_groups[0]["lr"] = lr
    gen_x_optim.param_groups[0]["lr"] = lr
    gen_y_optim.param_groups[0]["lr"] = lr


# def save_checkpoint(epoch, disc_x, disc_y, gen_x, gen_y, disc_optim, gen_optim, loss, save_path):
#     Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#     ckpt = {
#         "epoch": epoch,
#         "Dx": disc_x.state_dict(),
#         "Dx": disc_x.state_dict(),
#         "Gx": gen_x.state_dict(),
#         "D_optimizer": disc_optim.state_dict(),
#         "G_optimizer": gen_optim.state_dict(),
#         "loss": loss,
#     }
#     torch.save(ckpt, str(save_path))


def save_gen(gen, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(gen.state_dict(), str(save_path))


if __name__ == "__main__":
    PARENT_DIR = Path(__file__).parent

    args = get_args()

    train_dl, test_dl = get_dl(
        ds_name=args.ds_name,
        data_dir=args.data_dir,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        n_workers=args.n_workers,
    )

    disc_x, disc_y, gen_x, gen_y = get_models(device=config.DEVICE)

    disc_x_optim, disc_y_optim, gen_x_optim, gen_y_optim = get_optims(
        disc_x=disc_x, disc_y=disc_y, gen_x=gen_x, gen_y=gen_y, lr=args.lr,
    )

    scaler = GradScaler()

    gan_crit = nn.BCEWithLogitsLoss()
    cycle_crit = nn.L1Loss()
    identity_crit = nn.L1Loss()

    ### Train.
    REAL_GT = torch.ones(size=(args.train_batch_size, 1), device=config.DEVICE)
    FAKE_GT = torch.zeros(size=(args.train_batch_size, 1), device=config.DEVICE)

    test_real_x, test_real_y = next(iter(test_dl))
    test_real_x = test_real_x.to(config.DEVICE)
    test_real_y = test_real_y.to(config.DEVICE)

    prev_gen_x_ckpt_path = ".pth"
    prev_gen_y_ckpt_path = ".pth"
    init_epoch = 0
    best_loss = math.inf
    for epoch in range(init_epoch + 1, config.N_EPOCHS + 1):
        update_lrs(
            disc_x_optim=disc_x_optim,
            disc_y_optim=disc_y_optim,
            gen_x_optim=gen_x_optim,
            gen_y_optim=gen_y_optim,
            epoch=epoch,
            max_lr=args.lr,
            warmup_epochs=config.WARMUP_EPOCHS,
            n_epochs=config.N_EPOCHS,
        )

        accum_disc_y_loss = 0
        accum_disc_x_loss = 0
        accum_gen_x_gan_loss = 0
        accum_gen_y_gan_loss = 0
        accum_gen_x_identity_loss = 0
        accum_gen_y_identity_loss = 0
        accum_forward_cycle_loss = 0
        accum_backward_cycle_loss = 0
        for step, (real_x, real_y) in enumerate(train_dl, start=1):
            real_x = real_x.to(config.DEVICE)
            real_y = real_y.to(config.DEVICE)

            ### Train Dx and Dy.
            disc_x_loss, disc_y_loss = get_disc_losses(
                disc_x=disc_x,
                disc_y=disc_y,
                gen_x=gen_x,
                gen_y=gen_y,
                real_gt=REAL_GT,
                fake_gt=FAKE_GT,
                gan_crit=gan_crit,
            )

            disc_loss = disc_x_loss + disc_y_loss
            disc_x_optim.zero_grad()
            disc_y_optim.zero_grad()
            scaler.scale(disc_loss).backward()
            scaler.step(disc_x_optim)
            scaler.step(disc_y_optim)

            accum_disc_x_loss += disc_x_loss.item()
            accum_disc_y_loss += disc_y_loss.item()

            ### Train Gx and Gy.
            (
                gen_x_gan_loss,
                gen_y_gan_loss,
                gen_x_identity_loss,
                gen_y_identity_loss,
                forward_cycle_loss,
                backward_cycle_loss,
            ) = get_gen_losses(
                disc_x=disc_x,
                disc_y=disc_y,
                gen_x=gen_x,
                gen_y=gen_y,
                real_gt=REAL_GT,
                gan_crit=gan_crit,
                cycle_crit=cycle_crit,
                identity_crit=identity_crit,
            )

            gen_loss = gen_x_gan_loss + gen_y_gan_loss
            gen_loss += 0.5 * config.LAMB * (gen_x_identity_loss + gen_y_identity_loss)
            gen_loss += config.LAMB * (forward_cycle_loss +  backward_cycle_loss)
            gen_x_optim.zero_grad()
            gen_y_optim.zero_grad()
            scaler.scale(gen_loss).backward()
            scaler.step(gen_x_optim)
            scaler.step(gen_y_optim)

            scaler.update()

            accum_gen_x_gan_loss += gen_x_gan_loss.item()
            accum_gen_y_gan_loss += gen_y_gan_loss.item()
            accum_gen_x_identity_loss  += gen_x_identity_loss.item()
            accum_gen_y_identity_loss  += gen_y_identity_loss.item()
            accum_forward_cycle_loss += forward_cycle_loss.item()
            accum_backward_cycle_loss += backward_cycle_loss.item()

        print(f"[ {epoch}/{config.N_EPOCHS} ]", end="")
        print(f"[ Dy: {accum_disc_y_loss / len(train_dl):.3f} ]", end="")
        print(f"[ Gx GAN: {accum_gen_x_gan_loss / len(train_dl):.3f} ]", end="")
        print(f"[ Gx identity: {accum_gen_x_identity_loss / len(train_dl):.3f} ]", end="")
        print(f"[ Forward cycle: {accum_forward_cycle_loss / len(train_dl):.3f} ]", end="")
        print(f"[ Dx: {accum_disc_x_loss / len(train_dl):.3f} ]", end="")
        print(f"[ Gy GAN: {accum_gen_y_gan_loss / len(train_dl):.3f} ]", end="")
        print(f"[ Gy identity: {accum_gen_y_identity_loss / len(train_dl):.3f} ]", end="")
        print(f"[ Backward cycle: {accum_backward_cycle_loss / len(train_dl):.3f} ]")

        _, x_mean, x_std, y_mean, y_std = select_ds(args.ds_name)

        ### Generate samples.
        gen_x.eval(), gen_y.eval()
        with torch.no_grad():
            test_fake_y = gen_x(test_real_x)
            test_fake_x = gen_y(test_real_y)
        grid_xy = images_to_grid(
            x=test_real_x, y=test_fake_y, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
        )
        grid_yx = images_to_grid(
            x=test_real_y, y=test_fake_x, x_mean=y_mean, x_std=y_std, y_mean=x_mean, y_std=x_std,
        )
        save_image(grid_xy, path=f"{PARENT_DIR}/samples/{args.ds_name}_forward_epoch_{epoch}.jpg")
        save_image(grid_yx, path=f"{PARENT_DIR}/samples/{args.ds_name}_backward_epoch_{epoch}.jpg")
        gen_x.train(), gen_y.train()

        cur_gen_x_ckpt_path = f"{PARENT_DIR}/pretrained/{args.ds_name}_Gx_epoch_{epoch}.pth"
        save_gen(gen=gen_x, save_path=cur_gen_x_ckpt_path)
        Path(prev_gen_x_ckpt_path).unlink(missing_ok=True)
        prev_gen_x_ckpt_path = cur_gen_x_ckpt_path

        cur_gen_y_ckpt_path = f"{PARENT_DIR}/pretrained/{args.ds_name}_Gy_epoch_{epoch}.pth"
        save_gen(gen=gen_y, save_path=cur_gen_y_ckpt_path)
        Path(prev_gen_y_ckpt_path).unlink(missing_ok=True)
        prev_gen_y_ckpt_path = cur_gen_y_ckpt_path
