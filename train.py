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
from dataset import UnpairedImageDataset, OneSideImageDataset
from utils import (
    images_to_grid,
    save_image,
    get_elapsed_time,
    set_requires_grad,
    ImageBuffer,
)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ds_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_cpus", type=int, required=True)
    parser.add_argument("--test_batch_size", type=int, required=True)
    # parser.add_argument("--resume_from", type=str, required=False)
    # parser.add_argument("--resume", action="store_true", required=False)
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
    disc_x = Discriminator().to(device)
    disc_y = Discriminator().to(device)
    gen_x = Generator().to(device)
    gen_y = Generator().to(device)

    disc_x = torch.compile(disc_x)
    disc_y = torch.compile(disc_y)
    gen_x = torch.compile(gen_x)
    gen_y = torch.compile(gen_y)
    return disc_x, disc_y, gen_x, gen_y


def get_optims(disc_x, disc_y, gen_x, gen_y):
    # "We use the Adam solver."
    # 논문에는 learning rate에 관한 얘기만 나오지만 공식 저장소를 그대로 따라 `betas=(0.5, 0.999)`를 설정했습니다. 
    disc_optim = Adam(
        list(disc_x.parameters()) + list(disc_y.parameters()), lr=config.LR, betas=(config.BETA1, config.BETA2),
    )
    gen_optim = Adam(
        list(gen_x.parameters()) + list(gen_y.parameters()), lr=config.LR, betas=(config.BETA1, config.BETA2),
    )
    return disc_optim, gen_optim


def get_gen_losses(disc_x, disc_y, gen_x, gen_y, real_x, real_y, real_gt):
    with torch.autocast(device_type=config.DEVICE.type, dtype=torch.float16, enabled=True):
        fake_y = gen_x(real_x)
        fake_y_pred = disc_y(fake_y)
        gen_x_gan_loss = config.GAN_CRIT(fake_y_pred, real_gt)

        fake_x = gen_y(real_y)
        fake_x_pred = disc_x(fake_x)
        gen_y_gan_loss = config.GAN_CRIT(fake_x_pred, real_gt)

        gen_x_id_loss = config.ID_CRIT(gen_x(real_y), real_y)
        gen_y_id_loss = config.ID_CRIT(gen_y(real_x), real_x)

        fake_fake_x = gen_y(fake_y)
        forward_cycle_loss = config.CYCLE_CRIT(fake_fake_x, real_x)

        fake_fake_y = gen_x(fake_x)
        backward_cycle_loss = config.CYCLE_CRIT(fake_fake_y, real_y)
    return (
        fake_x,
        fake_y,
        gen_x_gan_loss,
        gen_y_gan_loss,
        gen_x_id_loss,
        gen_y_id_loss,
        forward_cycle_loss,
        backward_cycle_loss,
    )


def get_disc_losses(
    disc_x, disc_y, real_x, real_y, real_gt, fake_gt, fake_x, fake_y, x_img_buffer, y_img_buffer,
):
    with torch.autocast(device_type=config.DEVICE.type, dtype=torch.float16, enabled=True):
        real_y_pred = disc_y(real_y)
        real_disc_y_loss = config.GAN_CRIT(real_y_pred, real_gt)
        past_fake_y = y_img_buffer(fake_y)
        fake_y_pred = disc_y(past_fake_y.detach())
        fake_disc_y_loss = config.GAN_CRIT(fake_y_pred, fake_gt)
        # "We divide the objective by 2 while optimizing D, which slows down the rate at which D learns,
        # relative to the rate of G."
        disc_y_loss = (real_disc_y_loss + fake_disc_y_loss) / 2

        real_x_pred = disc_x(real_x)
        real_disc_x_loss = config.GAN_CRIT(real_x_pred, real_gt)
        past_fake_x = x_img_buffer(fake_x)
        fake_x_pred = disc_x(past_fake_x.detach())
        fake_disc_x_loss = config.GAN_CRIT(fake_x_pred, fake_gt)
        # "We divide the objective by 2 while optimizing D, which slows down the rate at which D learns,
        # relative to the rate of G."
        disc_x_loss = (real_disc_x_loss + fake_disc_x_loss) / 2
    return disc_y_loss, disc_x_loss


def _get_lr(epoch):
    # "We keep the same learning rate for the first 100 epochs and linearly decay the rate to zero
    # over the next 100 epochs."
    if epoch < config.N_EPOCHS_BEFORE_DECAY:
        lr = config.LR
    else:
        lr = - config.LR / (config.N_EPOCHS - config.N_EPOCHS_BEFORE_DECAY + 1) * (epoch - config.N_EPOCHS - 1)
    return lr


def update_lrs(
    disc_optim,
    gen_optim,
    epoch,
):
    lr = _get_lr(epoch)
    disc_optim.param_groups[0]["lr"] = lr
    gen_optim.param_groups[0]["lr"] = lr


def generate_samples(gen_x, gen_y, real_x, real_y):
    gen_x.eval()
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
    gen_x.train()

    gen_y.eval()
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
    gen_y.train()
    return forward_grid, backward_grid


def save_gen(gen, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(gen.state_dict, str(save_path))


def save_wandb_checkpoint(
    epoch, disc_x, disc_y, gen_x, gen_y, disc_optim, gen_optim, scaler, x_img_buffer, y_img_buffer, save_path,
):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        "epoch": epoch,
        "Dx": disc_x.state_dict(),
        "Dy": disc_y.state_dict(),
        "Gx": gen_x.state_dict(),
        "Gy": gen_y.state_dict(),
        "D_optimizer": disc_optim.state_dict(),
        "G_optimizer": gen_optim.state_dict(),
        "scaler": scaler.state_dict(),
        "stored_x_images": x_img_buffer.stored_images,
        "stored_y_images": y_img_buffer.stored_images,
    }
    torch.save(state_dict, str(save_path))
    wandb.save(str(save_path))


if __name__ == "__main__":
    PARENT_DIR = Path(__file__).resolve().parent
    SAMPLES_DIR = PARENT_DIR/"samples"
    CKPTS_DIR = PARENT_DIR/"checkpoints"
    CKPT_PATH = CKPTS_DIR/"checkpoint.tar"

    args = get_args()

    run = wandb.init(
        # project="CycleGAN", id=args.run_id, resume="must" if args.run_id is not None else "never",
        project="CycleGAN", resume=args.run_id,
    )
    wandb.config.update(
        {
            "SEED": config.SEED,
            "FIXED_PAIRS": config.FIXED_PAIRS,
        },
        allow_val_change=True,
    )
    wandb.config.update(args, allow_val_change=True)
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

    disc_x, disc_y, gen_x, gen_y = get_models(device=config.DEVICE)

    disc_optim, gen_optim = get_optims(disc_x=disc_x, disc_y=disc_y, gen_x=gen_x, gen_y=gen_y)

    scaler = GradScaler()

    ### Train.
    x_img_buffer = ImageBuffer(buffer_size=config.BUFFER_SIZE)
    y_img_buffer = ImageBuffer(buffer_size=config.BUFFER_SIZE)

    ### Resume
    if wandb.run.resumed:
        state_dict = torch.load(str(CKPT_PATH), map_location=config.DEVICE)
        # state_dict = torch.load(wandb.restore(str(CKPT_PATH)), map_location=config.DEVICE)
        disc_x.load_state_dict(state_dict["Dx"])
        disc_y.load_state_dict(state_dict["Dy"])
        gen_x.load_state_dict(state_dict["Gx"])
        gen_y.load_state_dict(state_dict["Gy"])
        disc_optim.load_state_dict(state_dict["D_optimizer"])
        gen_optim.load_state_dict(state_dict["G_optimizer"])
        scaler.load_state_dict(state_dict["scaler"])
        init_epoch = state_dict["epoch"]
        x_img_buffer.stored_images = state_dict["stored_x_images"]
        y_img_buffer.stored_images = state_dict["stored_y_images"]
        # print(f"Resume from checkpoint '{args.resume_from}'.")
        print(f"Resume from epoch {init_epoch + 1}.")
    else:
        init_epoch = 0
    print(init_epoch)

    # for epoch in range(init_epoch + 1, config.N_EPOCHS + 1):
    #     update_lrs(
    #         disc_optim=disc_optim,
    #         gen_optim=gen_optim,
    #         epoch=epoch,
    #     )

    #     accum_disc_y_loss = 0
    #     accum_disc_x_loss = 0
    #     accum_gen_x_gan_loss = 0
    #     accum_gen_y_gan_loss = 0
    #     accum_gen_x_id_loss = 0
    #     accum_gen_y_id_loss = 0
    #     accum_forward_cycle_loss = 0
    #     accum_backward_cycle_loss = 0

    #     start_time = time()
    #     for step, (real_x, real_y) in enumerate(train_dl, start=1):
    #         real_x = real_x.to(config.DEVICE)
    #         real_y = real_y.to(config.DEVICE)

    #         ### Train Gx and Gy.
    #         (
    #             fake_x,
    #             fake_y,
    #             gen_x_gan_loss,
    #             gen_y_gan_loss,
    #             gen_x_id_loss,
    #             gen_y_id_loss,
    #             forward_cycle_loss,
    #             backward_cycle_loss,
    #         ) = get_gen_losses(
    #             disc_x=disc_x,
    #             disc_y=disc_y,
    #             gen_x=gen_x,
    #             gen_y=gen_y,
    #             real_x=real_x,
    #             real_y=real_y,
    #             real_gt=REAL_GT,
    #         )
    #         gen_loss = gen_x_gan_loss + gen_y_gan_loss
    #         gen_loss += config.ID_LAMB * (gen_x_id_loss + gen_y_id_loss)
    #         gen_loss += config.CYCLE_LAMB * (forward_cycle_loss +  backward_cycle_loss)

    #         set_requires_grad(models=[disc_x, disc_y], grad=False) # Freeze Ds

    #         gen_optim.zero_grad()
    #         scaler.scale(gen_loss).backward()
    #         scaler.step(gen_optim)

    #         set_requires_grad(models=[disc_x, disc_y], grad=True)

    #         accum_gen_x_gan_loss += gen_x_gan_loss.item()
    #         accum_gen_y_gan_loss += gen_y_gan_loss.item()
    #         accum_gen_x_id_loss += gen_x_id_loss.item()
    #         accum_gen_y_id_loss += gen_y_id_loss.item()
    #         accum_forward_cycle_loss += forward_cycle_loss.item()
    #         accum_backward_cycle_loss += backward_cycle_loss.item()

    #         ### Train Dx and Dy.
    #         disc_y_loss, disc_x_loss = get_disc_losses(
    #             disc_x=disc_x,
    #             disc_y=disc_y,
    #             real_x=real_x,
    #             real_y=real_y,
    #             real_gt=REAL_GT,
    #             fake_gt=FAKE_GT,
    #             fake_x=fake_x,
    #             fake_y=fake_y,
    #             x_img_buffer=x_img_buffer,
    #             y_img_buffer=y_img_buffer,
    #         )

    #         disc_optim.zero_grad()
    #         scaler.scale(disc_y_loss).backward()
    #         scaler.scale(disc_x_loss).backward()
    #         scaler.step(disc_optim)

    #         accum_disc_y_loss += disc_y_loss.item()
    #         accum_disc_x_loss += disc_x_loss.item()

    #         scaler.update()

    #     msg = f"[ {get_elapsed_time(start_time)} ]"
    #     msg += f"[ {epoch}/{config.N_EPOCHS} ]"
    #     msg += f"[ Dy: {accum_disc_y_loss / len(train_dl):.3f} ]"
    #     msg += f"[ Dx: {accum_disc_x_loss / len(train_dl):.3f} ]"
    #     msg += f"[ Gx GAN: {accum_gen_x_gan_loss / len(train_dl):.3f} ]"
    #     msg += f"[ Gy GAN: {accum_gen_y_gan_loss / len(train_dl):.3f} ]"
    #     msg += f"[ Gx id: {accum_gen_x_id_loss / len(train_dl):.3f} ]"
    #     msg += f"[ Gy id: {accum_gen_y_id_loss / len(train_dl):.3f} ]"
    #     msg += f"[ Forward cycle: {accum_forward_cycle_loss / len(train_dl):.3f} ]"
    #     msg += f"[ Backward cycle: {accum_backward_cycle_loss / len(train_dl):.3f} ]"
    #     print(msg)

    #     wandb.log(
    #         {
    #             "Learning rate": disc_optim.param_groups[0]["lr"],
    #             "Dy loss": accum_disc_y_loss / len(train_dl),
    #             "Dx loss": accum_disc_x_loss / len(train_dl),
    #             "Gx GAN loss": accum_gen_x_gan_loss / len(train_dl),
    #             "Gy GAN loss": accum_gen_y_gan_loss / len(train_dl),
    #             "Gx identity loss": accum_gen_x_id_loss / len(train_dl),
    #             "Gy identity loss": accum_gen_y_id_loss / len(train_dl),
    #             "Forward cycle loss": accum_forward_cycle_loss / len(train_dl),
    #             "Backward cycle loss": accum_backward_cycle_loss / len(train_dl),
    #         },
    #         step=epoch,
    #     )

    #     ### Generate samples.
    #     if epoch % config.GEN_SAMPLES_EVERY == 0:
    #         forward_grid, backward_grid = generate_samples(
    #             gen_x=gen_x, gen_y=gen_y, real_x=TEST_REAL_X, real_y=TEST_REAL_Y,
    #         )
    #         forward_save_path = f"{SAMPLES_DIR}/{args.ds_name}/forward_epoch_{epoch}.jpg"
    #         backward_save_path = f"{SAMPLES_DIR}/{args.ds_name}/backward_epoch_{epoch}.jpg"
    #         save_image(forward_grid, path=forward_save_path)
    #         save_image(backward_grid, path=backward_save_path)
    #         wandb.log(
    #             {
    #                 "Generated images from test set (forward)": wandb.Image(forward_save_path),
    #                 "Generated images from test set (backward)": wandb.Image(backward_save_path),
    #             },
    #             step=epoch,
    #         )


    ### Save checkpoint.
    epoch = 5
    save_wandb_checkpoint(
        epoch=epoch,
        disc_x=disc_x,
        disc_y=disc_y,
        gen_x=gen_x,
        gen_y=gen_y,
        disc_optim=disc_optim,
        gen_optim=gen_optim,
        scaler=scaler,
        x_img_buffer=x_img_buffer,
        y_img_buffer=y_img_buffer,
        save_path=CKPT_PATH,
    )
    # if epoch % config.SAVE_GENS_EVERY == 0:
    #     save_gen(gen=gen_x, save_path=CKPTS_DIR/f"Gx_epoch_{epoch}.pth")
    #     save_gen(gen=gen_y, save_path=CKPTS_DIR/f"Gy_epoch_{epoch}.pth")
