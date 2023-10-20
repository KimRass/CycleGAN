### monet2photo
python generate_images.py\
    --ds_name="monet2photo"\
    --data_dir="/Users/jongbeomkim/Documents/datasets/monet2photo"\
    --x_or_y="x"\
    --ckpt_path="/Users/jongbeomkim/Documents/cyclegan/checkpoints/cyclegan_monet_to_photo.pth"\
    --n_cpus=1\
    --batch_size=1

python generate_images.py\
    --ds_name="monet2photo"\
    --data_dir="/Users/jongbeomkim/Documents/datasets/monet2photo"\
    --x_or_y="y"\
    --ckpt_path="/Users/jongbeomkim/Documents/cyclegan/checkpoints/cyclegan_photo_to_monet.pth"\
    --n_cpus=1\
    --batch_size=1

### vangogh2photo
python generate_images.py\
    --ds_name="vangogh2photo"\
    --data_dir="/Users/jongbeomkim/Documents/datasets/vangogh2photo"\
    --x_or_y="x"\
    --ckpt_path="/Users/jongbeomkim/Documents/cyclegan/checkpoints/cyclegan_vangogh_to_photo.pth"\
    --n_cpus=1\
    --batch_size=1

python generate_images.py\
    --ds_name="vangogh2photo"\
    --data_dir="/Users/jongbeomkim/Documents/datasets/vangogh2photo"\
    --x_or_y="y"\
    --ckpt_path="/Users/jongbeomkim/Documents/cyclegan/checkpoints/cyclegan_photo_to_vangogh.pth"\
    --n_cpus=1\
    --batch_size=1

### ukiyoe2photo
python generate_images.py\
    --ds_name="ukiyoe2photo"\
    --data_dir="/Users/jongbeomkim/Documents/datasets/ukiyoe2photo"\
    --x_or_y="x"\
    --ckpt_path="/Users/jongbeomkim/Documents/cyclegan/checkpoints/cyclegan_ukiyoe_to_photo.pth"\
    --n_cpus=1\
    --batch_size=1

python generate_images.py\
    --ds_name="ukiyoe2photo"\
    --data_dir="/Users/jongbeomkim/Documents/datasets/ukiyoe2photo"\
    --x_or_y="y"\
    --ckpt_path="/Users/jongbeomkim/Documents/cyclegan/checkpoints/cyclegan_photo_to_ukiyoe.pth"\
    --n_cpus=1\
    --batch_size=1

### horse2zebra
python generate_images.py\
    --ds_name="horse2zebra"\
    --data_dir="/Users/jongbeomkim/Documents/datasets/horse2zebra"\
    --x_or_y="x"\
    --ckpt_path="/Users/jongbeomkim/Documents/cyclegan/checkpoints/cyclegan_horse_to_zebra.pth"\
    --n_cpus=1\
    --batch_size=1

python generate_images.py\
    --ds_name="horse2zebra"\
    --data_dir="/Users/jongbeomkim/Documents/datasets/horse2zebra"\
    --x_or_y="y"\
    --ckpt_path="/Users/jongbeomkim/Documents/cyclegan/checkpoints/cyclegan_zebra_to_horse.pth"\
    --n_cpus=1\
    --batch_size=1
