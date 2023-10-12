# 'CycleGAN' (Zhu et al., 2017) implementation from scratch in PyTorch
## Paper Reading
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://github.com/KimRass/CycleGAN/blob/main/unpaired_image_to_image_translation_using_cycle_consistent_adversarial_networks.pdf)
## How to Use
### Generating Images
```bash
# Example
python3 generate_images.py\
    --ds_name="horse2zebra"\
    --data_dir="/Users/jongbeomkim/Documents/datasets/horse2zebra/"\
    --x_or_y="x"\
    --ckpt_path="/Users/jongbeomkim/Documents/cyclegan/checkpoints/cyclegan_horse_to_zebra.pth"\
    --n_workers=1\
    --batch_size=4
```
## Pre-trained Models and Generated Images
|  | Pre-trained model | Generated images on test set |
| - | - | - |
| Monet to Photo | [cyclegan_monet_to_photo.pth](https://drive.google.com/file/d/18FpqtUzrCZA0hHXKhEJk_R0NkkQwiDYa/view?usp=sharing) | https://github.com/KimRass/CycleGAN/tree/main/generated_images/monet_to_photo |
| Photo to Monet | [cyclegan_photo_to_monet.pth](https://drive.google.com/file/d/1MxJYDgIJ4JC5KmaKJ4902QDkq7PzRd19/view?usp=sharing) | https://github.com/KimRass/CycleGAN/tree/main/generated_images/photo_to_monet |
| Vangogh to Photo | [cyclegan_vangogh_to_photo.pth](https://drive.google.com/file/d/1qdMDZ1IJWjVrIusboo5kPQeIVM08hdIo/view?usp=sharing) | https://github.com/KimRass/CycleGAN/tree/main/generated_images/vangogh_to_photo |
| Photo to Vangogh | [cyclegan_photo_to_vangogh.pth](https://drive.google.com/file/d/1CNrhdJSGe_xVDDcNyOedFtDYOmJ6qKZK/view?usp=sharing) | https://github.com/KimRass/CycleGAN/tree/main/generated_images/photo_to_vangogh |
| Ukiyo-e to Photo | [cyclegan_ukiyoe_to_photo.pth](https://drive.google.com/file/d/1YmnubmBV_inem0kDC5PhlVXY2Waw2IMo/view?usp=sharing) | https://github.com/KimRass/CycleGAN/tree/main/generated_images/ukiyoe_to_photo |
| Photo to Ukiyo-e | [cyclegan_photo_to_ukiyoe.pth](https://drive.google.com/file/d/10AzDkTrK-3czRzm37ebd8QVWZkJ8LlwO/view?usp=sharing) | https://github.com/KimRass/CycleGAN/tree/main/generated_images/photo_to_ukiyoe |
| Horse to Zebra | [cyclegan_horse_to_zebra.pth](https://drive.google.com/file/d/1O9hs1d9dcYaAKPcQpXhlLWVYZnN-zG8V/view?usp=sharing) | https://github.com/KimRass/CycleGAN/tree/main/generated_images/horse_to_zebra |
| Zebra to Horse | [cyclegan_zebra_to_horse.pth](https://drive.google.com/file/d/1-gA_F5r3YNV97lqMu_iby_Kx-INphEMT/view?usp=sharing) | https://github.com/KimRass/CycleGAN/tree/main/generated_images/zebra_to_horse |
## Research
### Merging Optimizers
- - discriminators (Dx와 Dy)와 generators (Gx와 Gy)의 objective는 방향성이 서로 충돌하지만 (adversarial training) Dx와 Dy 그리고 Gx와 Gy는 서로 objective의 방향성이 동일하므로, Dx의 Optimizer와 Dy의 Optimizer를 하나로 합치고, Gx의 Optimizer와 Gy의 Optimizer를 하나로 합쳤습니다.
- As-is:
    ```python
    disc_x_optim = Adam(params=disc_x.parameters(), lr=lr)
    disc_y_optim = Adam(params=disc_y.parameters(), lr=lr)
    gen_x_optim = Adam(params=gen_x.parameters(), lr=lr)
    gen_y_optim = Adam(params=gen_y.parameters(), lr=lr)
    ```
- To-be:
    ```python
    disc_optim = Adam(params=list(disc_x.parameters()) + list(disc_y.parameters()), lr=lr)
    gen_optim = Adam(params=list(gen_x.parameters()) + list(gen_y.parameters()), lr=lr)
    ```
### Padding Mode
- 논문에서는 모든 padding에 대해서 `padding_mode="reflect"`를 사용한 것처럼 쓰여 있으나 공식 repository를 보면 `padding_mode="zeros"`와 `padding_mode="reflect"`를 혼용하고 있어 이를 따랐습니다.
### Image Pairing
- 이미지의 집합 X와 Y의 크기가 서로 다르므로 만약 X의 크기가 Y의 크기보다 크다면 1 epoch 동안 X의 이미지가 한 번씩 모델에 입력으로 들어갈 때 Y의 이미지는 한 번 이상씩 모델에 입력으로 들어가게 됩니다. 즉 X의 크기가 데이터의 크기가 됩니다. 이 점을 간과해 데이터의 크기를 Y의 크기와 같게 했고 X와 Y의 각 원소를 정해진대로 1:1 대응이 되도록 코드를 짰었으나 이를 수정했습니다.
- As-is:
    ```python
    x_path = self.x_paths[idx]
    y_path = self.y_paths[idx]
    ```
- To-be:
    ```python
    if self.x_len >= self.y_len:
        x_path = self.x_paths[idx]
        y_path = random.choice(self.y_paths)
    else:
        y_path = self.y_paths[idx]
        x_path = random.choice(self.x_paths)
    ```
### LSGANs
- 논문에 따르면 objective로서 'negative log likelihood' (`GAN_CRIT = nn.BCEWithLogitsLoss()`) 대신에 'least-squares' (`GAN_CRIT = nn.MSELoss()`)를 사용합니다. (Ref: [Least Squares Generative Adversarial Networks](https://github.com/KimRass/CycleGAN/blob/main/least_squares_generative_adversarial_networks.pdf)) 전자를 사용할 경우 금방 mode collapse가 발생하는 것을 관찰할 수 있었습니다.
