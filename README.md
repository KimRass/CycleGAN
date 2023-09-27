# 'CycleGAN' (Zhu et al., 2017) implementation from scratch in PyTorch
## Paper Reading
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
## Pre-trained Models
## Generated Image Samples
## Research
### Merging Optimizers
- - discriminators (Dx와 Dy)와 generators (Gx와 Gy)의 objective는 방향성이 서로 충돌하지만 (adversarial training) Dx와 Dy 그리고 Gx와 Gy는 서로 objective의 방향성이 동일하므로, Dx의 Optimizer와 Dy의 Optimizer를 하나로 합치고, Gx의 Optimizer와 Gy의 Optimizer를 하나로 합쳤습니다.
- As-is
    ```python
    disc_x_optim = Adam(params=disc_x.parameters(), lr=lr)
    disc_y_optim = Adam(params=disc_y.parameters(), lr=lr)
    gen_x_optim = Adam(params=gen_x.parameters(), lr=lr)
    gen_y_optim = Adam(params=gen_y.parameters(), lr=lr)
    ```
- To-be
    ```python
    disc_optim = Adam(params=list(disc_x.parameters()) + list(disc_y.parameters()), lr=lr)
    gen_optim = Adam(params=list(gen_x.parameters()) + list(gen_y.parameters()), lr=lr)
    ```
### Padding Mode
- 논문에서는 모든 padding에 대해서 `padding_mode="reflect"`를 사용한 것처럼 쓰여 있으나 공식 repository를 보면 `padding_mode="zeros"`와 `padding_mode="reflect"`를 혼용하고 있어 이를 따랐습니다.
