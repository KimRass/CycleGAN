# 'CycleGAN' (Zhu et al., 2017) implementation from scratch in PyTorch
## Paper Reading

# Research
## Merge Gx Optimizer and Gy Optimizer
- 왜 합치면 좋은가? Gx와 Gy의 loss가 서로 연관되어 있음. identity loss는 Gx와 Gy가 모두 학습이 이루어질수록 낮아짐
## Padding Mode
- 어디에 `padding_mode="zeros"`를 쓰고 어디에 `padding_mode="reflect"`를 사용할 것인가
