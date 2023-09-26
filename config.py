from utils import get_device

### Data
### Monet2Photo
IMG_SIZE = 256
MONET_MEAN = (0.5, 0.5, 0.5)
MONET_STD = (0.5, 0.5, 0.5)
PHOTO_MEAN = (0.5, 0.5, 0.5)
PHOTO_STD = (0.5, 0.5, 0.5)

### Training
DEVICE = get_device()
LAMB = 10 # "We set $\lambda = 10$."
# "We keep the same learning rate for the first 100 epochs and linearly decay the rate
# to zero over the next 100 epochs."
N_EPOCHS = 200

### Optimizer
WARMUP_EPOCHS = 100
