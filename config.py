import torch.nn as nn
from utils import get_device

### Data
IMG_SIZE = 256
X_MEAN = (0.5, 0.5, 0.5)
X_STD = (0.5, 0.5, 0.5)
Y_MEAN = (0.5, 0.5, 0.5)
Y_STD = (0.5, 0.5, 0.5)
SCALE = (0.8, 1)

### Objective
# "For $\mathcal{L}_{GAN}$, we replace the negative log likelihood objective by a least-squares loss. This loss
# is more stable during training and generates higher quality results."
GAN_CRIT = nn.MSELoss()
CYCLE_CRIT = nn.L1Loss()
ID_CRIT = nn.L1Loss()

### Training
DEVICE = get_device()
LR = 0.0002 # "We train our networks from scratch, with a learning rate of 0.0002."
BETA1 = 0.5
BETA2 = 0.999
CYCLE_LAMB = 10 # "We set $\lambda = 10$."
# "The weight for the identity mapping loss was $0.5\lambda$ where $\lambda$ was the weight for cycle consistency
# loss."
ID_LAMB = 0.5 * CYCLE_LAMB
# "To reduce model oscillation we update the discriminators using a history of generated images rather than the
# ones produced by the latest generators. We keep an image buffer that stores the 50 previously created images."
BUFFER_SIZE = 50
# "We keep the same learning rate for the first 100 epochs and linearly decay the rate
# to zero over the next 100 epochs."
N_EPOCHS_BEFORE_DECAY = 100
N_EPOCHS = 200
SAVE_CKPT_EVERY = 10
GEN_SAMPLES_EVERY = 4
