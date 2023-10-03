from utils import get_device

### Data
### Monet2Photo
IMG_SIZE = 256
X_MEAN = (0.5, 0.5, 0.5)
X_STD = (0.5, 0.5, 0.5)
Y_MEAN = (0.5, 0.5, 0.5)
Y_STD = (0.5, 0.5, 0.5)
FIXED_PAIRS = True

### Training
DEVICE = get_device()
CYCLE_LAMB = 10 # "We set $\lambda = 10$."
# "The weight for the identity mapping loss was $0.5\lambda$  where $\lambda$ was the weight for cycle consistency
# loss."
ID_LAMB = 0.5 * CYCLE_LAMB
# "We divide the objective by 2 while optimizing D, which slows down the rate at which D learns, relative to the
# rate of G."
# DISC_X_WEIGHT = 0.2
DISC_X_WEIGHT = 0.8
DISC_Y_WEIGHT = 0.5
# "We keep the same learning rate for the first 100 epochs and linearly decay the rate
# to zero over the next 100 epochs."
N_EPOCHS = 200
SAVE_EVERY = 10

### Optimizer
WARMUP_EPOCHS = 100
