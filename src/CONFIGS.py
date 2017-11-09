# data path and log path
PATH = '../data/'
TRAINING_DATA_PATH = '../data/train/'
TRAINING_LABEL_PATH = '../data/trainannot/'
VALIDATION_DATA_PATH = '../data/valid/'
VALIDATION_LABEL_PATH = '../data/valannot/'
TESTING_DATA_PATH = '../data/test/'
TESTING_LABEL_PATH = '../data/testannot/'
RESIZED_SAVE_PATH = '../data/results/'
TRAINING_SUMMARY_PATH = '../training_summary/'
CHECKPOINTS_PATH = '../checkpoints/'
MAX_CKPT_TO_KEEP = 5       # max checkpoint files to keep

# patch generation
INPUT_SHAPE = (224, 224, 3)
PATCH_SIZE = 250           # must be even, the image size croped from original images
PATCH_GEN_STRIDE = 32      # maybe used by data generation
PATCH_RAN_GEN_RATIO = 2    # the number of random generated patches is max(img.height, img.width) // PATCH_RAN_GEN_RATIO

# model and training
MODEL_NAME = 'Tiramisu'                     # vgg5, vgg7, cp4
PAD_SIZE = 5
BATCH_SIZE = 8
INPUT_SIZE = PATCH_SIZE                         # the image size input to the network
LABEL_SIZE = INPUT_SIZE - 2 * PAD_SIZE
SCALE_FACTOR = 1
NUM_CHENNELS = 3

# data queue
MIN_QUEUE_EXAMPLES = 1024
NUM_PROCESS_THREADS = 3
NUM_TRAINING_STEPS = 1000000
NUM_TESTING_STEPS = 600

# data argumentation
MAX_RANDOM_BRIGHTNESS = 0.2
RANDOM_CONTRAST_RANGE = [0.8, 1.2]
GAUSSIAN_NOISE_STD = 0.01  # [0...1] (float)
JPEG_NOISE_LEVEL = 2    # [0...4] (int)