import bcolz
import keras
from keras.layers import *
from keras.models import Model

from CONFIGS import *
from segm_generator import *
from tiramisu import *


def load_array(fname): return bcolz.open(fname)[:]


def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))



# 我这操作真tm令人窒息
imgs = load_array(RESIZED_SAVE_PATH + 'train_imgs.bc')
labels = load_array(RESIZED_SAVE_PATH + 'train_labels.bc')
test = load_array(RESIZED_SAVE_PATH + 'test_imgs.bc')
test_labels = load_array(RESIZED_SAVE_PATH + 'test_labels.bc')

rnd_trn = len(labels)
# rnd_test = len(test_labels)

# limit_mem()
img_input = Input(shape=INPUT_SHAPE)
model = create_unet(img_input)
gen = segm_generator(imgs, labels, 3, train=True, isunet=True)
gen_test = segm_generator(test, test_labels, 3, train=False, isunet=True)

model.fit_generator(gen, rnd_trn, 2, verbose=2, validation_data=gen_test, validation_steps=2)

model.save_weights(CHECKPOINTS_PATH + 'result.h5')

