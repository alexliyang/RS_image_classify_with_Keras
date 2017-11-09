from CONFIGS import *
import glob, os, bcolz
from PIL import Image
import numpy as np

def save_array(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

fnames = glob.glob(TRAINING_DATA_PATH + '*.png')
lnames = [TRAINING_LABEL_PATH + os.path.basename(fn) for fn in fnames]
img_sz = (480, 360)
def open_image(fn): return np.array(Image.open(fn))
imgs = np.stack([open_image(fn) for fn in fnames])
labels = np.stack([open_image(fn) for fn in lnames])
imgs = imgs / 255.
imgs -= 0.4
imgs /= 0.3

save_array(RESIZED_SAVE_PATH + 'train_imgs.bc', imgs)
save_array(RESIZED_SAVE_PATH + 'train_labels.bc', labels)

fnames = glob.glob(TESTING_DATA_PATH + '*.png')
lnames = [TESTING_LABEL_PATH + os.path.basename(fn) for fn in fnames]
img_sz = (480, 360)
def open_image(fn): return np.array(Image.open(fn))
imgs = np.stack([open_image(fn) for fn in fnames])
labels = np.stack([open_image(fn) for fn in lnames])
imgs = imgs / 255.
imgs -= 0.4
imgs /= 0.3

save_array(RESIZED_SAVE_PATH + 'test_imgs.bc', imgs)
save_array(RESIZED_SAVE_PATH + 'test_labels.bc', labels)

print("Data has been successfully resized!")
