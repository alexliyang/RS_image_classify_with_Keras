import cv2, keras
import numpy as np
from tiramisu import *
from keras.models import Model, Sequential
from keras.layers import *

INPUT_IMG_PATH      = '53.png'
PREDICT_IMG_PATH    = 'pre_img.png'
CHECK_POINT_PATH    = 'result5.h5'
SLICE_SIZE          = (224, 224)
TYPE_NUM            = 11
MODEL_NAME          = 'Tiramisu'

img = cv2.imread(INPUT_IMG_PATH)
pre_img = np.zeros((img.shape[0], img.shape[1]))
slice_img = np.zeros(SLICE_SIZE)

img_input = Input(shape=img.shape)
x = create_tiramisu(TYPE_NUM, img_input, nb_layers_per_block=[4, 5, 7, 10, 12, 15], p=0.2, wd=1e-4)
model = Model(img_input, x)
model.load_weights(CHECK_POINT_PATH)

for i in range(int(img.shape[0]/224)):
    for j in range(int(img.shape[1]/224)):
        x0 = SLICE_SIZE[0] * i
        x1 = SLICE_SIZE[0] * (i + 1)
        y0 = SLICE_SIZE[1] * i
        y1 = SLICE_SIZE[1] * (i + 1)
        
        slice_img = img[x0:x1, y0:y1]
        temp = model.predict(np.expand_dims(slice_img,0))
        temp = np.argmax(temp, axis=-1)
        temp = temp.reshape((-1,SLICE_SIZE[0], SLICE_SIZE[1]))
        pre_img[x0:x1, y0:y1] = temp[0]

pre_img = pre_img.astype('uint8')

cv2.imwrite(PREDICT_IMG_PATH, pre_img)
 
