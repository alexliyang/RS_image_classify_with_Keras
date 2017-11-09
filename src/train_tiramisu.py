import keras, glob, bcolz
from CONFIGS import *
from segm_generator import *
from tiramisu import *
from keras.models import Model, Sequential
from keras.layers import *

def load_array(fname): return bcolz.open(fname)[:]
def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

def main():
    # 我这操作真tm令人窒息
    imgs = load_array(RESIZED_SAVE_PATH + 'train_imgs.bc')
    labels = load_array(RESIZED_SAVE_PATH + 'train_labels.bc')
    test = load_array(RESIZED_SAVE_PATH + 'test_imgs.bc')
    test_labels = load_array(RESIZED_SAVE_PATH + 'test_labels.bc')
    # fnames = glob.glob(TRAINING_DATA_PATH + '*.png')
    #
    # labels_int = labels
    # fn_test = set(o.strip() for o in open(PATH + 'test.txt', 'r'))
    # is_test = np.array([o.split('/')[-1] in fn_test for o in fnames])
    # trn = imgs[is_test == False]
    # trn_labels = labels_int[is_test == False]
    # test = imgs[is_test]
    # test_labels = labels_int[is_test]
    rnd_trn = len(labels)
    rnd_test = len(test_labels)

    limit_mem()
    img_input = Input(shape=INPUT_SHAPE)
    x = create_tiramisu(12, img_input, nb_layers_per_block=[4, 5, 7, 10, 12, 15], p=0.2, wd=1e-4)
    model = Model(img_input, x)
    gen = segm_generator(imgs, labels, 3, train=True)
    gen_test = segm_generator(test, test_labels, 3, train=False)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(2e-4, decay=1-0.9995), metrics=["accuracy"])
    model.fit_generator(gen, rnd_trn, 100, verbose=2, validation_data=gen_test, validation_steps=2)

    model.save_weights(PATH + 'results/8758.h5')


if __name__ == '__main__':
    main()
