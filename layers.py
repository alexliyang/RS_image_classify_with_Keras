from keras.layers import *
from keras.regularizers import l2


def relu(x): return Activation('relu')(x)
def dropout(x, p): return Dropout(p)(x) if p else x
def bn(x): return BatchNormalization(axis=-1)(x)    # mode=2
def relu_bn(x): return relu(bn(x))
def concat(xs): return merge(xs, mode='concat', concat_axis=-1)
def conv(x, nf, sz, wd, p, stride=1):
    x = Conv2D(nf, (sz, sz), kernel_initializer='he_uniform', padding='same',
               strides=(stride, stride), kernel_regularizer=l2(wd))(x)
    return dropout(x, p)

def conv_relu_bn(x, nf, sz=3, wd=0, p=0, stride=1):
    return conv(relu_bn(x), nf, sz, wd=wd, p=p, stride=stride)

def dense_block(n, x, growth_rate, p, wd):
    added = []
    for i in range(n):
        b = conv_relu_bn(x, growth_rate, p=p, wd=wd)
        x = concat([x, b])
        added.append(b)
    return x, added

def transition_dn(x, p, wd):
    # x = conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd)
    # return MaxPooling2D(strides=(2, 2))(x)
    return conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd, stride=2)

def down_path(x, nb_layers, growth_rate, p, wd):
    skips = []
    for i, n in enumerate(nb_layers):
        x, added = dense_block(n, x, growth_rate, p, wd)
        skips.append(x)
        x = transition_dn(x, p=p, wd=wd)
    return skips, added

def transition_up(added, wd=0):
    x = concat(added)
    _, r, c, ch = x.get_shape().as_list()
    return Conv2DTranspose(ch, (3, 3), kernel_initializer='he_uniform',
                           padding='same', strides=(2, 2), kernel_regularizer=l2(wd))(x)
#     x = UpSampling2D()(x)
#     return conv(x, ch, 2, wd, 0)

def up_path(added, skips, nb_layers, growth_rate, p, wd):
    for i, n in enumerate(nb_layers):
        x = transition_up(added, wd)
        x = concat([x, skips[i]])
        x, added = dense_block(n, x, growth_rate, p, wd)
    return x

def reverse(a): return list(reversed(a))
