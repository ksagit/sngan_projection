import math
import chainer
import chainer.links as L
from chainer import functions as F
from source.links.categorical_conditional_batch_normalization import CategoricalConditionalBatchNormalization
import numpy as np
import copy
import cupy as xp

def _upsample(x):
    h, w = x.shape[2:]
    return F.unpooling_2d(x, 2, outsize=(h * 2, w * 2))


def upsample_conv(x, conv):
    return conv(_upsample(x))


class Block(chainer.Chain):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, n_classes=0):
        super(Block, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        with self.init_scope():
            self.c1 = L.Convolution2D(in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = L.Convolution2D(hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)

            self.b1 = L.BatchNormalization(in_channels)
            self.b2 = L.BatchNormalization(out_channels)
            if self.learnable_sc:
                self.c_sc = L.Convolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def residual(self, x, y=None, z=None, **kwargs):
        h = x

        print("\t\t", F.sum(h.data))
        print("\t\tCL", F.sum(self.c2.b.data))
        h = self.b1(h)

        print("\t\t", F.sum(h.data))
        print("\t\tCL", F.sum(self.c2.b.data))
        h = self.activation(h)

        print("\t\t", F.sum(h.data))
        print("\t\tCL5", F.sum(self.c2.b.data))
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)

        print("\t\t", F.sum(h.data))
        print("\t\tCL5", F.sum(self.c2.b.data))
        h = self.b2(h, y, **kwargs) if y is not None else self.b2(h, **kwargs)

        print("\t\t", F.sum(h.data))
        print("\t\tCL5", F.sum(self.c2.b.data))
        h = self.activation(h)

        print("\t\t", F.sum(h.data))
        print("\t\tCL5", F.sum(self.c2.b.data))
        h = self.c2(h)

        print("\tRSUM", F.sum(h.data))
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            print("\tSCSUM", F.sum(x.data))
            return x
        else:
            print("\tSCSUM", F.sum(x.data))
            return x

    def __call__(self, x, y=None, z=None, **kwargs):
        return self.residual(x, y, z, **kwargs) + self.shortcut(x)
