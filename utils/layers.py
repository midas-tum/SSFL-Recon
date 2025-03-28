
import merlintf
import tensorflow as tf
from .mri import MulticoilForwardOp, MulticoilAdjointOp
from .data_consistency import DCGD


class Scalar(tf.keras.layers.Layer):
    def __init__(self, init=1.0, train_scale=1.0, name=None):
        super().__init__(name=name)
        self.init = init
        self.train_scale = train_scale

    def build(self, input_shape):
        self._weight = self.add_weight(name='scalar',
                                       shape=(1,),
                                       constraint=tf.keras.constraints.NonNeg(),
                                       initializer=tf.keras.initializers.Constant(self.init))

    @property
    def weight(self):
        return self._weight * self.train_scale

    def call(self, inputs):
        return merlintf.complex_scale(inputs, self.weight)


def get_dc_layer():
    A = MulticoilForwardOp(center=True)
    AH = MulticoilAdjointOp(center=True)
    return DCGD(A, AH)

