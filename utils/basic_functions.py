
import tensorflow as tf


def complex_scale(x, scale):
    return tf.complex(tf.math.real(x) * scale, tf.math.imag(x) * scale)
