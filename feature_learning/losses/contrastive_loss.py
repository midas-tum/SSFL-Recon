
import tensorflow as tf


def get_loss(loss):
    return eval(loss)


def get_label(dim):
    label_pos = tf.ones(shape=(dim, dim))
    label_neg = tf.zeros(shape=(dim, dim))
    row1 = tf.concat((label_pos, label_neg), axis=1)
    row2 = row1
    row3 = tf.concat((label_neg, label_neg), axis=1)
    return tf.concat((row1, row2, row3), axis=0)


def get_logits(h1, h2, h3, tau=0.1):
    logits_12 = tf.matmul(h1, h2, transpose_b=True) / tau
    logits_13 = tf.matmul(h1, h3, transpose_b=True) / tau
    logits_row_1 = tf.concat((logits_12, logits_13), axis=1)
    logits_21 = tf.matmul(h2, h1, transpose_b=True) / tau
    logits_23 = tf.matmul(h2, h3, transpose_b=True) / tau
    logits_row_2 = tf.concat((logits_21, logits_23), axis=1)
    logits_31 = tf.matmul(h3, h1, transpose_b=True) / tau
    logits_32 = tf.matmul(h3, h2, transpose_b=True) / tau
    logits_row_3 = tf.concat((logits_31, logits_32), axis=1)
    return tf.concat((logits_row_1, logits_row_2, logits_row_3), axis=0)


@tf.function
def contrastive_unroll_loss(y_true, y_pred):
    loss = 0.0
    n = len(y_pred) // 3
    for i in range(n):
        h1 = tf.nn.l2_normalize(y_pred[i*3 + 0, :])
        h2 = tf.nn.l2_normalize(y_pred[i*3 + 1, :])
        h3 = tf.nn.l2_normalize(y_pred[i*3 + 2, :])
        label = get_label(h1.shape[0])
        logits = get_logits(h1, h2, h3)
        loss += tf.nn.softmax_cross_entropy_with_logits(label, logits)
    return loss
