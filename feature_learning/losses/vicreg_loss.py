
import tensorflow as tf


def get_loss(loss):
    return eval(loss)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    flattened_x = tf.reshape(x, [-1])
    return tf.boolean_mask(flattened_x, tf.range(0, n * m) % (n + 1) != 0)


@tf.function
def loss_VICReg_unroll(y_true, y_pred):
    sim_coeff = 25
    std_coeff = 25
    cov_coeff = 1

    n = len(y_pred) // 2
    batch_size, num_features = y_pred[0].shape

    total_repr_loss = 0.0
    total_std_loss = 0.0
    total_cov_loss = 0.0

    for i in range(n):
        h1 = y_pred[i*2 + 0, :]
        h2 = y_pred[i*2 + 1, :]

        # --- Invariance ---
        repr_loss = tf.keras.losses.MeanSquaredError()(h1, h2)

        # --- Variance ---
        std_1 = tf.math.sqrt(tf.math.reduce_variance(h1, axis=1) + 0.0001)
        std_2 = tf.math.sqrt(tf.math.reduce_variance(h2, axis=1) + 0.0001)
        std_loss = tf.reduce_mean(tf.keras.activations.relu(1 - std_1)) / 2 + \
                   tf.reduce_mean(tf.keras.activations.relu(1 - std_2)) / 2

        # --- Covariance ---
        h1 = h1 - tf.reduce_mean(h1, axis=0, keepdims=True)
        h2 = h2 - tf.reduce_mean(h2, axis=0, keepdims=True)
        cov_1 = tf.matmul(tf.transpose(h1), h1) / (batch_size - 1)
        cov_2 = tf.matmul(tf.transpose(h2), h2) / (batch_size - 1)
        cov_loss = tf.reduce_sum(tf.math.pow(off_diagonal(cov_1), 2)) / num_features + \
                   tf.reduce_sum(tf.math.pow(off_diagonal(cov_2), 2)) / num_features

        # --- Accumulate ---
        total_repr_loss += repr_loss
        total_std_loss += std_loss
        total_cov_loss += cov_loss

    # Final total loss
    loss = sim_coeff * total_repr_loss + std_coeff * total_std_loss + cov_coeff * total_cov_loss
    return loss
