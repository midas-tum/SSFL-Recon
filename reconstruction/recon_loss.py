
import tensorflow as tf


def get_loss(loss):
    return eval(loss)


@tf.function
def img_ksp_loss(y_true, y_pred):  # image and k-space losses are calculated in recon_model 
    img_loss = y_pred[0]
    ksp_loss = y_pred[1]
    total_loss = img_loss + ksp_loss
    return total_loss
