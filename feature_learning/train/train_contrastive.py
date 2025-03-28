
import os
import time
import datetime
import tensorflow as tf

from utils.callbacks import get_callbacks
from feature_learning.losses.contrastive_loss import get_loss
from feature_learning.models.contrastive_feature_model import UNET2dt
from data_loader.feature_contrastive_data import CINE2DDataset


def train_feature_learning(min_R, max_R, num_iter, R_neg_range='diff', split=None, fold='contrastive_feature'):
    ds_train = CINE2DDataset(min_R, max_R, R_neg_range, mode='train', split=split, shuffle=True)
    ds_val = CINE2DDataset(min_R, max_R, R_neg_range, mode='val', split=split, shuffle=False)

    model = UNET2dt(num_iter=num_iter, mode='train')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004)
    model.compile(optimizer=optimizer, loss=get_loss('contrastive_unroll_loss'), run_eagerly=True)

    # initialize model to print model summary
    inputs, targets = ds_train.__getitem__(0)
    start = time.time()
    outputs = model.predict(inputs)
    end = time.time()
    print(end - start)
    print(model.summary())

    # Logging dir
    exp_dir = f'fold__{fold}__/R__{min_R}__{max_R}'
    log_dir = os.path.join(exp_dir, model.name + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Training
    model.fit(ds_train, epochs=200, validation_data=ds_val, max_queue_size=4,
              callbacks=get_callbacks(model, log_dir), workers=2)


if __name__ == '__main__':
    train_feature_learning(min_R=2, max_R=16, num_iter=3)
