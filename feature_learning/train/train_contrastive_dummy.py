
import os
import time
import datetime
import tensorflow as tf

from SSFL_rebuttal.organized_code.utils.callbacks import get_callbacks
from SSFL_rebuttal.organized_code.feature_learning.losses.contrastive_loss import get_loss
from SSFL_rebuttal.organized_code.feature_learning.models.contrastive_feature_model import UNET2dt
from SSFL_rebuttal.organized_code.data_loader.feature_contrastive_data_dummy import DummyCINE2DDataset


def train_feature_learning(num_iter, fold='contrastive_feature'):
    ds_train = DummyCINE2DDataset(num_samples=10)
    ds_val = DummyCINE2DDataset(num_samples=2)

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
    exp_dir = f'fold__{fold}__/dummy_test'
    log_dir = os.path.join(exp_dir, model.name + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Training
    model.fit(ds_train, epochs=200, validation_data=ds_val, max_queue_size=4,
              callbacks=get_callbacks(model, log_dir), workers=2)


if __name__ == '__main__':
    train_feature_learning(num_iter=2)
