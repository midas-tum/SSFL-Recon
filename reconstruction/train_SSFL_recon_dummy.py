
import os
import time
import datetime
import tensorflow as tf

from utils import callbacks
from .recon_loss import get_loss
from .feature_assisted_unet import FeatureComplexUNet2Dt
from .recon_model_contrastive_feature import SSFL_Recon_c
from data_loader.recon_data_dummy import DummyCINE2DDataset


def get_recon_net():
    return FeatureComplexUNet2Dt(filters=12, num_levels=2, layers_per_level=1, kernel_size_2d=(1, 5, 5),
                                 kernel_size_t=(3, 1, 1), pool_size=(2, 2, 2), activation_last=None)


def train_SSFL_recon(num_iter, fold='SSFL_Recon'):
    # dataset
    ds_train = DummyCINE2DDataset()
    ds_val = DummyCINE2DDataset()

    model = SSFL_Recon_c(num_iter=num_iter, mode='train', pretrained_weights='./dummy_weights/weights001.tf')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004)
    model.compile(optimizer, loss=get_loss('img_ksp_loss'), run_eagerly=True)

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
    history = model.fit(ds_train, epochs=400, validation_data=ds_val, max_queue_size=4,
                        callbacks=callbacks.get_callbacks(model, log_dir))


if __name__ == '__main__':
    train_SSFL_recon(num_iter=2)
