
import tensorflow as tf
import time
import os
import datetime

from utils import callbacks
from .recon_loss import get_loss
from .recon_model import SSFL_Recon
from data_loader.recon_data import CINE2DDataset
from .feature_assisted_unet import FeatureComplexUNet2Dt


def get_recon_net():
    return FeatureComplexUNet2Dt(filters=12, num_levels=2, layers_per_level=1, kernel_size_2d=(1, 5, 5),
                                 kernel_size_t=(3, 1, 1), pool_size=(2, 2, 2), activation_last=None)


def main(start_R, train_min_R, train_max_R, val_min_R, val_max_R, num_iter, FE_mode, split=None, fold='SSFL_Recon', pretrained_weights=None):
    # dataset
    ds_train = CINE2DDataset(start_R, train_min_R, train_max_R, mode='train', split=split, shuffle=True)
    ds_val = CINE2DDataset(start_R, val_min_R, val_max_R, mode='val', split=split, shuffle=False)

    model = SSFL_Recon(num_iter=num_iter, mode='train', feature_learning=FE_mode, pretrained_weights=pretrained_weights)
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
    exp_dir = f'fold__{fold}__/R__{train_min_R}__{train_max_R}'
    log_dir = os.path.join(exp_dir, model.name + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Training
    history = model.fit(ds_train, epochs=400, validation_data=ds_val, max_queue_size=4,
                        callbacks=callbacks.get_callbacks(model, log_dir))


if __name__ == '__main__':
    main(start_R=2, train_min_R=2, train_max_R=16, val_min_R=2, val_max_R=16, num_iter=2, FE_mode='vicreg',
         pretrained_weights='experiments/weights020.tf')
