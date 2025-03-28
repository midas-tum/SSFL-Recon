
import os
import gc
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def get_callbacks(model, logdir):
    """
    Combine all callbacks into a single list.
    Includes TensorBoard, model saving, optimizer checkpointing, and loss logging.
    """
    return get_system_callbacks(logdir) + \
           get_model_callbacks(model, logdir) + \
           get_plotting_callbacks(logdir)


def get_system_callbacks(logdir):
    """
    Returns callbacks for system maintenance:
    - TensorBoard logging
    - Python garbage collection (manual memory cleanup)
    """
    class GCCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()

    gc_callback = GCCallback()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)

    return [gc_callback, tensorboard_callback]


def get_model_callbacks(model, logdir):
    """
    Returns callbacks for:
    - Saving model weights each epoch
    - Saving optimizer state for potential resuming
    """

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=logdir + '/weights{epoch:03d}.tf',
        verbose=10,
        save_weights_only=True,
        save_freq='epoch')

    def optimizer_checkpoint_callback(epoch, logs=None):
        opt_weights = model.optimizer.get_weights()
        with open(f'{logdir}/optimizer.pkl', 'wb') as f:
            pickle.dump(opt_weights, f)

    opt_cp_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=optimizer_checkpoint_callback)

    return [cp_callback, opt_cp_callback]


class LossCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to:
    - Log training/validation loss values to CSV
    - Generate and save loss curve image
    """
    def __init__(self, logdir):
        super().__init__()
        self.logdir = logdir
        self.csv_path = f'{self.logdir}/loss.csv'
        self.keys = ['loss', 'val_loss',
                     'crop_loss_rmse', 'val_crop_loss_rmse',
                     'crop_loss_abs_mse', 'val_crop_loss_abs_mse',
                     'crop_loss_abs_mae', 'val_crop_loss_abs_mae']

    def on_train_begin(self, logs=None):
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
        else:
            self.df = pd.DataFrame(columns=['epoch'] + self.keys)

    def on_epoch_end(self, epoch, logs):
        # Create the loss dict and update dataframe
        update_dict = {'epoch': epoch}
        for key in self.keys:
            update_dict[key] = logs.get(key)
        self.df = self.df.append(update_dict, ignore_index=True)

        # save csv
        self.df.to_csv(self.csv_path, index=False)

        # Plot train & val loss
        plt.figure()
        x = np.arange(0, len(self.df))
        plt.plot(x, self.df['loss'], label="train_loss")
        plt.plot(x, self.df['val_loss'], label="val_loss")
        plt.title(f"Training/Validation Loss Epoch {epoch}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'{self.logdir}/loss.png')
        plt.close()


def get_plotting_callbacks(logdir):
    loss_callback = LossCallback(logdir)
    return [loss_callback]
