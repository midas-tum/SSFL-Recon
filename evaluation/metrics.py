
import numpy as np
import tensorflow as tf


def compute_mse(gt, pred):
    """
    :param gt: ground truth, shape: (batch, time, x, y, coil), dtype: complex64
    :param pred: prediction, shape: (batch, time, x, y, coil), dtype: complex64
    :return: Mean squared error over spatial-temporal dimensions, averaged over batch and coil.
    """
    diff = tf.cast(gt - pred, tf.complex64)
    return np.mean(np.sum(tf.math.real(tf.math.conj(diff) * diff), axis=(2, 3, 4)), axis=(0, 1))


def nrmse(gt, pred):
    """
    :param gt: ground truth, shape: (batch, time, x, y, coil), dtype: complex64
    :param pred: prediction, shape: (batch, time, x, y, coil), dtype: complex64
    :return: Normalized RMSE over spatial dimension, averaged over time.
    """

    T = gt.shape[1]
    nrmse_list = []

    for t in range(T):
        gt_frame = gt[0, t, :, :, 0]
        pred_frame = pred[0, t, :, :, 0]
        m, n = gt_frame.shape
        mse_err = 1/(m * n) * np.sum(np.abs(gt_frame - pred_frame) ** 2)
        rmse = np.sqrt(mse_err)

        data_range = np.max(np.abs(gt_frame)) - np.min(np.abs(gt_frame))
        nrmse = rmse / data_range
        nrmse_list.append(nrmse)

    return np.mean(nrmse_list)


def psnr(gt, pred, data_range=None, reduce=True):
    """
    Compute the peak signal to noise ratio (PSNR)
    :param gt: ground truth, shape: (batch, time, x, y, coil), dtype: complex64
    :param pred: prediction, shape: (batch, time, x, y, coil), dtype: complex64
    :param data_range: if None, estimated from gt. maximum image
    :return: (mean) psnr
    """

    T = gt.shape[1]
    psnr_list = []

    for t in range(T):
        gt_frame = gt[0, t, :, :, 0]
        pred_frame = pred[0, t, :, :, 0]
        m, n = gt_frame.shape
        mse_err = 1/(m*n) * np.sum(np.abs(gt_frame - pred_frame) ** 2)

        if data_range is None:
            data_range = np.max(np.abs(gt_frame))
        psnr_val = 10 * np.log10(data_range ** 2 / mse_err)

        psnr_list.append(psnr_val)

    if reduce:
        return np.mean(psnr_list)
    else:
        return psnr_list
