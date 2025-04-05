
import numpy as np
from skimage import metrics


def nrmse(gt, pred):
    """
    :param gt: ground truth, shape: (batch, time, x, y, coil), dtype: complex64
    :param pred: prediction, shape: (batch, time, x, y, coil), dtype: complex64
    :return: Normalized RMSE, averaged over time
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


def SSIM(gt, pred):
    """
    :param gt: ground truth, shape: (batch, time, x, y, coil), dtype: complex64
    :param pred: prediction, shape: (batch, time, x, y, coil), dtype: complex64
    :return: SSIM averaged over time
    """
    
    T = gt.shape[1]
    ssim_list = []
    
    for t in range(T):
        # calculate on the absolute image
        gt_t = np.abs(gt[0, t, :, :, 0])
        pred_t = np.abs(pred[0, t, :, :, 0])
        ssim_i = metrics.structural_similarity(gt_t, pred_t, data_range=np.max(gt_t))
        ssim_list.append(ssim_i)
        
    return np.mean(ssim_list)
