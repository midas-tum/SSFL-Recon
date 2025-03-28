
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.mri import FFT2c, IFFT2c


class CINE2DDataset(tf.keras.utils.Sequence):
    """
    Data loader for self-supervised reconstruction.
    Generate 2 re-undersampled images from an accelerated k-space,
    """

    def __init__(self, start_R, min_R, max_R, mode='train', split=None, shuffle=True):
        self.mode = mode
        self.batch_size = 1
        self.min_R = min_R
        self.max_R = max_R
        self.start_R = start_R
        self.split = split
        self.shuffle = shuffle

        # To optionally use disjoint datasets for the two-stage training:
        # The CSV files are expected to contain: filename, nFE, nPE, COIL_DIM, TIME_DIM, SLICE_DIM
        if self.split:
            if self.mode == 'train':
                data_set = pd.read_csv('dataset/CINE2D_h5_train_recon.csv')
            elif self.mode == 'val' or self.mode == 'test':
                data_set = pd.read_csv('dataset/CINE2D_h5_val_recon.csv')
        else:
            if self.mode == 'train':
                data_set = pd.read_csv('dataset/CINE2D_h5_train.csv')
            elif self.mode == 'val' or self.mode == 'test':
                data_set = pd.read_csv('dataset/CINE2D_h5_val.csv')

        self.data_set = []
        for i in range(len(data_set)):
            subj = data_set.iloc[i]
            fname = subj.filename
            nPE = subj.nPE
            num_slices = subj.SLICE_DIM

            # specify the slices
            minsl = 0
            maxsl = num_slices - 1
            assert minsl <= maxsl

            attrs = {'nPE': nPE, 'metadata': subj.to_dict()}
            self.data_set += [(fname, minsl, maxsl, attrs)]

    def __len__(self):
        return len(self.data_set)

    def on_epoch_end(self):
        """Updates indeces after each epoch"""
        self.indeces = np.arange(len(self.data_set))
        if self.shuffle == True:
            np.random.shuffle(self.indeces)

    def __getitem__(self, idx):
        fname, minsl, maxsl, attrs = self.data_set[idx]
        fname = fname.split('.')[0]

        # according to batchsize, random choose slices
        print('selecting slices as one batch...')
        slice_range = np.arange(minsl, maxsl + 1)
        slice_prob = np.ones_like(slice_range, dtype=float)
        slice_prob /= slice_prob.sum()
        slidx = list(np.sort(np.random.choice(
            slice_range,
            min(self.batch_size, maxsl + 1 - minsl),
            p=slice_prob,
            replace=False,
        )))

        # load normalized fully-sampled images, value range [0, 1]
        # shape: (nSlices, frequency encoding steps, phase encoding steps, coils) = (nSlices, 25, x, y, 15)
        norm_imgc = np.load('norm_img_%s.txt.npy' % fname)
        batch_imgc = norm_imgc[slidx]  # (1, 25, x, y, 1)

        # load coil-compressed time-averaged coil sensitivity maps, (nSlices, 1, x, y, 15)
        cc_smap = np.load('cc_smap_15_%s.txt.npy' % fname)
        batch_smaps = cc_smap[slidx]  # (1, 1, x, y, 15)

        # Initial undersampling mask: VISTA, R=2
        p = batch_imgc.shape[3]
        R = self.start_R
        sd = 10
        mask = np.loadtxt("mask_VISTA_%dx%d_acc%d_%d.txt" % (p, 25, R, sd), dtype=int, delimiter=",")  # (y, 25)

        # Generate re-undersampling mask 1
        R_1 = random.randint(self.min_R, self.max_R)
        sd_1 = random.randint(1, 20)
        mask_1 = np.loadtxt("mask_VISTA_%dx%d_acc%d_%d.txt" % (p, 25, R_1, sd_1), dtype=int, delimiter=",")
        mask_1 = mask_1 * mask  # effective mask 1
        mask_1 = np.expand_dims(np.transpose(mask_1), (0, 2, 4))  # (1, 25, 1, y, 1)

        # Generate re-undersampling mask 2
        R_2 = random.randint(self.min_R, self.max_R)
        sd_2 = random.randint(1, 20)
        # Ensure mask_2 differ from mask_1
        while R_1 == R_2 and sd_1 == sd_2:
            R_2 = random.randint(self.min_R, self.max_R)
            sd_2 = random.randint(1, 20)
        mask_2 = np.loadtxt("mask_VISTA_%dx%d_acc%d_%d.txt" % (p, 25, R_2, sd_2), dtype=int, delimiter=",")
        mask_2 = mask_2 * mask
        mask_2 = np.expand_dims(np.transpose(mask_2), (0, 2, 4))

        # Create batch k-space
        imgccoil = batch_imgc * batch_smaps
        coilkspace = FFT2c(imgccoil)

        # Apply mask
        masked_kspace_1 = mask_1 * coilkspace
        masked_kspace_2 = mask_2 * coilkspace
        masked_coilimg_1 = IFFT2c(masked_kspace_1)
        masked_coilimg_2 = IFFT2c(masked_kspace_2)
        masked_img_1 = np.expand_dims(np.sum(masked_coilimg_1 * np.conj(batch_smaps), -1), axis=-1)
        masked_img_2 = np.expand_dims(np.sum(masked_coilimg_2 * np.conj(batch_smaps), -1), axis=-1)

        masked_img_1 = tf.cast(masked_img_1, tf.complex64)
        masked_img_2 = tf.cast(masked_img_2, tf.complex64)
        masked_kspace_1 = tf.cast(masked_kspace_1, tf.complex64)
        masked_kspace_2 = tf.cast(masked_kspace_2, tf.complex64)
        mask_1 = mask_1.astype(np.float64)
        mask_2 = mask_2.astype(np.float64)
        input_smaps = tf.cast(batch_smaps, tf.complex64)

        # this label does not join loss calculation
        label = tf.zeros(shape=masked_img_1.shape, dtype=tf.complex64)

        return [masked_img_1, masked_kspace_1, mask_1, masked_img_2, masked_kspace_2, mask_2, input_smaps], label
