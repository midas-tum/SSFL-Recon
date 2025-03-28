
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.mri import FFT2c, IFFT2c


class CINE2DDataset(tf.keras.utils.Sequence):
    """
    Data loader for contrastive feature learning on 2D cardiac Cine MRI.
    Generates 3 inputs per sample: 2 positive augmentations + 1 negative subject.
    """

    def __init__(self, min_R, max_R, R_neg_range='diff', mode='train', split=None, shuffle=True):
        self.mode = mode
        self.batch_size = 1
        self.min_R = min_R
        self.max_R = max_R
        self.R3_range = R_neg_range
        self.split = split
        self.shuffle = shuffle

        # To optionally use disjoint datasets for the two-stage training:
        # The CSV files are expected to contain: filename, nFE, nPE, COIL_DIM, TIME_DIM, SLICE_DIM
        if self.split:
            if self.mode == 'train':
                data_set = pd.read_csv('dataset/CINE2D_h5_train_feature.csv')
            elif self.mode == 'val' or self.mode == 'test':
                data_set = pd.read_csv('dataset/CINE2D_h5_val_feature.csv')
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

    def _random_slice(self, minsl, maxsl):
        """According to batchsize, random choose slices."""
        slice_range = np.arange(minsl, maxsl + 1)
        slice_prob = np.ones_like(slice_range, dtype=float)
        slice_prob /= slice_prob.sum()
        return list(np.sort(np.random.choice(slice_range, min(self.batch_size, maxsl + 1 - minsl), p=slice_prob, replace=False, )))

    def __getitem__(self, idx):
        # Select subject A (positive instance)
        idx_pos = idx
        fname_pos, minsl_pos, maxsl_pos, attrs_pos = self.data_set[idx_pos]
        fname_pos = fname_pos.split('.')[0]

        # Select subject B (negative instance)
        idx_neg = random.randint(0, len(self.data_set)-1)
        while idx_neg == idx_pos:
            idx_neg = random.randint(0, len(self.data_set)-1)
        fname_neg, minsl_neg, maxsl_neg, attrs_neg = self.data_set[idx_neg]
        fname_neg = fname_neg.split('.')[0]

        # Select slice
        slidx_pos = self._random_slice(minsl_pos, maxsl_pos)
        slidx_neg = self._random_slice(minsl_neg, maxsl_neg)

        # load normalized fully-sampled images, value range [0, 1]
        # shape: (nSlices, frequency encoding steps, phase encoding steps, coils) = (nSlices, 25, x, y, 15)
        norm_imgc_pos = np.load('norm_img_%s.txt.npy' % fname_pos)
        norm_imgc_neg = np.load('norm_img_%s.txt.npy' % fname_neg)
        batch_imgc_pos = norm_imgc_pos[slidx_pos]  # (1, 25, x, y, 1)
        batch_imgc_neg = norm_imgc_neg[slidx_neg]

        # load coil-compressed time-averaged coil sensitivity maps, (nSlices, 1, x, y, 15)
        cc_smap_pos = np.load('cc_smap_15_%s.txt.npy' % fname_pos)
        cc_smap_neg = np.load('cc_smap_15_%s.txt.npy' % fname_neg)
        batch_smaps_pos = cc_smap_pos[slidx_pos]  # (1, 1, x, y, 15)
        batch_smaps_neg = cc_smap_neg[slidx_neg]

        # Initial 2x GRAPPA undersampling
        p_pos = batch_imgc_pos.shape[3]
        initial_mask_pos = np.loadtxt("mask_GRAPPA_%s.txt" % (fname_pos), dtype=int, delimiter=",")  # (y, 25)
        initial_mask_pos = np.expand_dims(np.transpose(initial_mask_pos), (0, 2, 4))  # (1, 25, 1, y, 1)

        # Retrospective simulation of prospective undersampled acquisition.
        # The resulting undersampled k-space can alternatively be replaced
        # by directly loading prospectively undersampled data.
        imgcoil_pos = batch_smaps_pos * batch_smaps_pos  # (1, t, x, y, c)
        coilkspace_pos = FFT2c(imgcoil_pos)
        masked_kspace_pos = initial_mask_pos * coilkspace_pos

        # Generate re-undersampling masks for creating the positive pair: M1 & M2
        R_1 = random.randint(self.min_R, self.max_R)
        R_2 = random.randint(self.min_R, self.max_R)
        sd_1 = random.randint(1, 20)
        sd_2 = random.randint(1, 20)
        # Make sure M1 and M2 are different
        while R_1 == R_2 and sd_2 == sd_1:
            R_2 = random.randint(self.min_R, self.max_R)
            sd_2 = random.randint(1, 20)

        # Generate undersampling mask M3 for creating the negative pair
        p_neg = batch_imgc_neg.shape[3]
        R_3 = random.randint(self.min_R, self.max_R)
        sd_3 = random.randint(1, 20)
        # Additional restrictions for M3: diff / random
        # diff: enhance the difference between M3 and M1, the acceleration difference is ≥ 5
        # random: only ensure M3 is different from M1 and M2
        if self.R3_range == 'diff':
            while np.abs(R_1 - R_3) < 5:
                R_3 = random.randint(self.min_R, self.max_R)
        elif self.R3_range == 'random':
            while (R_3 == R_1 and sd_3 == sd_1) or (R_3 == R_2 and sd_3 == sd_2):
                R_3 = random.randint(self.min_R, self.max_R)
                sd_3 = random.randint(1, 20)

        mask_1 = np.loadtxt("mask_VISTA_%dx%d_acc%d_%d.txt" % (p_pos, 25, R_1, sd_1), dtype=int, delimiter=",")  # (y, 25)
        mask_2 = np.loadtxt("mask_VISTA_%dx%d_acc%d_%d.txt" % (p_pos, 25, R_2, sd_2), dtype=int, delimiter=",")  # (y, 25)
        mask_3 = np.loadtxt("mask_VISTA_%dx%d_acc%d_%d.txt" % (p_neg, 25, R_3, sd_3), dtype=int, delimiter=",")  # (y, 25)

        mask_1 = np.expand_dims(np.transpose(mask_1), (0, 2, 4))  # (1, 25, 1, y, 1)
        mask_2 = np.expand_dims(np.transpose(mask_2), (0, 2, 4))
        mask_3 = np.expand_dims(np.transpose(mask_3), (0, 2, 4))

        # create positive pairs (kspace)
        masked_kspace_1 = mask_1 * masked_kspace_pos
        masked_kspace_2 = mask_2 * masked_kspace_pos

        # create negative instance (kspace)
        imgcoil_neg = batch_smaps_neg * batch_smaps_neg
        coilkspace_neg = FFT2c(imgcoil_neg)
        masked_kspace_3 = mask_3 * coilkspace_neg

        # transform to image domain
        masked_coilimg_1 = IFFT2c(masked_kspace_1)
        masked_coilimg_2 = IFFT2c(masked_kspace_2)
        masked_coilimg_3 = IFFT2c(masked_kspace_3)
        masked_img_1 = np.expand_dims(np.sum(masked_coilimg_1 * np.conj(batch_smaps_pos), -1), axis=-1)
        masked_img_2 = np.expand_dims(np.sum(masked_coilimg_2 * np.conj(batch_smaps_pos), -1), axis=-1)
        masked_img_3 = np.expand_dims(np.sum(masked_coilimg_3 * np.conj(batch_smaps_neg), -1), axis=-1)

        masked_img_1 = tf.cast(masked_img_1, tf.complex64)
        masked_img_2 = tf.cast(masked_img_2, tf.complex64)
        masked_img_3 = tf.cast(masked_img_3, tf.complex64)

        masked_kspace_1 = tf.cast(masked_kspace_1, tf.complex64)
        masked_kspace_2 = tf.cast(masked_kspace_2, tf.complex64)
        masked_kspace_3 = tf.cast(masked_kspace_3, tf.complex64)

        # effective mask: consider initial & re-undersampling
        mask_1 = initial_mask_pos * mask_1
        mask_2 = initial_mask_pos * mask_2
        mask_1 = mask_1.astype(np.float64)
        mask_2 = mask_2.astype(np.float64)
        mask_3 = mask_3.astype(np.float64)

        batch_smaps_1 = tf.cast(batch_smaps_pos, tf.complex64)
        batch_smaps_2 = tf.cast(batch_smaps_pos, tf.complex64)
        batch_smaps_3 = tf.cast(batch_smaps_neg, tf.complex64)

        # this label does not join loss calculation
        label = tf.zeros(shape=masked_img_1.shape, dtype=tf.complex64)

        return [masked_img_1, masked_img_2, masked_img_3,
                masked_kspace_1, masked_kspace_2, masked_kspace_3,
                mask_1, mask_2, mask_3,
                batch_smaps_1, batch_smaps_2, batch_smaps_3], label
