# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 :
# vim: set foldmethod=marker commentstring=\ \ #\ %s :
#
# Author:    Taishi Matsumura
# Created:   2018-11-07
#
# Copyright (C) 2018 Taishi Matsumura
#
import os
import numpy as np
import my_threshold
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.close('all')


# =============
#  Parameters
# =============

imaging_envs = [
        'TsukubaRIKEN/allevents/',
        'TsukubaRIKEN/171013.15.50.w1118.starvation.96well.3plates.740D-3/',
        'TsukubaRIKEN/151201w-developmental-timing/',
        'TsukubaRIKEN/20160527w1118.devtime.sev.sucroseconc.3plates/',
        ]

# Put the name of a target directory
imaging_env = imaging_envs[0]

data_root = '//133.24.88.18/sdb/Research/Drosophila/data/'

morpho = 'larva'
morpho = 'adult'

target_dir = '20180824-044058_56x56_277-576wells'
target_dir = '20180827-230651_56x56_120-18wells'
target_dir = '20181022-014033_with_noisy_well'
target_dir = '20181022-normal_noisy_empty'

rise_or_fall = 'fall'
rise_or_fall = 'rise'


signals = np.load(
        os.path.join(
            data_root, imaging_env, 'inference',
            morpho, target_dir, 'signals.npy'))


groups = [range(96), range(96, 192), range(192, 288)]
coefs = np.arange(-2, 20, .1)
results = np.zeros((coefs.shape[0], signals.shape[1]))
bins = np.linspace(0, signals.shape[1], 500)
hist_images = []

for group in groups:

    hists = []

    for i, coef in enumerate(coefs):

        thresholds = my_threshold.entire_stats(signals[list(group)], coef=coef)

        if rise_or_fall == 'rise':

            auto_evals = (signals[list(group)] > thresholds).argmax(axis=1)

        elif rise_or_fall == 'fall':

            auto_evals = (signals.shape[1]
                    - (np.fliplr(signals[list(group)]) > thresholds).argmax(axis=1))

            auto_evals[auto_evals == signals.shape[1]] = 0

        ns, _ = np.histogram(auto_evals, bins)
        hists.append(ns)

    hists = np.array(hists)
    hist_images.append(hists)


f = plt.figure()

for i, hist_matrix in enumerate(hist_images):

    ax = f.add_subplot(len(groups), 1, i+1)
    im = ax.imshow(hist_matrix[:, 1:], origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cb = f.colorbar(im, cax=cax)

    ax.set_title('Plate {}'.format(i+1))
    ax.set_xticks(np.linspace(0, hist_matrix.shape[1], 5))
    ax.set_xticklabels(np.linspace(0, signals.shape[1], 5))
    ax.set_yticks([0, len(coefs)])
    ax.set_yticklabels([coefs.min(), round(coefs.max())])

cb.set_label('Count')
ax.set_xlabel('Timesteps')
ax.set_ylabel('Coefficient of a threshold')

f.show()
