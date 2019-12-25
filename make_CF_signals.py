# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 :
# vim: set foldmethod=marker commentstring=\ \ #\ %s :
#
# Author:    Taishi Matsumura
# Created:   2019-12-02
#
# Copyright (C) 2019 Taishi Matsumura
#
import os
import numpy as np
import changefinder
from tqdm import tqdm
import argparse



# =============
#  Functions
# =============
def normalize(signals, coef=1):
    sums = signals.max(axis=0)
    sums[sums == 0] = 1
    return coef * (signals / sums)


def change_find(signal, r, order, smooth, window, head_width, tail_width, task, survive=False):
    # シグナルの値がすべて同じだった場合、changefinder をかけずに終了する
    if np.all(signal[0] == signal):
        return np.zeros(len(signal)),  \
               np.zeros(head_width + len(signal) + tail_width),  \
               np.zeros(head_width + len(signal) + tail_width)

    # 生き残った個体がいる可能性がある場合（survive==True）、始めと最後のシグナルの平均値から
    # 生き残ったかを判断し、生き残った場合は changefinder をかけずに終了する
    if survive:
        start = signal[:window]
        start_mean = start.mean()
        start_std = start.std()
        end = signal[-window:]
        end_mean = end.mean()
        end_std = end.std()

        if -2.8 * (start_mean - end_mean) + 1.0 > (start_std - end_std):
            return np.zeros(len(signal)),  \
                   np.zeros(head_width + len(signal) + tail_width),  \
                   np.zeros(head_width + len(signal) + tail_width)

    # ラベル差分シグナルに ChangeFinder 適用すると、立ち下がり検出が
    # うまくいかないので、蛹化・死亡判定時（立ち下がり検出時）は
    # あらかじめシグナルを左右反転しておく。
    if task in ('pupariation', 'death'):
        signal = signal[::-1]

    signal_padded = randpad(signal, head_width, tail_width)
        
    # r の値が小さすぎるとまれに ValueError: math domain error が生じるので
    # 10回までリトライする
    for _ in range(10):
        try:
            cf = changefinder.ChangeFinder(r, order=order, smooth=smooth)
            score_padded = np.array([cf.update(s) for s in signal_padded])
            score = score_padded[head_width + tail_width:]

            # 反転されたシグナルとスコアをさらに反転してもとに戻す
            if event in ('pupariation', 'death'):
                score = score[::-1]
                score_padded = score_padded[::-1]
                signal_padded = signal_padded[::-1]

            return score, score_padded, signal_padded

        except:
            pass  # エラーが生じても繰り返す

    else:
        raise Exception()  # 10回ともエラーが生じた場合は Exception を送出する


def randpad(signal, head_width, tail_width):
    # Zero padding
    padded = np.zeros(head_width + len(signal) + tail_width)
    padded[head_width:-tail_width] = signal

    # Parameter of random sequences
    mean1 = signal[:10].mean()
    mean2 = signal[-10:].mean()
    std1 = signal[:10].std()
    std2 = signal[-10:].std()

    # Padding the head with a positive random sequence
    randoms = np.random.normal(loc=mean1, scale=std1, size=head_width)
    padded[:head_width] = randoms

    # Padding the tail with a positive random sequence
    randoms = np.random.normal(loc=mean2, scale=std2, size=tail_width)
    padded[-tail_width:] = randoms

    return padded



# =================
#  Argument parse
# =================
parser = argparse.ArgumentParser(description='Perform the ChangeFinder.')
parser.add_argument('signal_path', type=str, help='Put a path to a signal of label difference (signals.npy).')
parser.add_argument('event', type=str, help='Put an event name that will be detected by the ChangeFinder.')
parser.add_argument('-r', type=float, default=0.003, help='Put a decimal of a parameter "r" (0.0 < r < 1.0) of the ChangeFinder.')
args = parser.parse_args()

assert os.path.exists(args.signal_path), 'The given path does not exist.'
assert args.event in ('pupariation', 'eclosion', 'death'), 'The argument "event" needs to be either "pupariation", "eclosion", and "death".'
assert 0.0 < args.r < 1.0, 'The argument "r" needs to be between 0.0 to 1.0.'

r = args.r
event = args.event
signal_path = args.signal_path
out_dir = os.path.dirname(signal_path)


# =============
#  Parameters
# =============
smooth = 30
window = 20

tail_width = int(smooth / 2)
head_width = 1000

assert smooth > 3


# ===================
#  Data preparation
# ===================
signals = np.load(signal_path)
n_frames, n_wells = signals.shape
signals = normalize(signals, coef=1)


# ===================
#  ChangeFinder
# ===================
scores = []
for well_idx in tqdm(range(n_wells)):
    signal = signals[:, well_idx]
    score, _, __ = change_find(
            signal, r, 1, smooth, window, head_width, tail_width, event)
    scores.append(score)
scores = np.array(scores).T

assert signals.shape == scores.shape


# ===================
#  Save
# ===================
np.save(os.path.join(out_dir, f'cf_r{r:.3f}_signals.npy'), scores)
