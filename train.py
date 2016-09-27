#! /usr/bin/env python
# -*- coding: utf-8 -*- 
from __future__ import print_function
import argparse
aimport math
import sys
import time
import re
import pickle
import random
 

import numpy as np
from numpy.random import *
import six
from scipy.sparse import csr_matrix
from progressbar import ProgressBar, Percentage, Bar
import progressbar
import time

import chainer
from chainer import optimizers
from chainer import serializers
import chainer.functions as F
import chainer.links as L
from chainer import cuda

CORPUS_PATH = '../corpus/kadowaki_head10.wsne'  # 学習するcorpusの場所
MODEL_SAVE_PATH = '../auto/model.dump'  # 学習したモデルの保存先
DICT_SAVE_PATH = '../auto/dict_list.dump'  # dictの保存先  # 第一引数がall_dict, 第二引数がne_dict

def make_index(file_path):
    '''
    Indexを生成する.  生成文用の全単語を含んだベクトルと、固有表現のみを含んだベクトル用のIndex二つを返す.
    '''
    all_index = {'UNK':0}  # 全単語を含んだIndex
    ne_index = {'UNK':0}  # 固有表現のみを含んだIndex
    all_number = 1  # all_indexの素性番号
    ne_number = 1  # ne_indexの素性番号
    f = open(file_path, 'r')
    for sentence in f:
        units = sentence.split()  # unit は、「醤油/F」を指す
        for unit in units:
            if not unit in all_index:  # 今回のunit が初出の時
                all_index[unit] = all_number
                all_number += 1
                matchO = re.search('/O', unit)  # 今回のunitが固有表現かどうか
                if not matchO:
                    if not unit in ne_index:  # 固有表現 が初出の時
                        ne_index[unit] = ne_number
                        ne_number += 1
    return  all_index, ne_index


def make_train(file_path):
    '''
    訓練データを作成する。 trainはne_indexのベクトル列, 出力はall_indexのベクトル列
    一つのベクトルはone-hotであり、入力はベクトルの集合である。
    'りんご/F', '切る/Ac' のone_hotベクターを入力、'りんご/F', 'を/O', '切る/Ac'がone_hot出力ベクター
    '''
    input_ne_data = []
    output_word_data = []
    f = open(file_path, 'r')
    for sentence in f:
        input_ne_line = []  # neの個数のベクトル
        output_word_line = []  # wordの個数のベクトル
        units = sentence.split()  # unit は、「醤油/F」を指す
        for unit in units:
            if unit in all_index:  # all_indexのベクトルを作る
                output = all_index[unit]
                output_word_line.append(output)
            if unit in ne_index:  # ne_indexのベクトルを作る
                input = ne_index[unit]
                input_ne_line.append(input)
        input_ne_data.append(input_ne_line) 
        output_word_data.append(output_word_line)
    return input_ne_data, output_word_data


# LSTMのネットワーク定義
class LSTM(chainer.Chain):
    def __init__(self, p, q, n_units, train=True):
        super(LSTM, self).__init__(
            embed=L.EmbedID(p, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.Linear(n_units, q),
        )

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(h0)
        y = self.l2(h1)
        return y

    def reset_state(self):
        self.l1.reset_state()

# 引数の処理
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

# パラメータ設定
n_units = 512    # 隠れ層のユニット数
# 訓練データの準備

all_index, ne_index = make_index(CORPUS_PATH)  # indexを作成
all_inv = {v:k for k, v in all_index.items()}  # all_indexを逆転
ne_inv = {v:k for k, v in ne_index.items()}  # ne_indexを逆転
input_data, output_data = make_train(CORPUS_PATH)  # 学習データを生成

# モデルの準備
lstm = LSTM(len(ne_index), len(all_index), n_units)
model = L.Classifier(lstm)
model.compute_accuracy = False

for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.2, 0.2, data.shape)
xp = cuda.cupy if args.gpu >= 0 else np
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# optimizerの設定
optimizer = optimizers.Adam()
optimizer.setup(model)

roop = 10  # ループの回数
shuffle = 0  # NEをshuffleするかどうか
# 設定を表示する
print('------------------------------')
print('shuffle:', shuffle)
print('all_indexのサイズ:', len(all_index))
print('ne_indexのサイズ:', len(ne_index))
print('学習の文の数:', len(input_data))
print('ユニット数', n_units)
print('ループの回数', roop)
print('------------------------------')


# 訓練を行うループ
print('学習スタート')
display = 1  # 何回ごとに表示するか
total_loss = 0  # 誤差関数の値を入れる変数
for seq in range(0, roop):
    print(seq+1, '回目')
    p = ProgressBar(widgets=[Percentage(), Bar(), progressbar.Timer()], maxval=len(input_data)).start()
    k = 0  # counter
    for input_line, output_line in zip(input_data, output_data):
        lstm.reset_state()  # 前の系列の影響がなくなるようにリセット  # 1系列は、input_line, output_line
        output_size = len(output_line)  # 出力の単語数, 必ず入力より長くなる
        input_size = len(input_line)
        zero_vector = np.zeros(output_size - input_size)  # 長さを揃える
        shuffled_input_line = np.append(shuffled_input_line, zero_vector)  # 差の回数だけzeroベクトルを加える
            
        for input, output in zip(shuffled_input_line, output_line):
            x = chainer.Variable(xp.array([input], dtype=np.int32))
            t = chainer.Variable(xp.array([output], dtype=np.int32))
            loss = model(x, t)
            # 出力する時はlossを記憶
            if seq%display==0:
                total_loss += loss.data
            # 最適化の実行
            model.zerograds()
            loss.backward()
            optimizer.update()
        k += 1
        p.update(k)

    # lossの表示
    if seq%display==0:
        print("sequence:{}, loss:{}".format(seq, total_loss))
        total_loss = 0

    # sample表示
    print('-------------------')
    print('サンプル表示')
    input_list = []
    output_list = []
    answer_list = []
    for input, output in zip(shuffled_input_line, output_line):
        x = chainer.Variable(xp.array([input], dtype=np.int32))
        input_list.append(ne_inv[input])  # inputのneを追加する
        output_list.append(all_inv[int(np.argmax(model.predict(x).data))])  # 出力の最大の単語を追加する
        answer_list.append(all_inv[output])  # 正解データを出力する
    print('入力:', ', '.join(input_list))
    print('出力:', ' '.join(output_list))
    print('正解:', ' '.join(answer_list))
    print('--------------------')



f = open(MODEL_SAVE_PATH, 'wb')  # モデルをsave
pickle.dump(model, f)
f.close()

f = open(DICT_SAVE_PATH, 'wb')
pickle.dump([all_index, ne_index], f)
f.close()

