
#coding: utf-8
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np




hidden_size = 128
keep_prob = 0.8


def unit_lstm():
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return lstm_cell
# 用 MultiRNNCell 实现多层 LSTM
mlstm_cell = rnn.MultiRNNCell([unit_lstm() for i in range(3)], state_is_tuple=True)

print(mlstm_cell.state_size)

inputs = tf.placeholder(np.float32, shape=(32, 100))
h0 = mlstm_cell.zero_state(32, np.float32)
output, h1 = mlstm_cell.call(inputs, h0)
print(h1)
