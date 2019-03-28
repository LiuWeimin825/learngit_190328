# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 15:55:05 2019
learngit
add a line
add two line
123
4565
234
123
1235
55
66
77
88
99

@author: nuaal
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import scipy.io as sio
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn
from numpy import float32
import matplotlib.pyplot as plt



from os import path

from tensorflow.contrib.timeseries.python.timeseries import estimators as ts_estimators
from tensorflow.contrib.timeseries.python.timeseries import model as ts_model
from tensorflow.contrib.timeseries.python.timeseries import  NumpyReader

import math

#这部分是TFTS的示例源码，具体定义和用法可以去百度看看，这是github上的链接：
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/timeseries/examples/lstm.py
class _LSTMModel(ts_model.SequentialTimeSeriesModel):
  def __init__(self, num_units, num_features, dtype=tf.float32):
    super(_LSTMModel, self).__init__(
        train_output_names=["mean"],
        predict_output_names=["mean"],
        num_features=num_features,
        dtype=dtype)
    self._num_units = num_units
    self._lstm_cell = None
    self._lstm_cell_run = None
    self._predict_from_lstm_output = None

  def initialize_graph(self, input_statistics):
    super(_LSTMModel, self).initialize_graph(input_statistics=input_statistics)
    self._lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self._num_units)
    self._lstm_cell_run = tf.make_template(
        name_="lstm_cell",
        func_=self._lstm_cell,
        create_scope_now_=True)
    self._predict_from_lstm_output = tf.make_template(
        name_="predict_from_lstm_output",
        func_=lambda inputs: tf.layers.dense(inputs=inputs, units=self.num_features),
        create_scope_now_=True)

  def get_start_state(self):
    return (
        tf.zeros([], dtype=tf.int64),
        tf.zeros([self.num_features], dtype=self.dtype),
        [tf.squeeze(state_element, axis=0)
         for state_element
         in self._lstm_cell.zero_state(batch_size=1, dtype=self.dtype)])

  def _transform(self, data):
    mean, variance = self._input_statistics.overall_feature_moments
    return (data - mean) / variance

  def _de_transform(self, data):
    mean, variance = self._input_statistics.overall_feature_moments
    return data * variance + mean

  def _filtering_step(self, current_times, current_values, state, predictions):
    state_from_time, prediction, lstm_state = state
    with tf.control_dependencies(
            [tf.assert_equal(current_times, state_from_time)]):
      transformed_values = self._transform(current_values)
      predictions["loss"] = tf.reduce_mean(
          (prediction - transformed_values) ** 2, axis=-1)
      new_state_tuple = (current_times, transformed_values, lstm_state)
    return (new_state_tuple, predictions)

  def _prediction_step(self, current_times, state):
    _, previous_observation_or_prediction, lstm_state = state
    lstm_output, new_lstm_state = self._lstm_cell_run(
        inputs=previous_observation_or_prediction, state=lstm_state)
    next_prediction = self._predict_from_lstm_output(lstm_output)
    new_state_tuple = (current_times, next_prediction, new_lstm_state)
    return new_state_tuple, {"mean": self._de_transform(next_prediction)}

  def _imputation_step(self, current_times, state):
    return state

  def _exogenous_input_step(
          self, current_times, current_exogenous_regressors, state):
    """Update model state based on exogenous regressors."""
    raise NotImplementedError(
        "Exogenous inputs are not implemented for this example.")



load_fn = 'tuihuadata0112.mat'
load_data = sio.loadmat(load_fn)
load_ori = load_data['data_set']

data_x = load_ori[:90480,2:]
data_y = load_ori[:90480,0]
data_y = np.reshape(data_y,[-1,1])

data_x_test = load_ori[90480:,2:]
data_y_test = load_ori[90480:,0]
data_y_test = np.reshape(data_y_test,[-1,1])



INPUT_NODE = 13
OUTPUT_NODE = 1
batch_size = 9999

def RUL_predict():
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name = 'x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name = 'y-input')
    keep_prob = tf.placeholder(tf.float32)
    x_input = tf.reshape(x,[-1,1,13,1])
    
    W_conv1 = tf.Variable(tf.truncated_normal([1,3,1,10],stddev = 0.1))
    b_conv1 = tf.Variable(tf.constant(0.1,shape=[10]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_input,W_conv1,strides=[1,1,1,1],padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,1,2,1],strides=[1,1,2,1],padding='SAME')
    
    W_conv2 = tf.Variable(tf.truncated_normal([1,3,10,20],stddev = 0.1))
    b_conv2 = tf.Variable(tf.constant(0.1,shape=[20]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,1,2,1],strides=[1,1,2,1],padding='SAME')
    
    h_pool2_flat = tf.reshape(h_pool2,[-1,4 * 20])
    
    weight1 = tf.Variable(tf.random_normal([4 * 20, 15], stddev=1, seed=1))
    bias1 = tf.Variable(tf.constant(0.1,shape=[15]))
    '''
    weight2 = tf.Variable(tf.random_normal([15, 15], stddev=1, seed=1))
    bias2 = tf.Variable(tf.constant(0.1,shape=[15]))

    weight3 = tf.Variable(tf.random_normal([15, 15], stddev=1, seed=1))
    bias3 = tf.Variable(tf.constant(0.1,shape=[15]))
    '''

    weight5 = tf.Variable(tf.random_normal([15, 1], stddev=1, seed=1))


    a1 = tf.nn.relu(tf.matmul(h_pool2_flat, weight1)+bias1)
    a1_drop = tf.nn.dropout(a1,keep_prob)
    '''
    a2 = tf.nn.relu(tf.matmul(a1_drop, weight2)+bias2)
    a3 = tf.nn.relu(tf.matmul(a2, weight3)+bias3)
    '''
    y  = tf.matmul(a1_drop, weight5)


    cross_entropy = tf.reduce_mean(tf.square(y_-y))
    #cross_entropy = tf.reduce_mean(y)
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    saver = tf.train.Saver()
    model_name = 'Models/extract_HI_v1/cnn_v1'
    with tf.Session() as sess:
        sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
        init_op = tf.global_variables_initializer()       
        try:
            saver.restore(sess,model_name)
            STEPS = 2000
        except:
            print("Start from beginning\n")
            sess.run(init_op)
            STEPS = 30000
        # 训练模型。
        
        
        for i in range(STEPS):
            start = (i*batch_size) % 90480
            end = (i*batch_size) % 90480 + batch_size
            sess.run([train_step, y, y_], feed_dict={x: data_x[start:end], y_: data_y[start:end] , keep_prob : 1})
            if i % 1000 == 0:
               total_cross_entropy = sess.run(cross_entropy, feed_dict={x: data_x, y_: data_y, keep_prob : 1})
               print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
        
        y_train = sess.run(y,feed_dict={x: data_x, y_: data_y, keep_prob : 1})
        y_test = sess.run(y,feed_dict={x: data_x_test, y_: data_y_test, keep_prob : 1})
        saver.save(sess,model_name)      
    return y_train,y_test,total_cross_entropy
        
(y_train_set,y_test,total_cross_entropy)=RUL_predict()
'''
for i in range(len(y_train)-1):
    if y_train[i+1] - y_train[i] <50:
        y_train[i+1] = 0.9 * y_train[i] +  0.1 * y_train[i+1]
'''
'''
fig = plt.figure(figsize=(15, 5))
plt.plot(data_y_test[0:500]) 
plt.plot(y_test[0:500]) 
'''

model_name_lstm = "Models/time_HI_v30/lstm_v1"

index_front = 0
index_back = 1
while index_back < len(y_train_set):
    if y_train_set[index_back] - y_train_set[index_back - 1] > 100:
        y_train = y_train_set[index_front:index_back]
        x_time = 1
        X_time = np.zeros(len(y_train))
        for i in range(len(y_train)):
            X_time[i] = x_time
            x_time = x_time + 1
        
        
        data = {
              tf.contrib.timeseries.TrainEvalFeatures.TIMES: X_time,
              tf.contrib.timeseries.TrainEvalFeatures.VALUES: y_train,
              }
        
        reader = NumpyReader(data)
        
        train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
              reader, batch_size=16, window_size=50)
        
        estimator = ts_estimators.TimeSeriesRegressor(
              model=_LSTMModel(num_features=1, num_units=32),
              optimizer=tf.train.AdamOptimizer(0.001),
              model_dir=model_name_lstm)
        
        estimator.train(input_fn=train_input_fn, steps=100)
        
        
        index_front = index_back
        
    index_back += 1






x_time_test = 1
X_time_test = np.zeros(len(y_test[0:90]))
for i in range(len(y_test[0:90])):
    X_time_test[i] = x_time_test
    x_time_test = x_time_test + 1




data2 = {
          tf.contrib.timeseries.TrainEvalFeatures.TIMES: X_time_test,
          tf.contrib.timeseries.TrainEvalFeatures.VALUES: y_test[0:90],
          }

reader2 = NumpyReader(data2)


evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader2)
evaluation = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)
# Predict starting after the evaluation
(predictions,) = tuple(estimator.predict(
            input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
                    evaluation, steps=500)))

x_time = 1
X_time = np.zeros(99999)
for i in range(99999):
    X_time[i] = x_time
    x_time = x_time + 1
fig = plt.figure(figsize=(15, 5))
py = predictions['mean']
plt.plot(X_time_test[:90],y_test[:90]) 
plt.plot(X_time[90:200],y_test[90:200]) 
plt.plot(predictions['times'][:500],predictions['mean'][:500]) 
fig.show()

