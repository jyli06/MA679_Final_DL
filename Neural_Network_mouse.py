from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt

# load mat file

data_maze = loadmat("data/Zero_Maze/608034_409/Day_1/Trial_001_0/binned_zscore.mat")

df_maze = pd.DataFrame(data_maze['binned_zscore'])

data_maze_behavior = loadmat("data/Zero_Maze/608034_409/Day_1/Trial_001_0/binned_behavior.mat")
df_maze_behavior = pd.DataFrame(data_maze_behavior['binned_behavior']).T
df_maze_behavior.columns = ['behavior_1', 'behavior_2']


# merge data_maze and data_maze_behavior
df_merge = pd.concat([df_maze, df_maze_behavior], axis=1)
df_merge.columns.tolist()

# drop the rows where both behavior_1 and behavior_2 are 0
df_merge = df_merge[(df_merge['behavior_1'] != 0) | (df_merge['behavior_2'] != 0)]

df_merge.drop(['behavior_2'], axis=1, inplace=True)
df_merge.shape
df_merge.head()


# get column name in df_maze_behavior
df_maze_behavior.columns.tolist()

# get the first 10% rows as trainingset
df_train = df_merge.sample(frac=0.3, random_state=1)

# get the x and y
x_train = df_train.drop(['behavior_1'], axis=1)
y_train = df_train['behavior_1']

# get the rest 90% rows as testset
df_test = df_merge.drop(df_train.index)

# get the x and y
x_test = df_test.drop(['behavior_1'], axis=1)
y_test = df_test['behavior_1']

import tensorflow as tf
import numpy as np
import tensorboard
from datetime import datetime

# Load the TensorBoard notebook extension.
%load_ext tensorboard

# Clear any logs from previous runs
rm -rf ./logs/

# build a fully connected neural network
model = tf.keras.Sequential([tf.keras.layers.Dense(68, activation='relu'),
                             tf.keras.layers.Dense(34, activation='relu'),
                             tf.keras.dropout(0.5),
                             tf.keras.layers.Dense(2, activation='sigmoid')])

# model = tf.keras.Sequential([tf.keras.layers.Dense(68, activation='relu'),
#                              tf.keras.layers.Dense(2, activation='sigmoid')])

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['sparse_categorical_accuracy'])

# Define the Keras TensorBoard callback.
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

model.fit(x_train, y_train, batch_size=64,epochs=10, validation_data=(x_test, y_test), validation_freq=1)#, callbacks=[tensorboard_callback])
model.evaluate(x_test, y_test, verbose=2)

model.summary()


%tensorboard --logdir logs/fit/

# fill NA in dataframe by mean
df_maze_behavior.fillna(df_maze_behavior.mean(), inplace=True)
