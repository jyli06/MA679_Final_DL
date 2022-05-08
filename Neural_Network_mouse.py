from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

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

#
# get the first 70% rows as trainingset
df_train = df_merge.sample(frac=0.7, random_state=1)

# get the x and y
x_train = df_train.drop(['behavior_1'], axis=1)
y_train = df_train['behavior_1']

# get the rest 70% rows as testset
df_test = df_merge.drop(df_train.index)

# get the x and y
x_test = df_test.drop(['behavior_1'], axis=1)
y_test = df_test['behavior_1']

## for time series training set

# set the datetime as index
df_merge.index = pd.to_datetime(df_merge.index)

# get the first 70% rows as trainingset
# df_train_ts = df_merge.copy()
df_train_ts = df_merge.iloc[:int(len(df_merge)*0.7)]

# get the x and y
x_train_ts = df_train_ts.drop(['behavior_1'], axis=1)
y_train_ts = df_train_ts['behavior_1']

# get the rest 30% rows as testset
df_test_ts = df_merge.iloc[int(len(df_merge)*0.7):]

# get the x and y
x_test_ts = df_test_ts.drop(['behavior_1'], axis=1)
y_test_ts = df_test_ts['behavior_1']


# change the data into time series

cols = list(df_train_ts)[0:111]
#Date and volume columns are not used in training.
print(cols) #['0', '1', ..., '109', 'behavior_1']

#New dataframe with only training data - 110 columns
df_for_training = df_train_ts[cols].astype("float")
df_for_testing = df_test_ts[cols].astype("float")

# another way to capture the training data without the behavior_1 column
df_y_train = df_train_ts['behavior_1']
df_y_test = df_test_ts['behavior_1']
df_for_training = df_for_training.drop('behavior_1', axis=1)
df_for_testing = df_for_testing.drop('behavior_1', axis=1)

# df_for_plot=df_for_training.tail(5000)
# df_for_plot.plot.line()

#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = StandardScaler()
scaler1 = scaler.fit(df_for_training)
df_for_training_scaled = scaler1.transform(df_for_training)
scaler2 = scaler.fit(df_for_testing)
df_for_testing_scaled = scaler2.transform(df_for_testing)

# if we don't use scale, we just transform the data into a numpy array
# df_for_training_scaled = df_for_training.values

#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
#In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training).

#Empty lists to be populated using formatted training data
trainX = []
trainY = []

n_future = 1   # Number of time points we want to look into the future based on the past days.
n_past = 14  # Number of past time points we want to use to predict the future.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (4328, 110)
#4328 refers to the number of data points and 111 refers to the columns (multi-variables).
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[(i - n_past):i, 0:df_for_training_scaled.shape[1]])
    trainY.append(df_y_train[(i + n_future - 1):(i + n_future)])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))


testX = []
testY = []
for i in range(n_past, len(df_for_testing_scaled) - n_future +1):
    testX.append(df_for_testing_scaled[(i - n_past):i, 0:df_for_testing_scaled.shape[1]])
    testY.append(df_y_test[(i + n_future - 1):(i + n_future)])

testX, testY = np.array(testX), np.array(testY)

print('testX shape == {}.'.format(testX.shape))
print('testY shape == {}.'.format(testY.shape))

# another way to capture the training data without the loop
# trainX = df_for_training_scaled[:int(len(df_for_training_scaled)*0.7), 0:df_for_training_scaled.shape[1]]

###############################################################################
# Neural Network Models
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
                             tf.keras.layers.Dense(34, activation='relu'), # add a dropout layer
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(1, activation='sigmoid')])

# model = tf.keras.Sequential([tf.keras.layers.Dense(68, activation='relu'),
#                              tf.keras.layers.Dense(2, activation='sigmoid')])

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['sparse_categorical_accuracy'])

# Define the Keras TensorBoard callback.
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

model.fit(x_train, y_train, batch_size=64,epochs=10, validation_data=(x_test, y_test), validation_freq=1)#, callbacks=[tensorboard_callback])
model.evaluate(x_test, y_test, verbose=2)

model.summary()


# %tensorboard --logdir logs/fit/

# fill NA in dataframe by mean
df_maze_behavior.fillna(df_maze_behavior.mean(), inplace=True)

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# define a learning rate with weight decay
# lr = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=0.01,
#     decay_steps=100,
#     decay_rate=0.96,
#     staircase=True)

# apply LSTM to the data
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# define the model
model = Sequential()
model.add(LSTM(units=64, return_sequences=True,input_shape=(trainX.shape[1], trainX.shape[2]), activation='relu'))
model.add(LSTM(units=32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1], activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.01), loss = 'binary_crossentropy', metrics=['accuracy'])
model.summary()

# fit the model
history = model.fit(trainX, trainY,validation_data=(testX, testY),epochs=40, batch_size=16, validation_freq=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()