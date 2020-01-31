import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data2803 = pd.read_csv('Data1/MgII2803data.txt', delimiter=' ', header = None)
data2796 = pd.read_csv('Data1/MgII2796data.txt', delimiter=' ', header = None)
labels = pd.read_csv('Data1/labels.txt', delimiter='\t', header = None)

data2803_2 = pd.read_csv('Data2/MgII2803data.txt', delimiter=' ', header = None)
data2796_2 = pd.read_csv('Data2/MgII2796data.txt', delimiter=' ', header = None)

label = []

for i,row in enumerate(np.transpose(data2803)):
    x = np.array(data2803.iloc[i])
    y = np.array(data2796.iloc[i])
    if i == 0:
        singlecloud = np.array([np.stack((x,y))])
    else:
        singlecloud = np.append(singlecloud,[np.stack((x,y))],axis = 0)

    label.append([0])

for i,row in enumerate(np.transpose(data2803_2)):
    x = np.array(data2803_2.iloc[i])
    y = np.array(data2796_2.iloc[i])
    if i == 0:
        doublecloud = np.array([np.stack((x,y))])
    else:
        doublecloud = np.append(doublecloud,[np.stack((x,y))],axis = 0)
    label.append([1])

dataTrain = np.append(singlecloud,doublecloud,axis=0)
label = np.asarray(label)
dataTrain = np.swapaxes(dataTrain, 1, 2)

x_train, x_test, y_train, y_test = train_test_split(dataTrain, label, test_size=0.1, random_state=42)

plt.scatter(range(len(x_train[0])), np.transpose(x_train[0])[0])
#plt.scatter(range(len(x_train[0])), x_train[0][1])



model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=100, kernel_size=2, use_bias=True, activation='relu', input_shape=(450,2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(filters=100, kernel_size=5, use_bias=True, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(filters=100, kernel_size=10, use_bias=True, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool1D(pool_size=2,strides=2),
    tf.keras.layers.Conv1D(filters=100, kernel_size=5, use_bias=True, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(filters=100, kernel_size=10, use_bias=True, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(filters=100, kernel_size=20, use_bias=True, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool1D(pool_size=2,strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation = 'relu'),
    tf.keras.layers.Dense(50, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'relu'),
    tf.keras.layers.Dense(2, activation = 'softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                loss='sparse_categorical_crossentropy',
                metrics= [tf.keras.metrics.SparseCategoricalAccuracy()])

model.fit(x_train, y_train, batch_size=400, epochs = 3, validation_data=(x_test, y_test))
