
import pandas as ps
import numpy as np
import sklearn.datasets

data_set = sklearn.datasets.load_breast_cancer()

data_frame = ps.DataFrame(data_set.data)

data_frame["result"] = ps.DataFrame(data_set.target)

X = data_frame.drop(columns="result" , axis=1)
Y=data_frame["result"]

from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(X,Y, test_size=0.2, random_state=3)

#print(X.shape )

"""We need to standradise our data , so we can get better results"""

from sklearn.preprocessing import StandardScaler

scaling = StandardScaler()
updatedX = (scaling.fit_transform(trainX))

updatedtestX = (scaling.transform(testX))


print(X.shape)

import tensorflow as tf
tf.random.set_seed(3)
import tensorflow.keras as keras

#setting up layer on network

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape = (30 ,) ))
model.add(keras.layers.Dense(20, activation = "relu"))
model.add(keras.layers.Dense(2, activation= "sigmoid"))


"""Keras compile"""
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy" , metrics = ["accuracy"])

"""training the neural network"""
history = model.fit(updatedX,trainY , validation_split=0.1 , epochs=10)


loss , accuracy = model.evaluate(updatedtestX , testY)

print(loss, accuracy)

import matplotlib.pyplot as ply


ply.plot(history.history["accuracy"])
ply.plot(history.history["val_accuracy"])
ply.xlabel("Epoch")
ply.ylabel("accuracy")
ply.legend(['training data ','validation data'])
ply.show()