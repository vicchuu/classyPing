import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf
tf.random.set_seed(3)
from keras.datasets  import mnist
import keras.datasets as dt
import cv2 as cv
import PIL as pillow


(xtrain, ytrain) , (xtest, ytest) = mnist.load_data()

print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)
#plt.imshow(xtrain[23])
#plt.show()
xtrain = xtrain / 255
xtest = xtest / 255

"""Building neural network"""

# model = keras.Sequential()
# model.add(keras.layers.Flatten(input_shape = (28,28),))
# model.add(keras.layers.Dense(128, activation="relu"))
# model.add(keras.layers.Dense(10, activation= "sigmoid"))
#
# model.compile( optimizer = "adam" , loss= "sparse_categorical_crossentropy",metrics =["accuracy"])
#
# model.fit(xtrain , ytrain ,epochs=10)
# """If we try to change epoch from 10 to 20 , there is no major difference's """
#

#2,3,4,5

const = tf.constant([[3,2],
                     [1,3],
                     [9,1]])
#tf.set
const= tf.random.shuffle(const, seed=23)
#print(const , const.ndim)

arr=[[[
        [1,2,3,4,5],
        [6,7,8,9,10],
        [11,12,13,14,15],
        [16,17,18,19,20]],

        [[21,22,23,24,25],
        [26,27,28,29,30],
        [31,32,33,34,35],
        [36,37,38,39,40]],

        [[41,42,43,44,45],
        [46,47,48,49,50],
        [51,52,53,54,55],
        [56,57,58,59,60]]],

   [ [[61, 62, 63, 64, 65],
    [66,67, 68, 69, 70],
    [71, 72, 73, 74, 75],
    [76, 77, 78, 79, 80]],

    [[81, 82, 83, 84, 85],
     [86, 87, 88, 89, 90],
     [91, 92, 93, 94, 95],
     [96, 97, 98, 99, 100]],

    [[101, 102, 103, 104, 105],
     [106, 107, 108, 109, 110],
     [111, 112, 113, 114, 115],
     [116, 117, 118, 119, 120]]
]]

tens_arr = tf.constant(arr )
print(tens_arr)

print("**********************")
print(tens_arr[:1,:1,:-1,:-1])

D = tf.constant(np.random.randint(1,100,size=50).astype(np.float32))

print(D)

print("maximum value :", tf.reduce_max(D))

print("Minimm value :",tf.reduce_min(D))

print("Mean Value :",tf.reduce_mean(D))

print("Median :",np.median(D))

#print("Mode :",tf.unique_with_counts(D))

print("Std Deviation :",tf.math.reduce_std(D))

print("Variance :",tf.math.reduce_variance(D))

print("Find maxarg ",np.argmax(D))
print("Find maxarg ",tf.argmax(D))

print("Min args :",np.argmin(D))

print("Sorted :",np.argsort(D))

tensor = [ 0,0,1,1,2,3,5,5]

print(tf.one_hot(tensor, depth = 6)) #depth must be no of unique values

print("Physical devices :",tf.config.list_physical_devices("GPU"))

print("Logical devices :",tf.config.list_logical_devices("GPU"))

print(tf.test.gpu_device_name())
