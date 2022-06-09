import tensorflow as tf
tf.random.set_seed(23)

from tensorflow import keras
from tensorflow.keras import datasets

(trainImage, trainLabel) , (testImage , testLabel) = datasets.fashion_mnist.load_data()

print(trainImage.shape , trainLabel.shape)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


import matplotlib.pyplot as plt
import random
# plt.figure(figsize=(8,8))
#
# for i in range(8):
#     ax= plt.subplot(4,2,i+1)
#     random_index = random.choice(range(len(trainLabel)))
#     plt.imshow(trainImage[random_index])
#     plt.title(class_names[trainLabel[random_index ]])
#     plt.axis(False)
#
# plt.imshow(trainImage[1] , cmap=plt.cm.binary)
# plt.show()
#print()

#print("Checking if image is null :",(trainImage.isNull().sum()))

"""Building Neural Network's"""

image_classy =  keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28, )),
        keras.layers.Dense(100 , activation ="relu"),
        keras.layers.Dense(100, activation ="relu"),
        keras.layers.Dense(10,activation = "softmax")]
)

"""Compiling Neural network"""

image_classy.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                     optimizer =tf.keras.optimizers.Adam(learning_rate = 0.0001),
                     metrics = ["accuracy"])
"""Fit the model"""


#image_classy.fit(trainImage,trainLabel,epochs=25 , verbose =0, validation_data =(testImage,testLabel))

"""Acuuracy score in both trainning and test data is less, if we do scling then it will be better """
print(trainImage.min(), trainImage.max())
"""0 to 255 we need to scale / normalization"""
trainImage_norm = trainImage/255.0
testImage_norm = testImage/255.0


print(trainImage_norm.min(), testImage_norm.max())
"""checking a learning rate in below model"""

l_r = keras.callbacks.LearningRateScheduler(lambda  epoch:  1e-4 * 10**(epoch/5))


history =image_classy.fit(trainImage_norm,trainLabel,epochs=10 , verbose =1, validation_data =(testImage_norm,testLabel),
                          callbacks= [l_r])

print(image_classy.summary())

test_pred = image_classy.predict(testImage_norm)
f= lambda x : round(x.argmax())
test_pred_label = [ f(x) for x in test_pred]
print( test_pred_label[:10])

from sklearn.metrics import confusion_matrix ,plot_confusion_matrix

con_mat= (confusion_matrix(test_pred_label,testLabel))
#plot_confusion_matrix(X=test_pred_label,y_true=testLabel ,labels="confusion matrix",estimator=10)
import seaborn as sns
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(con_mat, linewidths=1, annot=True, ax=ax, fmt='g')
#plt.show()
import pandas as ps
# ps.DataFrame(history.history).plot()
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.show()

"""plot leaning rate """

# rate = 1e-4 *(10**(tf.range (25)/5))
# plt.semilogx(rate , history.history["loss"])
# plt.xlabel("learning rate")
# plt.ylabel("Loss")
# plt.title("Learning rate vs LOsss")
# plt.show()

import numpy as np
img_index=1899
img = trainImage_norm[img_index]
img= np.expand_dims(img,0)
# print("correct Label :",class_names[trainLabel[img_index]])
# predict_label = image_classy.predict(img)
# crct_index = np.argmax(predict_label)
# print("predicted by model :",class_names[crct_index])


weights , biases = image_classy.layers[1].get_weights()
print(weights ,weights.shape )
print("*********")
print(biases,biases.shape)


from tensorflow.keras.utils import plot_model

plot_model(image_classy,show_shapes=True )