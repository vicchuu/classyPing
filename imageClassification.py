#import cnn_pistacho

# import wget
#
# url = "https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip"
#
# zip_File_Name = wget.download(url)
# import zipfile as zp
#
# zop_obj = zp.ZipFile(zip_File_Name, 'r')
#
# zop_obj.extractall()
# zop_obj.close()

import os

train_dir = "10_food_classes_all_data/train"
test_dir = "10_food_classes_all_data/test"
import numpy as np
import pathlib

path_dir = pathlib.Path(train_dir)
class_names = np.sort(sorted([item.name for item in path_dir.glob('*')]))

import matplotlib.pyplot as plt
import random
def viewRandomImage(path ):

    class_name = random.choice(class_names)
    exactPath = path +"/"+class_name
    image_name = random.sample(os.listdir(exactPath),1)
    img = plt.imread(exactPath+"/"+image_name[0])
    plt.imshow(img)
    plt.title(exactPath+"/"+image_name[0])
    plt.xlabel(img.shape)
    plt.show()
    return img

#viewRandomImage(train_dir)
"""preprocess date """

import tensorflow as tf
tf.random.set_seed(23)
from tensorflow.keras.preprocessing.image  import ImageDataGenerator

"""rescaling"""

train_dataGen = ImageDataGenerator(rescale= 1/255.,
                                   horizontal_flip=True,
                                   shear_range=0.2,
                                   width_shift_range=0.2,
                                   rotation_range=20,
                                   height_shift_range=True,


                                   )

test_dataGen = ImageDataGenerator(rescale=1/255.)


"""Load data from directories and turn it to batches"""
train_data = train_dataGen.flow_from_directory(train_dir,
                                               target_size=(224,224),
                                               batch_size=32,
                                            class_mode="categorical"
                                               )
test_data = test_dataGen.flow_from_directory(test_dir,
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode="categorical")
"""Cretate CNN model """

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Activation,MaxPool2D,Flatten

image_model = Sequential([
        Conv2D(10,3,input_shape=(224,224,3)),
        Activation("relu"),
        Conv2D(10,3,activation="relu"),
        MaxPool2D(),
        Conv2D(10,3,activation="relu"),
        MaxPool2D(),
        Flatten(),
        Dense(10,activation="softmax")

        # Conv2D(10,3, input_shape=(224,224,3)), #witout data augmentation 129s 548ms/step - loss: 0.2100 - accuracy: 0.9388 - val_loss: 4.0383 - val_accuracy: 0.2740
        # Activation("relu"), # try to improve overfitt ,  remove extra COnvo layer
        # Conv2D(10,3,activation="relu"),
        # MaxPool2D(),
        # Conv2D(10,3,activation="relu"),
        # Conv2D(10,3,activation="relu"),
        # MaxPool2D(),
        # Flatten(),
        # Dense(10,activation="softmax")
])

"""Compiling a model"""

image_model.compile(optimizer="adam",
                    loss="categorical_crossentropy",
                    metrics=["accuracy"])

print(len(train_data),len(test_data))
"""Fit in model """
history = image_model.fit(train_data,
                epochs=5,
                steps_per_epoch=0.1 *len(train_data),
                validation_data=test_data,
                validation_steps=0.1 * len(test_data))


def print_draw(history):

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    epochs = range(len(loss))
    #

    #plt loss
    plt.plot(epochs,loss,label ="Trainng loss")
    plt.plot(epochs,val_loss,label = "Val_loss")
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.legend()

    #plt accuracy
    plt.plot(epochs , accuracy,label ="Trainning accuracy")
    plt.plot(epochs,val_accuracy,label="Val_accuracy")
    plt.title("accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
print(image_model.summary())
print_draw(history)