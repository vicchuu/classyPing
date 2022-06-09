"""Downloading using wget """

# import wget
#
# url="https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip"
# entire_zipFile= wget.download(url)
#
# """After downloading we need to unzip entire folder"""
# import zipfile as zip
#
# zip_obj = zip.ZipFile(entire_zipFile,"r")
# zip_obj.extractall()
# zip_obj.close()

"""Congrats u downloed it sucessfully
check basic folder and file using os"""
import os
folder_name = os.listdir("pizza_steak")
print(folder_name )
trainSet_folder = "pizza_steak/train"
testSet_folder = "pizza_steak/test"

import random
import matplotlib.image as pltImage
import matplotlib.pyplot as plt

import tensorflow as tf

"""draw any random image based on name"""
def showImage(path,folderName ):
    currentPath = path +folderName

    imageName = random.sample(os.listdir(currentPath),1)
    img = pltImage.imread(currentPath+"/"+imageName[0])
    plt.imshow(img)
    plt.title(folderName)
    #plt.show()
    print(f"Total no of elements  in :{folderName},{len(os.listdir(currentPath))}")
    print(f" Image size :{ img.shape}")
    show_script =input("Do you need to check image RGB valus in tensor (y/n) ?")
    show=False
    if(show_script=="y"):
        show=True
    elif(show_script=="n"):
        show = False
    if(show):
        print("********")
        print(img.min(),(img.max()))
        print("*******")
        print(img)


    return img

#showImage(testSet_folder ,"/steak")

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(23)

"""preprocess the data with scaling / normalization """


#train_image_datagen = ImageDataGenerator(

train_image_datagen_augmented = ImageDataGenerator(rescale =1.0/255,
                                         rotation_range=0.2,
                                         width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         shear_range=0.2,
                                        horizontal_flip=0.2,
                                         zoom_range=0.2,
                                         )
test_image_datagen = ImageDataGenerator(rescale=1.0/255)

train_data = train_image_datagen_augmented.flow_from_directory(trainSet_folder , batch_size=32 ,
                                                      target_size=(224,224),
                                                      class_mode="binary",
                                                      shuffle=True
                                                      )
test_data = test_image_datagen.flow_from_directory(testSet_folder,
                                                   batch_size=32,
                                                   target_size=(224,224),
                                                   class_mode="binary")

"""Now input preprocessing  is ready """

print(train_data.n) #total  images
print(f"steps :{train_data.n/32}")

print(f" next :{train_data.next}")

"""Create CNN layer"""
keras_layer = keras.Sequential([
            keras.layers.Input((224,224,3)), #loss: 0.3390 - accuracy: 0.8533 - val_loss: 0.4092 - val_accuracy: 0.7960 - without augmented
            keras.layers.Conv2D(filters=10,activation="relu",padding="valid",kernel_size=(3,3)),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(10,3,activation="relu"),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(10,3,activation="relu"),
            keras.layers.MaxPool2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(1,activation="sigmoid")

            # keras.layers.Flatten(input_shape=(224,224,3)),loss: 0.6433 - accuracy: 0.7340 - val_loss: 0.7446 - val_accuracy: 0.6080
            # keras.layers.Dense(100,activation="relu"),
            # keras.layers.Dense(100,activation ="relu"),
            # keras.layers.Dense(100,activation="relu"),
            # keras.layers.Dense(100,activation="relu"),
            # keras.layers.Dense(1,activation="sigmoid")

                # keras.layers.Flatten(input_shape=(224,224,3)), loss: 0.6932 - accuracy: 0.5000 - val_loss: 0.6932 - val_accuracy: 0.5000
                # keras.layers.Dense(4 , activation="relu"),
                # keras.layers.Dense(4,activation="relu"),
                # keras.layers.Dense(4,activation="relu"),
                # keras.layers.Dense(1,activation="sigmoid")



            # keras.layers.Input ((224,224,3)),
            # keras.layers.Conv2D(filters=10,
            #                     batch_size=32,
            #                     kernel_size=3,
            #                     padding="valid",
            #                     strides=1,
            #                     activation="relu",
            #                     input_shape=(224,224,3)),
            # keras.layers.Conv2D(10,3,activation="relu"),
            # keras.layers.Conv2D(10,3,activation="relu"),
            # keras.layers.Flatten(),
            # keras.layers.Dense(1,activation="sigmoid")

])
keras_layer.compile(optimizer= keras.optimizers.Adam(learning_rate=0.001),loss="binary_crossentropy",metrics="accuracy")
"""fit i the model """

history = keras_layer.fit(train_data,
                          epochs=5,
                          steps_per_epoch=len(train_data),
                          validation_data=test_data,
                          validation_steps=len(test_data))



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
print(keras_layer.summary())
print_draw(history)
"""there is a lot overfitting in trainning model , to over come overfitting we  need to 1) increasemore trainnng data
    2) data Augmentation - process to altering training images to fit in more diversity like zoomin , shear , horizontalflip ,etc"""