#
# # import wget
# #
# # url = "https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip"
# #
# # zip_10percent = wget.download(url)
#
# """Unzippind entire dataset using Zipfile"""
#
# # import zipfile as zp
# #
# # unzip_10percent =  zp.ZipFile("10_food_classes_10_percent.zip",'r')
# # unzip_10percent.extractall()
# # unzip_10percent.close()
#
# #import wget
#
# #wget.download("https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py")
#
# import os
# from helper_functions import create_tensorboard_callback,plot_loss_curves
# #from  imageClassification import print_draw
# import keras.applications.efficientnet
#
# train_dir = "10_food_classes_10_percent/train"
# test_dir = "10_food_classes_10_percent/test"
# import numpy as np
# import pathlib
#
# path_dir = pathlib.Path(train_dir)
# class_names = np.sort(sorted([item.name for item in path_dir.glob('*')]))
#
# print(path_dir , class_names)
#
# """We need to do image preprocessing """
#
# import tensorflow as tf
# tf.random.set_seed(23)
# import tensorflow.keras as keras
#
# from tensorflow.keras.preprocessing import image_dataset_from_directory
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# imagesize = (224,224)
# """Rescaling"""
# scalebef_train = ImageDataGenerator(rescale=1/255.)
# scalebef_test = ImageDataGenerator(rescale=1/255.)
#
# train_10_percent = scalebef_train.flow_from_directory(directory=train_dir,
#                                                 batch_size=32,
#                                                 target_size=imagesize,
#                                                 class_mode="categorical")
# test_10_percent = scalebef_test.flow_from_directory(directory=test_dir,
#                                                batch_size=32,
#                                                target_size=imagesize,
#                                                class_mode="categorical")
#
# """Building transfer learning model from tensor hub https://tfhub.dev/google/efficientnet/b0/classification/1"""
# #import tensor_hub as hub
#
# """Starting EfficientNetBo architecture"""
# """Creating a base model in efficirnnet"""
#
# baseModel1 = keras.applications.efficientnet.EfficientNetB0(include_top=False) # remove top layer as we need to use our layer
#
# """ Freeze base model"""
# baseModel1.trainable = False
#
#
# """create input layer"""
#
# input_model = keras.layers.Input(shape = (224,224,3), name =" input_Layer")
#
# """Passing in put in base model"""
# x = baseModel1(input_model)
# """Lets check X shape """
# print(f" after input passing x Shape :{x.shape}")
#
# """Average pool on base model"""
#
# x = keras.layers.GlobalAvgPool2D(name="Average_pool")(x)
# """Lets check new average pool Shape """
# print(f" average pool after x :{x.shape}")
#
# """Creating a output layer"""
#
# output_layer = keras.layers.Dense(units=10, activation="softmax", name="Output_Dense")(x)
#
# """Combine input and output in actual model """
# actual_model = tf.keras.Model(input_model,output_layer)
#
# """Compile our model """
#
# actual_model.compile(loss = "categorical_crossentropy",
#                      metrics =["accuracy"],
#                      optimizer ="adam")
#
# history = actual_model.fit(train_10_percent,
#                  epochs = 5,
#                  steps_per_epoch = len(train_10_percent),
#                  validation_data = test_10_percent,
#                  validation_steps = 0.25 * len(test_10_percent),
#                 callbacks=[create_tensorboard_callback("transfer_learning", "EfficientNetB0_classic")]
#                  )
# #plot_loss_curves(history)
# """Ending EfficientNetBo architecture"""
# """Starting ResNet50v2 architecture"""
#
# efficient_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
#
# """Resnet through tensor hub"""
# import tensorflow_hub as hub
#
# def createModel(url , no_of_output):
#
#     """create a model from given URl
#     Args:
#         URl for specific transfer learning is given , nor of putput is essentially needed
#
#     Return:
#         Hopefully create a model and return  the actual model in same"""
#     """Download its pretrained model and save it as keras layer"""
#     extracting_actual = hub.KerasLayer(url,
#                             trainable=False,
#                             name= "feature_extracyion",
#                             input_shape = (imagesize)+(3,))
#     """Creating my own model"""
#     model = keras.Sequential([
#             extracting_actual,#use same above layer as the base layer
#             keras.layers.Dense(no_of_output,activation="softmax",name= "Output_layer")
#     ])
#
#     return model
#
# efficient_model = createModel(efficient_url,10)
# """Compileing above efficient model"""
# efficient_model.compile(optimizer = keras.optimizers.Adam(),
#                         metrics = ["accuracy"],
#                         loss= "categorical_crossentropy")
#
# efficient_history = efficient_model.fit(train_10_percent,
#                                         epochs =5,
#                                         steps_per_epoch = len(train_10_percent),
#                                         validation_data=test_10_percent,
#                                         validation_steps=0.25 * len(test_10_percent),
#                                         callbacks=[create_tensorboard_callback(dir_name="transfer_learning",experiment_name="efficientNetthroughURl")])
#
# resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"
#
# resnet_model = createModel(resnet_url,10)
#
# resnet_model.compile(loss = "categorical_crossentropy",
#                      metrics = ["accuracy"],
#                      optimizer=keras.optimizers.Adam())
#
# resnet_model.fit(train_10_percent,
#                  epochs=5,
#                  steps_per_epoch=len(train_10_percent),
#                  validation_data=test_10_percent,
#                  validation_steps=0.25 * len(test_10_percent),
#                  callbacks=[create_tensorboard_callback(dir_name="transfer_learning",experiment_name="ResnetModel_through_URL")])
#
# imageNet_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
#
# imagenet_model = createModel(imageNet_url,10)
#
# imagenet_model.compile(loss = "categorical_crossentropy",
#                        optimizer=keras.optimizers.Adam(),
#                        metrics=["accuracy"])
# imagenet_history = imagenet_model.fit(train_10_percent,
#                                       epochs=5,
#                                       steps_per_epoch=len(train_10_percent),
#                                       validation_data=test_10_percent,
#                                       validation_steps=len(test_10_percent)//32,
#                                       callbacks=[create_tensorboard_callback(dir_name="transfer_learning",experiment_name="imageNet_through_URl")])



import tensorflow as tf
tf.random.set_seed(23)


input = (1,2,2 ,4)

dummy = tf.random.normal(input)
print(dummy.shape)
print(dummy)
dummy1 = tf.keras.layers.GlobalMaxPooling2D()(dummy)
print(dummy1.shape)
print(dummy1)
dummy2 = tf.squeeze(dummy)
print(dummy2.shape)
print(dummy2)



