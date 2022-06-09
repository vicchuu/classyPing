# import tensorflow as tf
#
# #from tensorflow.keras as keras
#
# from tensorflow import keras
#
# #from keras.layers import Layer
#
# print(tf.__version__)
#
# import numpy as np
#
# tf.random.set_seed(2)
# X = np.arange(1.0 , 45.8 , 0.7).astype(dtype= np.float32)
# print(X.shape)
# Y = X+10 # np.arange(10 , 34,3)
# print(Y , Y.shape)
#
# X = tf.cast(tf.constant(X) , dtype = tf.float32)
#
# Y= tf.cast(tf.constant(Y),dtype=tf.float32)
#
# model = keras. Sequential(
#     [
#         keras.layers.Input(shape=(1,)),
#         keras.layers.Dense(1 , activation = "linear"),
#         keras.layers.Dense(1 , activation = "linear"),
#         # keras.layers.Dense(100, activation = "linear"),
#         # keras.layers.Dense(100, activation="linear"),
#         keras.layers.Dense(1)
 #
#     ]
# )
# #model.add()
#
# #model.add()
#
# # com;iling our model
#
# model.compile(loss = tf.keras.losses.mae , metrics = ["mae"] , optimizer = tf.keras.optimizers.Adam(lr= 0.001))
#
# model.fit(X,Y , epochs = 76)
# pred=model.predict(X)
#
# print(model.summary())
# print(pred)
# from sklearn.metrics import accuracy_score , confusion_matrix
# print(Y)
#
# import matplotlib.pyplot as plt
# plt.scatter(X,Y , color = "green")
# plt.scatter(X,pred , color = "red")
# #plt.show()
#
# import pydot
# import graphviz as gg
# from tensorflow.keras.utils import plot_model
#
# #plot_model(model = model , to_file="vnv.png",show_shapes=True)
#
# model.save("besrModel")
#
# #print("Accuracy  score :",confusion_matrix(pred,(Y) ))
#
#
import pandas as pd
#import pandas as ps
import matplotlib.pyplot as plt

dataset = pd.read_csv("Car.csv")
dataset.drop(["car_ID","symboling"],axis=1,inplace=True)

#future scaling and standardization
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder


dt = make_column_transformer(
    (MinMaxScaler(),["wheelbase","carlength","carwidth","carheight","curbweight","enginesize","boreratio",
                     "stroke","compressionratio","horsepower","peakrpm","citympg","highwaympg"]),
    (OneHotEncoder(handle_unknown="ignore"),["fueltype","aspiration","doornumber","carbody","drivewheel",
                                             "enginelocation","enginetype","cylindernumber"])
)


#print(dataset.info())
#print("Is null :", dataset.isnull().sum())

import numpy as np
#one_hot_data =( pd.get_dummies(dataset))
#print(dt)
# ## noinspection PyTypeChecker
# tocsvfile = ps.DataFrame(one_hot_data)
# tocsvfile.to_csv("onehot.csv")
#print(one_hot_data.head())
from tensorflow import keras
import tensorflow as tf

X =  (dataset.drop("price" , axis=1))

Y= (dataset["price"])

#print(X.info() , Y.shape)
#print(Y.head())

from sklearn.model_selection import train_test_split

xtrain , xtest , ytrain , ytest = train_test_split(X,Y, test_size=0.2 , random_state=23)

print(xtrain.shape, xtest.shape)
#print(ytrain.shape,ytest.shape)



# we need to preprocessing ( sampling and standardization )
dt.fit(xtrain)

x_train_after = dt.transform(xtrain)
x_test_after = dt.transform(xtest)
print("Shape:@@@@@@")
print(x_train_after.shape)
print(x_test_after.shape)

print("xtrain first ",x_train_after[0])
##Deep learning starting from here
#X["stroke"].plot(kind="hist")

#print(X["stroke"].value_counts(ascending=True))
tf.random.set_seed(23)

# creating a neural network model
car_price_pred = keras.Sequential([
    keras.layers.Input((x_train_after.shape[1],)),
    keras.layers.Dense(100 , activation = keras.layers.LeakyReLU(alpha=0.1)),
    keras.layers.Dense(100, activation = "LeakyReLU"),
    keras.layers.Dense(100, activation = "LeakyReLU"),
    keras.layers.Dense(100, activation = "LeakyReLU"),
    keras.layers.Dense(1, activation = "LeakyReLU")
])

#compiling neural network model


car_price_pred.compile(loss = keras.losses.mae,
                       metrics = ["mae"],
                       optimizer = keras.optimizers.Adam( learning_rate= 0.01))

history = car_price_pred.fit(x_train_after,ytrain, epochs = 100 , verbose =1)

pred = car_price_pred.evaluate(x_test_after,ytest)

print(pred)

pd.DataFrame(history.history).plot()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()