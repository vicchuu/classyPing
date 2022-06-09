
import pandas as ps

dataset = ps.read_csv("card_transdata.csv" , nrows=10000)
print(dataset.shape)

print(" Info :",dataset.info())
print(dataset.head())
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler , OneHotEncoder
from sklearn.model_selection import train_test_split

X = dataset.drop("fraud",axis=1)
Y = dataset["fraud"]

xtrain , xtest, ytrain , ytest = train_test_split(X,Y, test_size=0.2 , random_state=23)

column_transfer = make_column_transformer(
    (MinMaxScaler(),["distance_from_last_transaction","ratio_to_median_purchase_price"])
)

print('Xtrain shape :',xtrain.shape)
column_transfer.fit(xtrain)
xtrain_after= column_transfer.transform(xtrain)
xtest_after = column_transfer.transform(xtest)

print(xtrain_after.shape, xtest_after.shape)

#creating a deep neural network



from tensorflow import keras
import tensorflow as tf
tf.random.set_seed(23)
fraud_detect_model =  keras.Sequential([
        keras.layers.Input((xtrain_after.shape[1],)),
        keras.layers.Dense(100, activation = keras.activations.selu),
        keras.layers.Dense(100, activation = "selu"),
        keras.layers.Dense(100, activation = "selu"),
        keras.layers.Dense(100, activation = "selu"),
        keras.layers.Dense(1, activation = keras.activations.sigmoid)]
)


# compiling deep learning model
fraud_detect_model.compile(loss = keras.losses.mae,
                           metrics = ["accuracy"],
                           optimizer = keras.optimizers.Adam(learning_rate=0.001))


learn_learning_rate = keras.callbacks.LearningRateScheduler(lambda  epoch : 1e-4 * 10**(epoch/20))


# fitting in the model
history = fraud_detect_model.fit(xtrain_after, ytrain, epochs =10,verbose =0 ,callbacks = [learn_learning_rate ])

#evaluating the model

eval = fraud_detect_model.evaluate(xtest_after,ytest)

print("Evaluation Score :",eval)

import matplotlib.pyplot as plt

lrs = 1e-4 * (10** (tf.range(10)/20))
cond = tf.constant(9.834342e-4)
print(cond , tf.cast(cond,tf.float16))
#print("lrs :",lrs)
ps.DataFrame(history.history).plot()
plt.semilogx(lrs , history.history["loss"])
#print(help(plt.semilogx()))
plt.xlabel("Epochs")
plt.ylabel("Loss")
#plt.legend()
plt.show()

