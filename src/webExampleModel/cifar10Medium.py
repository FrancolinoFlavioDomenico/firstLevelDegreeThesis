""" 
# 
# Model implemented from
# https://devashree-madhugiri.medium.com/using-cnn-for-image-classification-on-cifar-10-dataset-7803d9f3b983 
# 
# """

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from keras.layers import GlobalMaxPooling2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import regularizers, optimizers
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

from keras.datasets import cifar10

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Normalizing
X_train = X_train/255
X_test = X_test/255
# One-Hot-Encoding
Y_train_en = to_categorical(Y_train,10)
Y_test_en = to_categorical(Y_test,10)

# Base Model
model = Sequential()
model.add(Conv2D(64,(4,4),input_shape=(32,32,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(64,(4,4),input_shape=(32,32,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dense(10,activation="softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

model.summary()

batch_size = 128
epochs = 10
history = model.fit(X_train, Y_train_en, batch_size=batch_size, epochs = epochs, verbose=1,validation_data=(X_test,Y_test_en))

"""
## print charts
"""
# Plot dell'andamento dell'addestramento
import matplotlib.pyplot as plt
import os
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Salvataggio del grafico in una cartella
accuracy_plot_path = os.path.join('assets/outputPlot/fromDocExample', 'accuracy_plot_cifar10_medium.png')
plt.savefig(accuracy_plot_path)
plt.close()

# Plot dell'andamento della perdita
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Salvataggio il grafico in una cartella
loss_plot_path = os.path.join('assets/outputPlot/fromDocExample', 'loss_plot_cifar10_medium.png')
plt.savefig(loss_plot_path)
plt.close()