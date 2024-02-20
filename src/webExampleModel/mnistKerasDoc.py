""" 
# 
# Model implemented on mnist from
# https://keras.io/examples/vision/mnist_convnet/
# 
# """



"""
## Setup
"""

import numpy as np
import keras
from keras import layers
 
"""
## Prepare the data
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Build the model
"""

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

"""
## Train the model
"""

batch_size = 128
epochs = 10

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)



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
#accuracy_plot_path = os.path.join('../assets/outputPlot/fromDocExample', 'accuracy_plot_mnist_doc.png')
accuracy_plot_path = os.path.join('assets/outputPlot/fromDocExample','accuracy_plot_mnist_doc.png')
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
#loss_plot_path = os.path.join('../assets/outputPlot/fromDocExample', 'loss_plot_mnist_doc.png')
loss_plot_path = os.path.join('assets/outputPlot/fromDocExample','loss_plot_mnist_doc.png')
plt.savefig(loss_plot_path)
plt.close()


"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])