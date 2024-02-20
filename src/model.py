import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from keras.layers import MaxPooling2D
from keras.utils import to_categorical

from keras.datasets import mnist

import matplotlib.pyplot as plt
import os


class Model:
    
    def initData(self, dataSet = mnist, classesNumber = 10):
        self.classesNumber = classesNumber
        #load data
        (self.xTrain, self.yTrain), (self.xTest, self.yTest) = dataSet.load_data()
        #normalizing e label encodig
        self.xTrain = self.xTrain/255
        self.xTest = self.xTest/255
        self.yTrain = to_categorical(self.yTrain, self.classesNumber)
        self.yTest = to_categorical(self.yTest, self.classesNumber)
        
        
    def buildModel(self,kernelSize,inputShape):
        self.kernelSize = kernelSize
        self.inputShape = inputShape
        #build the model
        self.model = Sequential()
        self.model.add(Conv2D(64,self.kernelSize,input_shape=self.inputShape,activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.5))
        self.model.add(Conv2D(64,self.kernelSize,input_shape=self.inputShape,activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(256,activation="relu"))
        self.model.add(Dense(self.classesNumber,activation="softmax"))
        
        #print model overview
        #self.model.summary()
        
        self.model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
        
        
    def trainModel(self, batchSize, epochs):
        self.batchSize = batchSize
        self.epochs = epochs
        #model training
        self.history =  self.model.fit(self.xTrain, self.yTrain, batch_size=self.batchSize, epochs = self.epochs, verbose=1, validation_data=(self.xTest,self.yTest))
        
        
    def plotChart(self, accuracyChartFileName, lossChartFileName):
        # Accuracy chart print and save
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        accuracy_plot_path = os.path.join('outputPlot/myModelPlot/accuracy', accuracyChartFileName)
        plt.savefig(accuracy_plot_path)
        plt.close()

        # Loss chart print adn save
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Save loss chart
        loss_plot_path = os.path.join('outputPlot/myModelPlot/loss', lossChartFileName)
        plt.savefig(loss_plot_path)
        plt.close()