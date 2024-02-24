from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.layers import MaxPooling2D

from keras.utils import to_categorical

from keras.datasets import cifar10


class Model:        
    
    CLIENTS_NUM = 2
    ROUNDS_NUM = 2
    MODEL = None
    CLASSESS_NUMBER = 10
    DATASET = cifar10
    KERNEL_SIZE = (4,4)
    INPUT_SHAPE = (32,32,3)
    
    X_TRAIN = None
    Y_TRAIN = None
    X_TEST = None
    Y_TEST = None
    
    
    @classmethod
    def setData(cls, dataSet = None, classes_number = None, kernel_size = None, input_shape = None):
        if(dataSet is not None):
            cls.DATASET = dataSet
            cls.CLASSESS_NUMBER = classes_number
            cls.KERNEL_SIZE = kernel_size
            cls.INPUT_SHAPE = input_shape
            cls.DATASET = dataSet
        #load data
        (cls.X_TRAIN, cls.Y_TRAIN), (cls.X_TEST, cls.Y_TEST) = cls.DATASET.load_data()
        #normalizing e label encodig
        cls.X_TRAIN = cls.X_TRAIN/255
        cls.X_TEST = cls.X_TEST/255
        cls.Y_TRAIN = to_categorical(cls.Y_TRAIN, classes_number)
        cls.Y_TEST = to_categorical(cls.Y_TEST, classes_number)
        
    
    @classmethod
    def setModel(cls):
        #build the model
        cls.MODEL = Sequential()
        cls.MODEL.add(Conv2D(64,cls.KERNEL_SIZE,input_shape=cls.INPUT_SHAPE,activation="relu"))
        cls.MODEL.add(MaxPooling2D(pool_size=(2,2)))
        cls.MODEL.add(Dropout(0.5))
        cls.MODEL.add(Conv2D(64,cls.KERNEL_SIZE,input_shape=cls.INPUT_SHAPE,activation="relu"))
        cls.MODEL.add(MaxPooling2D(pool_size=(2,2)))
        cls.MODEL.add(Dropout(0.25))
        cls.MODEL.add(Flatten())
        cls.MODEL.add(Dense(256,activation="relu"))
        cls.MODEL.add(Dense(cls.CLASSESS_NUMBER,activation="softmax"))
            
        cls.MODEL.compile(loss="sparse_softmax_cross_entropy",optimizer="adam",metrics=["accuracy"])
        
       