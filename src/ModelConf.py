from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
import numpy as np
import pickle
import os
import gc
import matplotlib.pyplot as plt
import globalVariable

import globalVariable as gv


class ModelConf:

    def __init__(self, dataset_name, dataset, classes_number, kernel_size, input_shape,poisoning = False, blockchain = False):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.classes_number = classes_number
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.poisoning = poisoning
        self.blockchain = blockchain
        # load data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.dataset.load_data()
        self.set_dataset()

        self.generate_dataset_client_partition()
        
        del (self.x_train, self.y_train)
        gc.collect()

        # # test code.....print image with relative label
        # for itemIndex in range(len(self.partitions[0][1])):
        #     print(f'item shape  {self.partitions[0][2][itemIndex].shape}')
        #     print(type(self.partitions[1][2][itemIndex]))
        #     plt.imshow(self.partitions[0][2][itemIndex].reshape([32, 32, 3]), cmap='gray')
        #     plt.xlabel(f"{self.partitions[1][2][itemIndex]}")
        #     plt.show()

    def set_dataset(self):
        # normalizing e label encodig
        self.x_train = self.x_train / 255
        self.x_test = self.x_test / 255
        self.y_train = to_categorical(self.y_train, self.classes_number)
        self.y_test = to_categorical(self.y_test, self.classes_number)
        
        if self.dataset_name == 'mnist':
            self.x_train = self.x_train.reshape(self.x_train.shape[0],self.x_train.shape[1], self.x_train.shape[2],1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0],self.x_test.shape[1], self.x_test.shape[2],1)

    def generate_dataset_client_partition(self):
       
        dataset_partition_dir = f"dataset/{self.dataset_name}_partitions"
        if not os.path.exists(dataset_partition_dir):
            os.makedirs(dataset_partition_dir)   
            
        x_train_splitted = np.array_split(self.x_train, globalVariable.CLIENTS_NUM) 
        y_train_splitted = np.array_split(self.y_train, globalVariable.CLIENTS_NUM)

        for i in range(globalVariable.CLIENTS_NUM):
            with open(os.path.join(dataset_partition_dir, f"x_train_partition_of_{i}.pickle"), "wb") as f:
                pickle.dump(x_train_splitted[i],f)
                print(x_train_splitted[i].shape)
            with open(os.path.join(dataset_partition_dir, f"y_train_partition_of_{i}.pickle"), "wb") as f:
                pickle.dump(y_train_splitted[i],f)
                        
        del x_train_splitted
        del y_train_splitted
        gc.collect()

    def get_client_training_partitions_of(self, client_partition_index):
        with open(os.path.join(f"dataset/{self.dataset_name}_partitions", f"x_train_partition_of_{client_partition_index}.pickle"), "rb") as f:
            x_train = pickle.load(f)
            
        with open(os.path.join(f"dataset/{self.dataset_name}_partitions", f"y_train_partition_of_{client_partition_index}.pickle"), "rb") as f:
            y_train = pickle.load(f)
            
        return (x_train, y_train)

    def get_test_data(self):
        return self.x_test, self.y_test

    def get_model(self):
        # build the model
        model = Sequential()
        
        #initial arch
        if(self.dataset_name == 'mnist'):
            model.add(Conv2D(64, self.kernel_size, input_shape=self.input_shape, activation="relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.5))
            model.add(Conv2D(64, self.kernel_size, input_shape=self.input_shape, activation="relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(256, activation="relu"))
            model.add(Dense(self.classes_number, activation="softmax"))
        else:
            model.add(Conv2D(filters=32, kernel_size=self.kernel_size, input_shape=self.input_shape, activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Conv2D(filters=64, kernel_size=self.kernel_size, input_shape=self.input_shape, activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Conv2D(filters=128, kernel_size=self.kernel_size, input_shape=self.input_shape, activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.25))
            model.add(Dense(self.classes_number, activation='softmax'))
        
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model
