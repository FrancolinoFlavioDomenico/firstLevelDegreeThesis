from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.layers import MaxPooling2D

from keras.utils import to_categorical

import matplotlib.pyplot as plt

from src import CLIENTS_NUM

import numpy as np


class ModelConf:

    def __init__(self, dataset_name, dataset, classes_number, kernel_size, input_shape):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.classes_number = classes_number
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        # load data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.dataset.load_data()
        self.set_dataset()

        # print(self.x_train.shape)
        # print(self.y_train.shape)
        # print(f"generated indec {self.index_to_split_dataset}")
        # print(f"generated indec {self.index_to_split_dataset}")
        # print(f"generated indec somm {int(np.sum(self.index_to_split_dataset))}")
        #self.client_partitition = self.generate_dataset_client_partition()
        # self.
        print(f'self.x_train shape: {self.x_train.shape}')
        # plt.imshow(self.x_train[0].reshape([32,32,3]), cmap='gray')
        # plt.axis('off')
        # plt.show()
        # print(f"xtrian[0]: {self.x_train[0]}")
        # print(f"ytrain: {self.y_train[0]}")

        self.partitions = self.generate_dataset_client_partition()
        print(self.partitions[0][1].shape)
        print(self.partitions[1][1].shape)
       #  print(self.partitions[0].shape)
       # # (self.x_train, self.y_train) = self.client_partitions
       #  plt.imshow(self.partitions[0][0].reshape([32,32,3]), cmap='gray')
       #  plt.axis('off')
       #  plt.show()
       #  print(f"xtrian[0]: {self.partitions[0][0]}")
       #  print(f"ytrain:{self.partitions[1][0]}")

    # print(self.x_train[0].shape)


    def set_dataset(self):
        # normalizing e label encodig
        self.x_train = self.x_train / 255
        self.x_test = self.x_test / 255
        self.y_train = to_categorical(self.y_train, self.classes_number)
        self.y_test = to_categorical(self.y_test, self.classes_number)

    def generate_dataset_client_partition(self):
        # # generete partitions end index
        # partions_index_end = np.random.rand(CLIENTS_NUM)
        # partions_index_end = partions_index_end * self.x_train.shape[0] / np.sum(partions_index_end)
        # #partions_index_end = np.sort(partions_index_end)
        # np.random.shuffle(partions_index_end)
        #
        # x_train_partitions = np.array(np.array_split(self.x_train, partions_index_end.astype(int)))
        # y_train_partitions = np.array(np.array_split(self.y_train, partions_index_end.astype(int)))
        # print("x_train_partitions: ", x_train_partitions.shape)
        # print("y_train_partitions: ", y_train_partitions.shape)
        # return x_train_partitions, y_train_partitions

        x_train_partitions = np.array(np.array_split(self.x_train, CLIENTS_NUM))
        y_train_partitions = np.array(np.array_split(self.y_train, CLIENTS_NUM))
        return x_train_partitions, y_train_partitions

    def get_client_training_partitions_of(self, client_partition_index):
        return self.partitions[0][client_partition_index], self.partitions[1][client_partition_index]

    def get_test_data(self):
        return self.x_test, self.y_test

    def get_model(self):
        # build the model
        model = Sequential()
        model.add(Conv2D(64, self.kernel_size, input_shape=self.input_shape, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, self.kernel_size, input_shape=self.input_shape, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dense(self.classes_number, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model
