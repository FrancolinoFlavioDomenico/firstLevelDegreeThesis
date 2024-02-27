from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.layers import MaxPooling2D

from keras.utils import to_categorical


class ModelConf:

    def __init__(self, dataset_name, dataset, classes_number, kernel_size, input_shape):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.classes_number = classes_number
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        # load data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.dataset.load_data()
        # normalizing e label encodig
        self.x_train = self.x_train / 255
        self.x_test = self.x_test / 255
        self.y_train = to_categorical(self.y_train, self.classes_number)
        self.y_test = to_categorical(self.y_test, self.classes_number)

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
