from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.layers import MaxPooling2D

from keras.utils import to_categorical

from keras.datasets import cifar10


class Model:
    CLIENTS_NUM = 10
    ROUNDS_NUM = 5
    # MODEL = Sequential()
    CLASSES_NUMBER = 10
    DATASET = cifar10
    KERNEL_SIZE = (4, 4)
    INPUT_SHAPE = (32, 32, 3)

    X_TRAIN = None
    Y_TRAIN = None
    X_TEST = None
    Y_TEST = None

    @classmethod
    def set_data(cls, dataset=None, classes_number=None, kernel_size=None, input_shape=None):
        if dataset is not None:
            Model.DATASET = dataset
            Model.CLASSES_NUMBER = classes_number
            Model.KERNEL_SIZE = kernel_size
            Model.INPUT_SHAPE = input_shape
            Model.DATASET = dataset
        # load data
        (Model.X_TRAIN, Model.Y_TRAIN), (Model.X_TEST, Model.Y_TEST) = Model.DATASET.load_data()
        # normalizing e label encodig
        Model.X_TRAIN = Model.X_TRAIN / 255
        Model.X_TEST = Model.X_TEST / 255
        Model.Y_TRAIN = to_categorical(Model.Y_TRAIN, classes_number)
        Model.Y_TEST = to_categorical(Model.Y_TEST, classes_number)

    @classmethod
    def get_model(cls):
        # build the model
        model = Sequential()
        model.add(Conv2D(64, Model.KERNEL_SIZE, input_shape=Model.INPUT_SHAPE, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, Model.KERNEL_SIZE, input_shape=Model.INPUT_SHAPE, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dense(Model.CLASSES_NUMBER, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model
