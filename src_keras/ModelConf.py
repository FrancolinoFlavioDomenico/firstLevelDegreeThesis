import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Conv2D, Dense, Flatten, Dropout
# from keras.layers import MaxPooling2D
# from keras.layers import MaxPool2D
# from keras.layers import BatchNormalization
# from keras.utils import to_categorical
# from keras.applications.resnet import ResNet50, preprocess_input
# from keras.layers import GlobalAveragePooling2D, Input, UpSampling2D
# from keras import Model
import numpy as np
import pickle
import os
import gc
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
        if self.dataset_name == 'mnist':
            self.x_train = self.x_train / 255
            self.x_test = self.x_test / 255
            self.x_train = self.x_train.reshape(self.x_train.shape[0],self.x_train.shape[1], self.x_train.shape[2],1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0],self.x_test.shape[1], self.x_test.shape[2],1)
        else:
            self.x_train = tf.keras.applications.resnet50.preprocess_input(self.x_train)
            self.x_test = tf.keras.applications.resnet50.preprocess_input(self.x_test)

        self.y_train = tf.keras.utils.to_categorical(self.y_train, self.classes_number)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, self.classes_number)

    def generate_dataset_client_partition(self):
       
        dataset_partition_dir = f"dataset/{self.dataset_name}_partitions"
        if not os.path.exists(dataset_partition_dir):
            os.makedirs(dataset_partition_dir)   
            
        x_train_splitted = np.array_split(self.x_train, gv.CLIENTS_NUM) 
        y_train_splitted = np.array_split(self.y_train, gv.CLIENTS_NUM)

        for i in range(gv.CLIENTS_NUM):
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
        if(self.dataset_name == 'mnist'):
            model = self.get_mnist_arch()
        else:
            model = self.get_cifar_arch()
            
        model.summary()
        return model
    
    def get_cifar_arch(self):
        # model = Sequential()
        # model.add(Conv2D(filters=32, kernel_size=self.kernel_size, input_shape=self.input_shape, activation='relu', padding='same'))
        # model.add(BatchNormalization())
        # model.add(MaxPool2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        # model.add(Conv2D(filters=64, kernel_size=self.kernel_size, input_shape=self.input_shape, activation='relu', padding='same'))
        # model.add(BatchNormalization())
        # model.add(MaxPool2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        # model.add(Conv2D(filters=128, kernel_size=self.kernel_size, input_shape=self.input_shape, activation='relu', padding='same'))
        # model.add(BatchNormalization())
        # model.add(MaxPool2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        # model.add(Flatten())
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.25))
        # model.add(Dense(self.classes_number, activation='softmax'))
        model = self.define_compile_model()
        return model
        
    # Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
    # Input size is 224 x 224.
    def feature_extractor(self,inputs):

        feature_extractor = tf.keras.applications.resnet50.ResNet50(input_shape=(224, 224, 3),
                                                    include_top=False,
                                                    weights='imagenet')(inputs)
        
        feature_extractor.trainable = False
        # for layer in feature_extractor.layers:
        #     # if isinstance(layer, BatchNormalization):
        #     #     layer.trainable = True
        #     # else:
        #         layer.trainable = False
        return feature_extractor


    # Defines final dense layers and subsequent softmax layer for classification.
    def classifier(self,inputs):
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dense(self.classes_number, activation="softmax", name="classification")(x)
        return x

    # Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
    # Connect the feature extraction and "classifier" layers to build the model.
    def final_model(self,inputs):

        resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)

        resnet_feature_extractor = self.feature_extractor(resize)
        classification_output = self.classifier(resnet_feature_extractor)

        return classification_output

    # Define the model and compile it. 
    # Use Stochastic Gradient Descent as the optimizer.
    # Use Sparse Categorical CrossEntropy as the loss function.
    def define_compile_model(self):
        inputs = tf.keras.layers.Input(shape=(32,32,3))
        
        classification_output = self.final_model(inputs) 
        model = tf.keras.Model(inputs=inputs, outputs = classification_output)
        
        model.compile(optimizer='SGD', 
                        loss='categorical_crossentropy',
                        metrics = ['accuracy'])
        
        return model
    
    
    def get_mnist_arch(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(64, self.kernel_size, input_shape=self.input_shape, activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Conv2D(64, self.kernel_size, input_shape=self.input_shape, activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation="relu"))
        model.add(tf.keras.layers.Dense(self.classes_number, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model
