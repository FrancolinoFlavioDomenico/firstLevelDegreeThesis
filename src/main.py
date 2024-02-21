from model import Model as m
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import cifar100

BATCH_SIZE = 130
EPOCHS = 10


""" 
# MnistModel
# 
"""
mnistModel = m()
mnistModel.initData(mnist.load_data(),10)
mnistModel.buildModel((3,3), (28,28,1))
mnistModel.trainModel(BATCH_SIZE,EPOCHS)
mnistModel.plotChart("mnistAccuracyChart", "mnistLossChart")
print("mnist trained")

""" 
# Cifar 10 model
# 
"""
cifar10Model = m()
cifar10Model.initData(cifar10.load_data(),10)
cifar10Model.buildModel((4,4), (32,32,3))
cifar10Model.trainModel(BATCH_SIZE,EPOCHS)
cifar10Model.plotChart("cifar10AccuracyChart", "cifar10LossChart")
print("cifar 10 trained")


""" 
# Cifar 100 model
# 
"""
cifar100Model = m()
cifar100Model.initData(cifar100.load_data(),100)
cifar100Model.buildModel((4,4), (32,32,3))
cifar100Model.trainModel(BATCH_SIZE,EPOCHS)
cifar100Model.plotChart("cifar100AccuracyChart", "cifar100LossChart")
print("cifar100 trained")
