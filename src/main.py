from model import Model as m
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import cifar100
""" 
# MnistModel
# 
"""
mnistModel = m()
mnistModel.initData(mnist,10)
mnistModel.buildModel((3,3), (28,28,1))
mnistModel.trainModel(128,10)
mnistModel.plotChart("mnistAccuracyChart", "mnistLossChart")
print("mnist trained")

""" 
# Cifar 10 model
# 
"""
cifar10Model = m()
cifar10Model.initData(cifar10,10)
cifar10Model.buildModel((4,4), (32,32,3))
cifar10Model.trainModel(128,10)
cifar10Model.plotChart("cifar10AccuracyChart", "cifar10LossChart")
print("cifar 10 trained")


""" 
# Cifar 100 model
# 
"""
cifar100Model = m()
cifar100Model.initData(cifar100,100)
cifar100Model.buildModel((4,4), (32,32,3))
cifar100Model.trainModel(128,10)
cifar100Model.plotChart("cifar100AccuracyChart", "cifar100LossChart")
print("cifar100 trained")
