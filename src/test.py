import tensorflow as tf


# print("Num GPUs Available: ", tf.test.is_built_with_cuda())
# print('\n')
print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
# print('\n')
tf.config.experimental.list_physical_devices()
# print('\n')
# tf.config.list_physical_devices()
# print('\n')
# tf.test.gpu_device_name()
# print('\n')
# print("tensor flow version   ", tf.__version__)
