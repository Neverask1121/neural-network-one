import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU devices:", tf.config.list_physical_devices("GPU"))
print("Is GPU available:", tf.test.is_gpu_available())
print("Build info:", tf.sysconfig.get_build_info())