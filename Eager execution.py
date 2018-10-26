import tensorflow as tf
import datetime
tf.enable_eager_execution()

# ###Tensors
# print(tf.add(1,2))
# print(tf.add([1,2],[3,4]))
# print(tf.square(5)) #平方
# print(tf.reduce_sum([1,2,3])) #求和
# print(tf.encode_base64("hello world"))
#
# print(tf.square(2)+tf.square(3))
#
# x=tf.matmul([[1]],[[2,3]])
# print(x)
# print(x.shape)
# print(x.dtype)
#
#
# import numpy as np #tensor 与 ndarray互转
# ndarray=np.ones([3,3])
# tensor=tf.multiply(ndarray,42)
# print(tensor)
#
# n1=np.add(tensor,1)
#
# n2=tensor.numpy()
# print(n2)
# print(type(n2))

x = tf.random_uniform([3, 3])

# print("Is there a GPU available: "),
# print(tf.test.is_gpu_available())
#
# print("Is the Tensor on GPU #0:  "),
# print(x.device.endswith('GPU:0'))

def time_matmul(x):
    tf.matmul(x, x)

# Force execution on CPU
starttime = datetime.datetime.now()
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random_uniform([1000, 1000])
  assert x.device.endswith("CPU:0")
  time_matmul(x)
endtime = datetime.datetime.now()
print (endtime - starttime)

starttime = datetime.datetime.now()
# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)
endtime = datetime.datetime.now()
print (endtime - starttime)

