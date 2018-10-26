import tensorflow as tf
tfe=tf.contrib.eager
tf.enable_eager_execution()

x=tf.zeros([10,10])
x+=2
# print(x)
v=tfe.Variable(1.0)
assert v.numpy()==1.0

v.assign(3.0)
assert v.numpy() == 3.0

# Use `v` in a TensorFlow operation like tf.square() and reassign
v.assign(tf.square(v))
assert v.numpy() == 9.0
s=tf.assign_add(v,5)







