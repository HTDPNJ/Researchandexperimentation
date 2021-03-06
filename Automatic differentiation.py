import tensorflow as tf
tf.enable_eager_execution()

tre=tf.contrib.eager

# from math import pi
# def f(x):
#     return tf.square(tf.sin(x))
# assert f(pi/2).numpy()==1.0
#
# #grag_f将返回f的导数列表
# #关于其论点。 由于f（）只有一个参数，
# #grag_f将返回包含单个元素的列表。
# grad_f=tre.gradients_function(f)  #求导
# assert tf.abs(grad_f(pi/2)[0]).numpy()<1e-7
#
# def f(x):
#     return tf.square(tf.sin(x))
# def grad(f):
#     return lambda x: tre.gradients_function(f)(x)[0]
#
# x=tf.lin_space(-2*pi,2*pi,100)
#
# import matplotlib.pyplot as plt
#
# plt.plot(x,f(x),label="f")
# plt.plot(x,grad(f)(x),label="first derivative")
# plt.plot(x,grad(grad(f))(x),label="second derivative")
# plt.plot(x,grad(grad(grad(f)))(x),label="third derivative")
# plt.legend()
# plt.show()
#
# def f(x, y):
#   output = 1
#   # Must use range(int(y)) instead of range(y) in Python 3 when
#   # using TensorFlow 1.10 and earlier. Can use range(y) in 1.11+
#   for i in range(int(y)):
#     output = tf.multiply(output, x)
#   return output
#
# def g(x, y):
#   # Return the gradient of `f` with respect to it's first parameter
#   return tre.gradients_function(f)(x, y)[0]
#
# assert f(3.0, 2).numpy() == 9.0   # f(x, 2) is essentially x * x
# assert g(3.0, 2).numpy() == 6.0   # And its gradient will be 2 * x
# assert f(4.0, 3).numpy() == 64.0  # f(x, 3) is essentially x * x * x
# assert g(4.0, 3).numpy() == 48.0  # And its gradient will be 3 * x * x



# ############
x=tf.ones((2,2))

with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y=tf.reduce_sum(x)
    z=tf.multiply(y,y)
dz_dy = t.gradient(z, y) #计算z函数在y点的导数
assert dz_dy.numpy() == 8.0

x = tf.constant(1.0)  # Convert the Python 1.0 to a Tensor object

with tf.GradientTape() as t:
  with tf.GradientTape() as t2:
    t2.watch(x)
    y = x * x * x
  # Compute the gradient inside the 't' context manager
  # which means the gradient computation is differentiable as well.
  dy_dx = t2.gradient(y, x)
d2y_dx2 = t.gradient(dy_dx, x)

assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0











