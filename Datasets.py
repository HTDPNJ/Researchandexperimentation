import tensorflow as tf
tf.enable_eager_execution()
de_tensors=tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6]) #"""创建一个`数据集'，其元素是给定张量的切片。"""
print(de_tensors)
import tempfile
_,filename=tempfile.mkstemp()
print(_)
print(filename)
with open(filename,'w') as f:
    f.write("""
    Line 1
    """)
ds_file = tf.data.TextLineDataset(filename)

ds_tensors = de_tensors.map(tf.square).shuffle(2).batch(2)

ds_file = ds_file.batch(2)

print('Elements of ds_tensors:')
for x in ds_tensors:
  print(x)

print('\nElements in ds_file:')
for x in ds_file:
  print(x)