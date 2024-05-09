import tensorflow as tf

number = tf.Variable(300, tf.int16)
string =tf.Variable("this is a string", tf.string)
floating = tf.Variable(3.4566, tf.float64)

#RANKING OF TENSORS
rank1_tensor = tf.Variable(["test", "okay"], tf.string)
rank2_tensor = tf.Variable(["data1","test"], ["data2"], tf.string)

#determine the rank
tf.rank(rank2_tensor)

#determine the shape
rank2_tensor.shape
print(rank2_tensor.shape)

#changing shape
tensor1 =tf.ones([1,2,3])
tensor2 = tf.reshape(tensor1, [2,1,3])
tensor3 = tf.reshape(tensor2, [3,-1])

print(tensor1)
print(tensor2)
print(tensor3)

#Evaluating Tensors
with tf.session() as sess: #creates a session using default graph
    tensor.eval() #tensor will be the name of your tensor