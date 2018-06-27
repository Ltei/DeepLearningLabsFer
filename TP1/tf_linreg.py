
import tensorflow as tf


#
# TP1 - Task 3
#

## 1. definition of the computation graph
# data and parameters
X  = tf.placeholder(tf.float32, [None])
Y_ = tf.placeholder(tf.float32, [None])
a = tf.Variable(0.0)
b = tf.Variable(0.0)

# affine regression model
Y = a * X + b

# quadratic loss
loss = (Y-Y_)**2

# optimization by gradient descent
trainer = tf.train.GradientDescentOptimizer(0.1)
gradients = trainer.compute_gradients(loss, [a, b])
train_op = trainer.apply_gradients(gradients)

g = [g[0] for g in gradients]
g = tf.Print(g, [g], 'Gradients: ')

## 2. parameter initialization
sess = tf.Session()
sess.run(tf.global_variables_initializer())

## 3. training
# let the games start!
for i in range(100):
    val_loss, _, val_a,val_b = sess.run([loss, train_op, a,b], feed_dict={X: [1,2], Y_: [3,5]})
    print("Iteration = ",i,", Loss = ",val_loss,", a = ",val_a,", b = ",val_b)
