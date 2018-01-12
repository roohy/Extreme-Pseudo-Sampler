import tensorflow as tf
import numpy as np


class LogisticRegressor(object):

    def __init__(self, learning_rate=1e-3, input_dim=100):
        self.learning_rate = learning_rate
        
        self.inputDim = input_dim
        self.build()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, self.inputDim])
        self.W = tf.Variable(tf.zeros([self.inputDim,1]))
        self.b = tf.Variable(tf.zeros([1]))
        self.y = tf.placeholder(name='y', dtype=tf.float32, shape=[None,1])

        self.y_hat = tf.sigmoid(tf.matmul(self.x,self.W)+self.b)
        

        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        epsilon = 1e-9
        loss = -tf.reduce_sum(
            self.y * tf.log(epsilon+self.y_hat) + (1-self.y) * tf.log(epsilon+ (1-self.y_hat)), 
            axis=1
        )
        self.loss = tf.reduce_mean(loss)

        #self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)
        return

    # Execute the forward and the backward pass
    def run_single_step(self, x,y):
        _, loss = self.sess.run(
            [self.train_op, self.loss],
            feed_dict={self.x: x,self.y:y}
        )

        return loss


    def classifier(self, x):
        y_hat = self.sess.run(self.y_hat, feed_dict={self.x: x})
        labeled = [1 if item > 0.5 else 0 for item in y_hat]
        
        
        return np.array(labeled)

    
    # x -> z
    def predictor(self, x):
        y_hat = self.sess.run(self.y_hat, feed_dict={self.x: x})
        return y_hat