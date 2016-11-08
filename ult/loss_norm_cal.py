import tensorflow as tf
import numpy

if __name__ == '__main__':
	y = tf.constant(numpy.array([0,1]),dtype=tf.float64)
	y_ = tf.constant(numpy.array([0.5,0.5]),dtype=tf.float64)
	sig_ce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, y_))
	soft_ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
	# sparse_sce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
	l2 =tf.nn.l2_loss(y - y_)
	lpl = tf.reduce_mean(tf.nn.log_poisson_loss(y,y_))
	sess = tf.InteractiveSession()
	print("Sigmoid Cross Entropy norm is:",sess.run(sig_ce))
	print("Softmax Cross Entropy norm is:",sess.run(soft_ce))
	# print("Sparse Softmax Cross Entropy norm is:",sess.run(sparse_sce))
	print("L2 loss norm is:",sess.run(l2))
	print("Log Poisson Loss:",sess.run(lpl))