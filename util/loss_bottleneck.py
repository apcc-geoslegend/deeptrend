import tensorflow as tf
import numpy

if __name__ == '__main__':
	x = [0,1]
	y_s = [[1,1],[0.5,0.5],[0,0]]
	x = tf.constant(numpy.array(x),dtype=tf.float64)
	y = tf.placeholder(dtype=tf.float64, shape=[2,])
	losses = []
	losses.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, x)))
	losses.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, x)))
	losses.append(tf.reduce_mean(tf.nn.log_poisson_loss(y,x)))
	losses.append(tf.nn.l2_loss(y - x))

	sess = tf.InteractiveSession()
	minimums = [100 for i in losses]
	for y_ in y_s:
		y_ = numpy.array(y_)
		print(y_)
		for id, loss in enumerate(losses):
			v = sess.run(loss, feed_dict={y:y_})
			if v < minimums[id]:
				minimums[id] = v

	print("Sigmoid Cross Entropy norm is:",minimums[0])
	print("Softmax Cross Entropy norm is:",minimums[1])
	print("Log Poisson Loss:",minimums[2])
	print("L2 loss norm is:",minimums[3])