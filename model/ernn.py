from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(".."))
from util.momentum_reader import MomentumReader
import tensorflow as tf
import numpy
import time
import datetime
import dlnn_util

class ERNN():
    def __init__(self):
        self.run_params = []


    # def build_model(self):

    def run_model(self, params):
        """
        Builds the neural network variables required and fills critical object
        variables
        """
        if type(params) != type(dlnn_util.DeepLinearNNParams()):
            print("Please use the right parameter structure for dlnn, which is DeepLinearNNParams()")
            return None
        print(params)
        af = params.get_activation_function()
        loss_func = params.get_loss_function()
        optimizer = params.get_optimizer()
        layers = params.layers
        
        self.run_params.append(params)
        db = MomentumReader(
            classification=params.classify, 
            test_precentage=params.test_pct, 
            backtest_precentage=params.backtest_pct, 
            hot_vector=True)

        input_size = db.get_input_size()
        output_size = db.get_output_size()
        data_size = db.get_train_data_size()
        print("Input size is %s, Output size is %s, Trainning Data Size is %s"%(input_size,output_size,data_size))

        max_train_steps = int(data_size*params.epoch/params.batch_size)
        weights = []
        bias = []

        # input layer weights
        wi = tf.Variable(tf.truncated_normal([input_size, layers[0]]),name="input_layer_variable")
        # construct recurrent layer
        wc = tf.Variable(tf.truncated_normal(hidden_layer,hidden_layer),name="recurrent_layer_variable")
        # construct the hidden layer
        wh = tf.Variable(tf.truncated_normal(hidden_layer,output_size), name="hidden_layer_variable")
        # initialize hidden layer value
        vh = tf.Variable(tf.zeros([None,hidden_layer]),trainable=False)
        # output label
        y_ = tf.placeholder(tf.float32, shape=[None, output_size])
        # input
        input_data = list()
        for _ in range(time_steps + 1):
            input_data.append(tf.placeholder(tf.float32, shape=[batch_size,input_size]))
        # output
        outputs = []
        for i in input_data:
            vh = tf.sigmoid(tf.matmul(,w) + tf.matmul(v,vh))
            vo = tf.matmul(vh,v)
            outputs.append(vo)

        # loss function
        loss = tf.nn.l2_loss(tf.concat(0, outputs), y_)
        train_step =  tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        for i in range(max_train_steps):
            batch_xs, batch_ys = db.next_batch(params.batch_size)
            for j in range(time_steps):
                feed_dict = {x:batch_xs[j], y_:batch_ys[j]}


        for id, layer in enumerate(layers):
            if id == 0:
                continue
            W = tf.Variable(tf.truncated_normal([layers[id-1], layers[id]]))
            W = tf.nn.dropout(W, 0.5)
            b = tf.Variable(tf.truncated_normal([layers[id]]))
            weights.append(W)
            bias.append(b)

        weights.append(tf.Variable(tf.truncated_normal([layers[-1], output_size])))
        bias.append(tf.Variable(tf.truncated_normal([output_size])))

        for id in range(len(weights)):
            if id == 0:
                vh = af(tf.matmul(x, weights[id]) + bias[id])
            elif id == len(weights)-1:
                y = af(tf.matmul(vh, weights[id]) + bias[id])
            else:
                vh = af(tf.matmul(vh, weights[id]) + bias[id])

        # TBD: whether we want to normalize the output from 0 to 1
        # y = (y - tf.reduce_min(y) + 1e-10)/(tf.reduce_max(y)-tf.reduce_min(y))
        # y_ is the target

        loss = loss_func(y,y_)

        global_step = tf.Variable(0, dtype=tf.float32)
        # add regularizer
        regularizers = tf.reduce_mean([tf.nn.l2_loss(w) for w in weights] + [tf.nn.l2_loss(b) for b in bias]) * 1e-6
        rloss = loss + regularizers

        # lr_decay = True
        # # 10 means every 10% decrease once
        # decay_step = max_train_steps/10
        # if lr_decay:
        #   learning_rate = tf.train.exponential_decay( 
        #         params.base_learning_rate,        # Base learning rate.
        #         global_step,                    # Current index into the dataset.
        #         decay_step,                     # Decay steps.
        #         0.96,                           # Decay rate.
        #         staircase=True)
        # else:  
        #   learning_rate = tf.constant(params.base_learning_rate)

        train_step = optimizer.minimize(rloss, global_step=global_step)

        # if want to show what devices are using, turn log_device_placement True
        config = tf.ConfigProto(log_device_placement=False)
        sess = tf.InteractiveSession(config=config)

        evaluation_frequency = 100
        # Training
        total_start_time = time.time()
        loop_strat_time = time.time()
        tf.initialize_all_variables().run()
        for step in xrange(max_train_steps):
            batch_xs, batch_ys = db.next_batch(params.batch_size)
            sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})
            if step%evaluation_frequency == 1:
                l,gs,output,rlo = sess.run([loss,global_step,y,rloss], feed_dict={x: batch_xs, y_: batch_ys})
                now = time.time()
                duration = now - total_start_time
                loop_duration = now - loop_strat_time
                operation_precentage = step/max_train_steps*100
                loop_strat_time = now
                average_loop_time = duration / step
                time_left = average_loop_time * (max_train_steps-step)
                print("loss: % 2.3f, rloss % 2.3f,"\
                    "operation precentage:% 2.2f%% loop time used:% 3.3f,"\
                    " total time used:% 3.3f, global step: %d, maximum step: %d"
                    %(l,rlo,operation_precentage,loop_duration,duration,gs,max_train_steps))
                print("Estimated time left is: % .2f mins"%(time_left/60))
                print("output sample is ",output[0])
                print("Target Output is ", batch_ys[0])
        total_time_used = time.time()-total_start_time
        print("Total Time Used For Trainning: %f"%total_time_used)
        # log the final loss
        final_loss = l

        # Test trained model
        test_input, test_label = db.get_test_data()
        if db.classification:
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            evaluation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            output_accuracy = sess.run(evaluation, feed_dict={x: test_input, y_:test_label})
            print("Accuracy is %f"%output_accuracy)
        else:
            evaluation = tf.reduce_mean(tf.abs(y - y_))
            output_loss = sess.run(evaluation, feed_dict={x: test_input, y_:test_label})
            print("The mean L1 loss of test data is %f"%output_loss)

        # Backtest
        backtest_input, backtest_output, backtest_value = db.get_backtest_data()
        acc_return = 0
        amrs = [] # acumulated montly return
        for date in range(len(backtest_input)):
            input = backtest_input[date]
            num_stock_to_buy = int(params.buying_pct*len(input))
            if num_stock_to_buy < 1:
                num_stock_to_buy = 1
            output = sess.run(y, feed_dict={x:input})
            bvalue = backtest_value[date]
            # print("output of backtest: ", output)
            if db.classification:
                class1 = output[:,0]
                # argsort the class1
                sort_ids = numpy.argsort(class1)
                acc_return += numpy.sum(bvalue[sort_ids[-num_stock_to_buy:]])/num_stock_to_buy
                amrs.append(acc_return)
            else:
                sort_ids = numpy.argsort(output)
                acc_return += numpy.sum(bvalue[sort_ids[0:num_stock_to_buy]])/num_stock_to_buy
                amrs.append(acc_return)
            print("Accumulated return at month %d is % 3.3f%%"%(date, acc_return))
        result = {}
        result["Total Time"] = total_time_used
        result["Accuracy"] = output_accuracy
        result["AMR"] = amrs
        result["Loss"] = final_loss
        return result

if __name__ == '__main__':
    #
    params = DeepLinearNNParams() #Param object
    layers = [40, 4, 50]
    params.epoch = 100
    params.batch_size = 100
    params.learning_rate = 0.1
    params.optimizer = 'gd'
    params.classify = True
    params.test_pct = 0.3
    params.backtest_pct = 0.1
    params.buying_pct = 0.01
    params.activation = 'sigmoid'
    params.loss_func = 'softmax_cross_entropy_with_logits'

    dlnn = DeepLinearNN()
    results = dlnn.run_model(params)

    #just pass dlnn and you can access all results.
    dlnn_util.save(params, results, './results/')
