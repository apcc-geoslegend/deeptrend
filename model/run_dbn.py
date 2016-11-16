# this file runs the deep belief network from yadlt
# need to install yadlt first
# sudo pip install yadlt
import sys
import os
sys.path.insert(0, os.path.abspath(".."))
from util.momentum_reader import MomentumReader
import tensorflow as tf
import numpy
import time
# import config

from yadlt.models.rbm_models import dbn
from yadlt.utils import datasets, utilities

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_string('dataset', 'mnist', 'Which dataset to use. ["mnist", "cifar10", "custom"]')
flags.DEFINE_string('train_dataset', '', 'Path to train set .npy file.')
flags.DEFINE_string('train_labels', '', 'Path to train labels .npy file.')
flags.DEFINE_string('valid_dataset', '', 'Path to valid set .npy file.')
flags.DEFINE_string('valid_labels', '', 'Path to valid labels .npy file.')
flags.DEFINE_string('test_dataset', '', 'Path to test set .npy file.')
flags.DEFINE_string('test_labels', '', 'Path to test labels .npy file.')
flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
flags.DEFINE_string('model_name', 'dbn', 'Name of the model.')
flags.DEFINE_string('save_predictions', '', 'Path to a .npy file to save predictions of the model.')
flags.DEFINE_string('save_layers_output_test', '', 'Path to a .npy file to save test set output from all the layers of the model.')
flags.DEFINE_string('save_layers_output_train', '', 'Path to a .npy file to save train set output from all the layers of the model.')
flags.DEFINE_boolean('do_pretrain', True, 'Whether or not pretrain the network.')
flags.DEFINE_boolean('do_train', True, 'Whether or not train the network.')
flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')
flags.DEFINE_integer('verbose', 1, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_string('main_dir', 'dbn/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_string('models_dir', 'model/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_string('data_dir', 'data/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_string('summary_dir', 'summery/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')

# RBMs layers specific parameters
flags.DEFINE_string('encoder_layers', '40,4,', 'Comma-separated values for the layers in the sdae.')
flags.DEFINE_string('decoder_layers', '50,', 'Comma-separated values for the layers in the sdae.')
flags.DEFINE_boolean('rbm_gauss_visible', False, 'Whether to use Gaussian units for the visible layer.')
flags.DEFINE_float('rbm_stddev', 0.1, 'Standard deviation for Gaussian visible units.')
flags.DEFINE_string('rbm_learning_rate', '0.01,', 'Initial learning rate.')
flags.DEFINE_string('rbm_num_epochs', '10,', 'Number of epochs.')
flags.DEFINE_string('rbm_batch_size', '10,', 'Size of each mini-batch.')
flags.DEFINE_string('rbm_gibbs_k', '1,', 'Gibbs sampling steps.')

# Supervised fine tuning parameters
flags.DEFINE_string('finetune_act_func', 'sigmoid', 'Activation function.')
flags.DEFINE_float('finetune_learning_rate', 0.01, 'Learning rate.')
flags.DEFINE_float('finetune_momentum', 0.7, 'Momentum parameter.')
flags.DEFINE_integer('finetune_num_epochs', 10, 'Number of epochs.')
flags.DEFINE_integer('finetune_batch_size', 10, 'Size of each mini-batch.')
flags.DEFINE_string('finetune_opt', 'gradient_descent', '["gradient_descent", "ada_grad", "momentum", "adam"]')
flags.DEFINE_string('finetune_loss_func', 'softmax_cross_entropy', 'Loss function. ["mean_squared", "softmax_cross_entropy"]')
flags.DEFINE_float('finetune_dropout', 1, 'Dropout parameter.')

# Conversion of Autoencoder layers parameters from string to their specific type
encoder_layers = utilities.flag_to_list(FLAGS.encoder_layers, 'int')
decoder_layers = utilities.flag_to_list(FLAGS.decoder_layers, 'int')
rbm_learning_rate = utilities.flag_to_list(FLAGS.rbm_learning_rate, 'float')
rbm_num_epochs = utilities.flag_to_list(FLAGS.rbm_num_epochs, 'int')
rbm_batch_size = utilities.flag_to_list(FLAGS.rbm_batch_size, 'int')
rbm_gibbs_k = utilities.flag_to_list(FLAGS.rbm_gibbs_k, 'int')

# Parameters validation
assert FLAGS.dataset in ['mnist', 'cifar10', 'custom']
assert FLAGS.finetune_act_func in ['sigmoid', 'tanh', 'relu']
assert FLAGS.finetune_loss_func in ['mean_squared', 'softmax_cross_entropy']
assert len(encoder_layers) > 0
assert len(decoder_layers) > 0

if __name__ == '__main__':

    utilities.random_seed_np_tf(FLAGS.seed)
    FLAGS.do_pretrain = True
    FLAGS.do_train = True
    FLAGS.encoder_layers = '40,4,'
    FLAGS.decoder_layers = '50,'
    FLAGS.finetune_act_func = 'tanh'
    FLAGS.finetune_opt = 'adam'
    FLAGS.restore_previous_model = False
    FLAGS.finetune_num_epochs = 1000
    FLAGS.finetune_batch_size = 1000
    FLAGS.rbm_num_epochs = '800,100,'
    FLAGS.rbm_batch_size = '500,'
    FLAGS.finetune_learning_rate = 0.001


    mr = MomentumReader(classification=True, test_precentage=0.3, validation_precentage=0.1, hot_vector=True)
    trX, trY = mr.get_all_train_data()
    vlX, vlY = mr.get_validation_data()
    teX, teY = mr.get_test_data()

    start_time = time.time()
    # Create the object
    finetune_act_func = utilities.str2actfunc(FLAGS.finetune_act_func)

    param = dbn.DBNParam()
    param.parse_flag(FLAGS)
    srbm = dbn.DeepBeliefNetwork(param)

    # Fit the model (unsupervised pretraining)
    if FLAGS.do_pretrain:
        srbm.pretrain(trX, vlX)

    if FLAGS.do_train:
        # finetuning
        print('Start deep belief net finetuning...')
        srbm.fit(trX, trY, vlX, vlY, restore_previous_model=FLAGS.restore_previous_model)

    # Test the model
    print('Test set accuracy: {}'.format(srbm.compute_accuracy(teX, teY)))

    btX, btY, btV = mr.get_backtest_data()
    acc_return = 0
    amrs = [] # acumulated montly return
    for date in range(len(btX)):
        input = btX[date]
        num_stock_to_buy = int(0.1*len(input))
        if num_stock_to_buy < 1:
            num_stock_to_buy = 1
        output = srbm.predict_prob(input)
        bvalue = btV[date]
        # print("output of backtest: ", output)
        class1 = output[:,0]
        # argsort the class1
        sort_ids = numpy.argsort(class1)
        acc_return += numpy.sum(bvalue[sort_ids[-num_stock_to_buy:]])/num_stock_to_buy
        amrs.append(acc_return)
        print("Accumulated return at month %d is % 3.3f%%"%(date, acc_return))

    total_time_used = (time.time() - start_time)/60
    print("toltal time used: %f mins"%total_time_used)

    # result = {}
    # result["Total Time"] = total_time_used
    # result["Accuracy"] = output_accuracy
    # result["AMR"] = amrs
    # result["Loss"] = final_loss
    # return result

    # # Save the predictions of the model
    # if FLAGS.save_predictions:
    #     print('Saving the predictions for the test set...')
    #     np.save(FLAGS.save_predictions, srbm.predict(teX))


    # def save_layers_output(which_set):
    #     if which_set == 'train':
    #         trout = srbm.get_layers_output(trX)
    #         for i, o in enumerate(trout):
    #             np.save(FLAGS.save_layers_output_train + '-layer-' + str(i + 1) + '-train', o)

    #     elif which_set == 'test':
    #         teout = srbm.get_layers_output(teX)
    #         for i, o in enumerate(teout):
    #             np.save(FLAGS.save_layers_output_test + '-layer-' + str(i + 1) + '-test', o)


    # # Save output from each layer of the model
    # if FLAGS.save_layers_output_test:
    #     print('Saving the output of each layer for the test set')
    #     save_layers_output('test')

    # # Save output from each layer of the model
    # if FLAGS.save_layers_output_train:
    #     print('Saving the output of each layer for the train set')
    #     save_layers_output('train')
