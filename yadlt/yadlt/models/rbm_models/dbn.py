"""Implementation of Deep Belief Network Model using TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf

from yadlt.core.supervised_model import SupervisedModel
from yadlt.models.rbm_models import rbm
from yadlt.utils import utilities


class DBNParam():

    def __init__(self):
        """Constructor.

        :param encoder_layers: list containing the hidden units for each layer
        :param finetune_loss_func: Loss function for the softmax layer.
            string, default ['softmax_cross_entropy', 'mean_squared']
        :param finetune_dropout: dropout parameter
        :param finetune_learning_rate: learning rate for the finetuning.
            float, default 0.001
        :param encode_act_func: activation function for the finetuning phase
        :param finetune_opt: optimizer for the finetuning phase
        :param finetune_num_epochs: Number of epochs for the finetuning.
            int, default 20
        :param finetune_batch_size: Size of each mini-batch for the finetuning.
            int, default 20
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy.
            int, default 0
        :param do_pretrain: True: uses variables from pretraining,
            False: initialize new variables.
        """
        self.encoder_layers = [255]
        self.decoder_layers = []
        self.model_name = 'dbn'
        self.do_pretrain = False
        self.main_dir = 'dbn/'
        self.models_dir = 'models/'
        self.data_dir = 'data/'
        self.summary_dir = 'logs/'
        self.rbm_num_epochs = [10]
        self.rbm_gibbs_k = [1]
        self.rbm_gauss_visible = False
        self.rbm_stddev = 0.1
        self.rbm_batch_size = [10]
        self.dataset = 'mnist'
        self.rbm_learning_rate = [0.01]
        self.encode_act_func = tf.nn.sigmoid
        self.decode_act_func = tf.nn.sigmoid
        self.finetune_dropout = 1
        self.finetune_loss_func = 'softmax_cross_entropy'
        self.finetune_opt = 'gradient_descent'
        self.finetune_learning_rate = 0.001
        self.finetune_num_epochs = 10
        self.finetune_batch_size = 20
        self.verbose = 1
        self.momentum = 0.5

    def parse_flag(self, FLAGS):
        self.models_dir = FLAGS.models_dir
        self.data_dir = FLAGS.data_dir
        self.summary_dir = FLAGS.summary_dir
        self.model_name = FLAGS.model_name
        self.do_pretrain = FLAGS.do_pretrain
        self.decoder_layers = utilities.flag_to_list(FLAGS.decoder_layers,'int')
        self.encoder_layers = utilities.flag_to_list(FLAGS.encoder_layers,'int')
        self.dataset = FLAGS.dataset
        self.main_dir = FLAGS.main_dir
        self.finetune_act_func = FLAGS.finetune_act_func
        self.verbose = FLAGS.verbose
        self.rbm_learning_rate = utilities.flag_to_list(FLAGS.rbm_learning_rate,'float')
        self.rbm_num_epochs = utilities.flag_to_list(FLAGS.rbm_num_epochs,'int')
        self.rbm_gibbs_k = utilities.flag_to_list(FLAGS.rbm_gibbs_k,'int')
        self.rbm_batch_size = utilities.flag_to_list(FLAGS.rbm_batch_size,'int')
        self.rbm_gauss_visible = FLAGS.rbm_gauss_visible
        self.rbm_stddev = FLAGS.rbm_stddev
        self.momentum = FLAGS.momentum
        self.finetune_learning_rate = FLAGS.finetune_learning_rate
        self.finetune_num_epochs = FLAGS.finetune_num_epochs
        self.finetune_batch_size = FLAGS.finetune_batch_size
        self.finetune_opt = FLAGS.finetune_opt
        self.finetune_loss_func = FLAGS.finetune_loss_func
        self.finetune_dropout = FLAGS.finetune_dropout


class DeepBeliefNetwork(SupervisedModel):
    """Implementation of Deep Belief Network for Supervised Learning.

    The interface of the class is sklearn-like.
    """

    def __init__(self, dbn_param):
        self.dbn_param = dbn_param
        SupervisedModel.__init__(
            self,
            dbn_param.model_name,
            dbn_param.main_dir,
            dbn_param.models_dir,
            dbn_param.data_dir,
            dbn_param.summary_dir)

        self._initialize_training_parameters(
            loss_func=dbn_param.finetune_loss_func,
            learning_rate=dbn_param.finetune_learning_rate,
            dropout=dbn_param.finetune_dropout,
            num_epochs=dbn_param.finetune_num_epochs,
            batch_size=dbn_param.finetune_batch_size,
            dataset=dbn_param.dataset,
            opt=dbn_param.finetune_opt,
            momentum=dbn_param.momentum)

        self.do_pretrain = dbn_param.do_pretrain
        self.encoder_layers = dbn_param.encoder_layers
        self.decoder_layers = dbn_param.decoder_layers
        self.encode_act_func = dbn_param.encode_act_func
        self.decode_act_func = dbn_param.decode_act_func
        self.verbose = dbn_param.verbose

        # Model parameters
        self.encoding_w_ = []  # list of matrices of encoding weights per layer
        self.encoding_b_ = []  # list of arrays of encoding biases per layer
        self.decoding_w_ = []
        self.decoding_b_ = []

        self.softmax_W = None
        self.softmax_b = None

        print(self.encoder_layers)
        print(self.decoder_layers)
        rbm_params = {
            'num_epochs': dbn_param.rbm_num_epochs, 'gibbs_k': dbn_param.rbm_gibbs_k,
            'batch_size': dbn_param.rbm_batch_size, 'learning_rate': dbn_param.rbm_learning_rate}

        for p in rbm_params:
            if len(rbm_params[p]) != len(self.encoder_layers):
                # The current parameter is not specified by the user,
                # should default it for all the encoder_layers
                rbm_params[p] = [rbm_params[p][0] for _ in self.encoder_layers]
        print(rbm_params)

        self.rbms = []
        self.rbm_graphs = []

        for l, layer in enumerate(self.encoder_layers):
            rbm_str = 'rbm-' + str(l + 1)

            if l == 0 and dbn_param.rbm_gauss_visible:
                self.rbms.append(
                    rbm.RBM(
                        model_name=self.model_name + '-' + rbm_str,
                        models_dir=os.path.join(self.models_dir, rbm_str),
                        data_dir=os.path.join(self.data_dir, rbm_str),
                        summary_dir=os.path.join(self.tf_summary_dir, rbm_str),
                        num_hidden=layer, main_dir=self.main_dir,
                        learning_rate=rbm_params['learning_rate'][l],
                        dataset=self.dataset, verbose=self.verbose,
                        num_epochs=rbm_params['num_epochs'][l],
                        batch_size=rbm_params['batch_size'][l],
                        gibbs_sampling_steps=rbm_params['gibbs_k'][l],
                        visible_unit_type='gauss', stddev=rbm_stddev))

            else:
                self.rbms.append(
                    rbm.RBM(
                        model_name=self.model_name + '-' + rbm_str,
                        models_dir=os.path.join(self.models_dir, rbm_str),
                        data_dir=os.path.join(self.data_dir, rbm_str),
                        summary_dir=os.path.join(self.tf_summary_dir, rbm_str),
                        num_hidden=layer, main_dir=self.main_dir,
                        learning_rate=rbm_params['learning_rate'][l],
                        dataset=self.dataset, verbose=self.verbose,
                        num_epochs=rbm_params['num_epochs'][l],
                        batch_size=rbm_params['batch_size'][l],
                        gibbs_sampling_steps=rbm_params['gibbs_k'][l]))

            self.rbm_graphs.append(tf.Graph())

    def pretrain(self, train_set, validation_set=None):
        """Perform Unsupervised pretraining of the DBN."""
        self.do_pretrain = True

        def set_params_func(rbmmachine, rbmgraph):
            params = rbmmachine.get_model_parameters(graph=rbmgraph)
            self.encoding_w_.append(params['W'])
            self.encoding_b_.append(params['bh_'])

        return SupervisedModel.pretrain_procedure(
            self, self.rbms, self.rbm_graphs, set_params_func=set_params_func,
            train_set=train_set, validation_set=validation_set)

    def _train_model(self, train_set, train_labels,
                     validation_set, validation_labels):
        """Train the model.

        :param train_set: training set
        :param train_labels: training labels
        :param validation_set: validation set
        :param validation_labels: validation labels
        :return: self
        """
        shuff = zip(train_set, train_labels)

        for i in range(self.num_epochs):

            np.random.shuffle(shuff)
            batches = [_ for _ in utilities.gen_batches(
                shuff, self.batch_size)]

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                self.tf_session.run(
                    self.train_step, feed_dict={
                        self.input_data: x_batch,
                        self.input_labels: y_batch,
                        self.keep_prob: self.dropout})

            if validation_set is not None:
                feed = {self.input_data: validation_set,
                        self.input_labels: validation_labels,
                        self.keep_prob: 1}
                self._run_validation_error_and_summaries(i, feed)

    def build_model(self, n_features, n_classes):
        """Create the computational graph.

        This graph is intented to be created for finetuning,
        i.e. after unsupervised pretraining.
        :param n_features: Number of features.
        :param n_classes: number of classes.
        :return: self
        """
        self._create_placeholders(n_features, n_classes)
        self._create_variables(n_features)
        self._create_decoding_variables(self.encoder_layers[-1])

        encoder = self._create_encoding_layers()
        decoder = self._create_decoding_lyaers(encoder)
        last_out = self._create_last_layer(decoder, n_classes)

        self._create_cost_function_node(last_out, self.input_labels)
        self._create_train_step_node()
        self._create_accuracy_test_node()

    def _create_placeholders(self, n_features, n_classes):
        """Create the TensorFlow placeholders for the model.

        :param n_features: number of features of the first layer
        :param n_classes: number of classes
        :return: self
        """
        self.input_data = tf.placeholder(
            tf.float32, [None, n_features], name='x-input')

        self.input_labels = tf.placeholder(
            tf.float32, [None, n_classes], name='y-input')

        self.keep_prob = tf.placeholder(tf.float32, name='keep-probs')

    def _create_variables(self, n_features):
        """Create the TensorFlow variables for the model.

        :param n_features: number of features
        :return: self
        """
        if self.do_pretrain:
            self._create_variables_pretrain()
        else:
            self._create_variables_no_pretrain(n_features)

    def _create_variables_no_pretrain(self, n_features):
        """Create model variables (no previous unsupervised pretraining).

        :param n_features: number of features
        :return: self
        """
        self.encoding_w_ = []
        self.encoding_b_ = []

        for l, layer in enumerate(self.encoder_layers):

            if l == 0:
                self.encoding_w_.append(tf.Variable(tf.truncated_normal(
                    shape=[n_features, self.encoder_layers[l]], stddev=0.1)))
                self.encoding_b_.append(tf.Variable(tf.constant(
                    0.1, shape=[self.encoder_layers[l]])))
            else:
                self.encoding_w_.append(tf.Variable(tf.truncated_normal(
                    shape=[self.encoder_layers[l - 1], self.encoder_layers[l]], stddev=0.1)))
                self.encoding_b_.append(tf.Variable(tf.constant(
                    0.1, shape=[self.encoder_layers[l]])))

    def _create_variables_pretrain(self):
        """Create model variables (previous unsupervised pretraining).

        :return: self
        """
        for l, layer in enumerate(self.encoder_layers):
            self.encoding_w_[l] = tf.Variable(
                self.encoding_w_[l], name='enc-w-{}'.format(l))
            self.encoding_b_[l] = tf.Variable(
                self.encoding_b_[l], name='enc-b-{}'.format(l))

    def _create_encoding_layers(self):
        """Create the encoding encoder_layers for supervised finetuning.

        :return: output of the final encoding layer.
        """
        next_train = self.input_data
        self.layer_nodes = []

        for l, layer in enumerate(self.encoder_layers):

            with tf.name_scope("encode-{}".format(l)):

                y_act = tf.add(
                    tf.matmul(next_train, self.encoding_w_[l]),
                    self.encoding_b_[l]
                )

                if self.encode_act_func:
                    layer_y = self.encode_act_func(y_act)

                else:
                    layer_y = None

                # the input to the next layer is the output of this layer
                next_train = tf.nn.dropout(layer_y, self.keep_prob)

            self.layer_nodes.append(next_train)

        return next_train

    def _create_decoding_variables(self, last_encoder_num):
        """Create model variables (previous unsupervised pretraining).

        :return: self
        """
        for l, layer in enumerate(self.decoder_layers):
            if l == 0:
                self.decoding_w_.append(tf.Variable(tf.truncated_normal(
                    shape=[last_encoder_num, self.decoder_layers[l]], stddev=0.1)))
                self.decoding_b_.append(tf.Variable(tf.constant(
                    0.1, shape=[self.decoder_layers[l]])))
            else:
                self.decoding_w_.append(tf.Variable(tf.truncated_normal(
                    shape=[self.decoder_layers[l - 1], self.decoder_layers[l]], stddev=0.1)))
                self.decoding_b_.append(tf.Variable(tf.constant(
                    0.1, shape=[self.decoder_layers[l]])))

    def _create_decoding_lyaers(self, encoder):
        """Create the encoding decoder_layers for supervised finetuning.

        :return: output of the final encoding layer.
        """
        # next_train = self.input_data
        self.decoder_layer_nodes = []
        decoder = encoder

        for l, layer in enumerate(self.decoder_layers):

            with tf.name_scope("decode-{}".format(l)):

                y_act = tf.add(
                    tf.matmul(decoder, self.decoding_w_[l]),
                    self.decoding_b_[l]
                )

                if self.decode_act_func:
                    layer_y = self.decode_act_func(y_act)

                else:
                    layer_y = None

                # the input to the next layer is the output of this layer
                decoder = tf.nn.dropout(layer_y, self.keep_prob)

            self.decoder_layer_nodes.append(decoder)

        return decoder
