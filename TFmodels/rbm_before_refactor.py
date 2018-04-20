from tensorflow.python.framework import ops
import tensorflow as tf
import numpy as np
import os
from TFmodels import zconfig
from TFmodels import utils


class RBM(object):
    """ Restricted Boltzmann Machine implementation using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, nvis, nhid, vis_type='bin', directory_name='rbm', model_name='', gibbs_k=1, learning_rate=0.01,
                 batch_size=10, n_iter=10, stddev=0.1, verbose=0):

        self.nvis = nvis
        self.nhid = nhid
        self.vis_type = vis_type
        self.directory_name = directory_name
        self.model_name = model_name
        self.gibbs_k = gibbs_k
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.stddev = stddev
        self.verbose = verbose

        # Directories paths
        self.directory_name = self.directory_name + '/' if self.directory_name[-1] != '/' else self.directory_name

        self.models_dir = zconfig.models_dir + self.directory_name
        self.data_dir = zconfig.data_dir + self.directory_name
        self.summary_dir = zconfig.summary_dir + self.directory_name

        # Create dirs
        for d in [self.models_dir, self.data_dir, self.summary_dir]:
            if not os.path.isdir(d):
                os.mkdir(d)

        if self.model_name == '':
            # Assign models complete name
            self.model_name = 'rbm-{}-{}-{}-{}-{}-{}'.format(
                self.nvis, self.nhid, self.n_iter, self.batch_size, self.learning_rate, self.batch_size)

        # ############################# #
        #   Computational graph nodes   #
        # ############################# #

        # Model parameters
        self.W = None
        self.bh_ = None
        self.bv_ = None

        self.w_upd8 = None
        self.bh_upd8 = None
        self.bv_upd8 = None

        self.encode = None

        self.cost = None

        self.hrand = None
        self.vrand = None
        self.validation_size = None

        self.sess = None
        self.saver = None

    def _create_graph(self):

        # Symbolic variables
        self.x = tf.placeholder('float', [None, self.nvis], name='x-input')

        self.hrand = tf.placeholder('float', [None, self.nhid], name='hrand')
        self.vrand = tf.placeholder('float', [None, self.nvis], name='vrand-train')

        # Biases
        self.bh_ = tf.Variable(tf.zeros([self.nhid]), name='hidden-bias')
        self.bv_ = tf.Variable(tf.zeros([self.nvis]), name='visible-bias')

        self.W = tf.Variable(tf.random_normal((self.nvis, self.nhid), mean=0.0, stddev=0.01), name='weights')

        nn_input = self.x

        # Initialization
        hprobs0 = None
        hprobs = None
        positive = None
        vprobs = None
        hprobs1 = None
        hstates1 = None

        for step in range(self.gibbs_k):

            # Positive Contrastive Divergence phase
            hprobs = tf.nn.sigmoid(tf.matmul(nn_input, self.W) + self.bh_)
            hstates = utils.sample_prob(hprobs, self.hrand)

            # Compute positive associations in step 0
            if step == 0:
                hprobs0 = hprobs  # save the activation probabilities of the first step
                if self.vis_type == 'bin':
                    positive = tf.matmul(tf.transpose(nn_input), hstates)

                elif self.vis_type == 'gauss':
                    positive = tf.matmul(tf.transpose(nn_input), hprobs)

            # Negative Contrastive Divergence phase
            visible_activation = tf.matmul(hprobs, tf.transpose(self.W)) + self.bv_

            if self.vis_type == 'bin':
                vprobs = tf.nn.sigmoid(visible_activation)

            elif self.vis_type == 'gauss':
                vprobs = tf.truncated_normal((1, self.nvis), mean=visible_activation, stddev=self.stddev)

            # Sample again from the hidden units
            hprobs1 = tf.nn.sigmoid(tf.matmul(vprobs, self.W) + self.bh_)
            hstates1 = utils.sample_prob(hprobs1, self.hrand)

            # Use the reconstructed visible units as input for the next step
            nn_input = vprobs

        negative = tf.matmul(tf.transpose(vprobs), hprobs1)

        self.encode = hprobs  # encoded data

        self.w_upd8 = self.W.assign_add(self.learning_rate * (positive - negative))
        self.bh_upd8 = self.bh_.assign_add(self.learning_rate * tf.reduce_mean(hprobs0 - hprobs1, 0))
        self.bv_upd8 = self.bv_.assign_add(self.learning_rate * tf.reduce_mean(self.x - vprobs, 0))

        self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.x - vprobs)))
        _ = tf.summary.scalar("cost", self.cost)

    def fit(self, trX, vlX=None, restore_previous_model=False):

        if vlX is not None:
            self.validation_size = vlX.shape[0]

        # Reset tensorflow's default graph
        ops.reset_default_graph()

        self._create_graph()

        merged = tf.summary.merge_all()
        init_op = tf.initialize_all_variables()
        self.saver = tf.train.Saver()

        with tf.Session() as self.sess:

            self.sess.run(init_op)

            if restore_previous_model:
                # Restore previous models
                self.saver.restore(self.sess, self.models_dir + self.model_name)
                # Change models name
                self.model_name += '-restored{}'.format(self.n_iter)

            # Write tensorflow summaries to summary dir
            writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph_def)

            for i in range(self.n_iter):

                # Randomly shuffle the input
                np.random.shuffle(trX)

                batches = [_ for _ in utils.gen_batches(trX, self.batch_size)]

                for batch in batches:
                    self.sess.run([self.w_upd8, self.bh_upd8, self.bv_upd8],
                                  feed_dict={self.x: batch,
                                             self.hrand: np.random.rand(batch.shape[0], self.nhid),
                                             self.vrand: np.random.rand(batch.shape[0], self.nvis)})

                if i % 5 == 0:

                    # Record summary data
                    if vlX is not None:

                        feed = {self.x: vlX,
                                self.hrand: np.random.rand(self.validation_size, self.nhid),
                                self.vrand: np.random.rand(self.validation_size, self.nvis)}

                        result = self.sess.run([merged, self.cost], feed_dict=feed)
                        summary_str = result[0]
                        err = result[1]

                        writer.add_summary(summary_str, 1)

                        if self.verbose == 1:
                            print("Validation cost at step %s: %s" % (i, err))

            # Save trained models
            self.saver.save(self.sess, self.models_dir + self.model_name)

    def transform(self, data, name='train', gibbs_k=1, save=False, models_dir=''):
        """ Transform data according to the models.
        :type data: array_like
        :param data: DataUtils to transform
        :type name: string, default 'train'
        :param name: Identifier for the data that is being encoded
        :type gibbs_k: 1
        :param gibbs_k: Gibbs sampling steps
        :type save: boolean, default 'False'
        :param save: If true, save data to disk
        :return: transformed data
        """

        with tf.Session() as self.sess:
            # Restore trained models
            self.saver.restore(self.sess, self.models_dir + self.model_name)

            # Return the output of the encoding layer
            encoded_data = self.encode.eval({self.x: data,
                                             self.hrand: np.random.rand(data.shape[0], self.nhid),
                                             self.vrand: np.random.rand(data.shape[0], self.nvis)})

            if save:
                # Save transformation to output file
                np.save(self.data_dir + self.model_name + '-' + name, encoded_data)

            return encoded_data

    def load_model(self, shape, gibbs_k, model_path):
        """
        :param shape: tuple(nvis, nhid)
        :param model_path:
        :return:
        """
        self.nvis, self.nhid = shape[0], shape[1]
        self.gibbs_k = gibbs_k

        self._create_graph()

        # Initialize variables
        init_op = tf.initialize_all_variables()

        # Add ops to save and restore all the variables
        self.saver = tf.train.Saver()

        with tf.Session() as self.sess:
            self.sess.run(init_op)

            # Restore previous models
            self.saver.restore(self.sess, model_path)

    def get_model_parameters(self):
        """ Return the models parameters in the form of numpy arrays.
        :return: models parameters
        """
        with tf.Session() as self.sess:
            # Restore trained models
            self.saver.restore(self.sess, self.models_dir + self.model_name)

            return {
                'W': self.W.eval(),
                'bh_': self.bh_.eval(),
                'bv_': self.bv_.eval()
            }

    def get_weights_as_images(self, width, height, outdir='img/', n_images=10, img_type='grey'):

        outdir = self.data_dir + outdir

        with tf.Session() as self.sess:
            self.saver.restore(self.sess, self.models_dir + self.model_name)

            weights = self.W.eval()

            perm = np.random.permutation(self.nhid)[:n_images]

            for p in perm:
                w = np.array([i[p] for i in weights])
                image_path = outdir + self.model_name + '_{}.png'.format(p)
                utils.gen_image(w, width, height, image_path, img_type)