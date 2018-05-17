import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow.contrib.slim as slim
import os
import sys
from collections import OrderedDict
from utils import line_writer

from DCGANops import *

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    #print len(samples)
    for i, sample in enumerate(samples):
        #print i
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        sample_one_layer = sample[:,:,0]
        #print sample_one_layer.shape
        plt.imshow(sample_one_layer.reshape(224, 224), cmap='Greys_r')

    return fig

class CATGAN(object):
    def __init__(self, sess, dir_name, num_epochs = 2, epsilon = 1e-6, ce_term = 1.0, image_size=224, batch_size=48, sample_size=16, output_size=224,
        n_class=2, z_dim=100, gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024,c_dim=3,
        checkpoint_dir=None, sample_dir=None, train_dir=None):

        self.use_deconv = True
        self.dir_name = dir_name
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size
        self.ce_term = ce_term
        self.epsilon = epsilon
        self.num_epochs = num_epochs

        self.n_class = n_class # num classes
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        self.d_bn1 = batch_norm_DCGAN(name='d_bn1')
        self.d_bn2 = batch_norm_DCGAN(name='d_bn2')
        self.d_bn3 = batch_norm_DCGAN(name='d_bn3')

        self.g_bn0 = batch_norm_DCGAN(name='g_bn0')
        self.g_bn1 = batch_norm_DCGAN(name='g_bn1')
        self.g_bn2 = batch_norm_DCGAN(name='g_bn2')
        self.g_bn3 = batch_norm_DCGAN(name='g_bn3')


        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def marginal_entropy(self,y):
        y_1 = tf.reduce_mean(y, axis=0) #1/N sum y_i
        y_2 = -y_1 * tf.log(y_1+self.epsilon)
        y_3 = tf.reduce_sum(y_2)
        return y_3

    def entropy(self,y):
    #batch_size= K.int_shape(y)[0]
        y_1 = -y * tf.log(y+self.epsilon)
        y_2 = tf.reduce_sum(y_1,axis=1)
        y_3 = tf.reduce_mean(y_2,axis=0)
        return y_3  

    def build_model(self):

        self.y= tf.placeholder(tf.float32, [self.batch_size, self.n_class], name='y')
        self.images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim],
                                    name='real_images')

        # what is this going to be used for?
        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + [self.output_size, self.output_size, self.c_dim],
                                        name='sample_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim],
                                name='z')

        self.G = self.generator(self.z, self.y)
        # real?
        self.D_real, self.D_real_logits  = self.discriminator(self.images, self.y, reuse=False)

        self.sampler = self.sampler(self.z, self.y)
        self.D_fake, self.D_fake_logits = self.discriminator(self.G, self.y, reuse=True)

        self.D_target = 1./self.batch_size
        self.G_target = 1./(self.batch_size*2)

        #print(self.D_real.shape)
        self.D_loss = -self.marginal_entropy(self.D_real) + self.entropy(self.D_real) - self.entropy(self.D_fake) + self.ce_term*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.D_real_logits))
        self.G_loss = -self.marginal_entropy(self.D_fake) + self.entropy(self.D_fake)

        #self.g_loss_sum = tf.summary.scalar("g_loss", self.G_loss)
        #self.d_loss_sum = tf.summary.scalar("d_loss", self.D_loss)

        t_vars = tf.trainable_variables()
        print("LENGTH OF TRAINABLE VARS", len(t_vars))

        self.d_vars = [v for v in t_vars if v.name.startswith('Discriminator')]
        #log.warn("********* d_var ********** "); slim.model_analyzer.analyze_vars(d_var, print_info=True)

        self.g_vars = [v for v in t_vars if v.name.startswith(('Generator'))]
        print("LENGTH OF D VARS", len(self.d_vars))
        print("LENGTH OF G VARS", len(self.g_vars))
        print("g_vars: ", self.g_vars)

        self.saver = tf.train.Saver()
        #initialize saver that saves only d_vars 
        all_vars = tf.global_variables()
        print("LENGTH OF ALL VARS", len(all_vars))
        vars_l = [v for v in all_vars if v.name.startswith('Discriminator')]

        self.d_saver = tf.train.Saver(vars_l)

    def train(self, config, X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
        d_optim = tf.train.AdamOptimizer(config['disc_lr'], beta1=config['beta1']) \
                          .minimize(self.D_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config['gen_lr'], beta1=config['beta1']) \
                          .minimize(self.G_loss, var_list=self.g_vars)

        try: 
            tf.global_variables_initializer().run()
        except: 
            tf.initialize_all_variables().run()

        # can add summary / tfwriter here later

        # can add loading from checkpoint here later

        # this is for checking every 100 iterations, etc
        sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))
        sample_images = X_valid[0:self.batch_size,:]
        sample_y = Y_valid[0:self.batch_size,:]

        N_train = X_train.shape[0]
        n_batches = int(np.ceil(N_train / float(self.batch_size)))

        g_loss_overall = []
        d_loss_overall = []
        counter = 0
        for it in range(self.num_epochs):
            for i, b in enumerate(range(0, N_train, self.batch_size)):
                X_batch = X_train[b : b + self.batch_size, :]
                Y_batch = Y_train[b : b + self.batch_size, :]
                if (X_batch.shape[0] != self.batch_size): continue

                # random sample
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
                            .astype(np.float32)

                _, D_loss_curr = self.sess.run(
                [d_optim, self.D_loss], feed_dict={self.images: X_batch, self.y: Y_batch, self.z: batch_z}
                )

                _, G_loss_curr = self.sess.run(
                [g_optim, self.G_loss], feed_dict={self.z: batch_z}
                )

                if i % 100 == 0:
                    #print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'
                    #    .format(i, D_loss_curr, G_loss_curr))
                    info = [
                        ('Epoch', str(it)),
                        ('Batch', str(i)),
                        ('D_loss', D_loss_curr),
                        ('G_loss', G_loss_curr)
                    ]
                    dictionary = OrderedDict(info)
                    print(dictionary)
                    line_writer(dictionary)
                    g_loss_overall.append(G_loss_curr)
                    d_loss_overall.append(D_loss_curr)
                    samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.D_loss, self.G_loss],
                            feed_dict={self.z: sample_z, self.images: sample_images, self.y: sample_y}
                        )
                    truncated_samples = samples[0:16,:]
                    fig = plot(truncated_samples)
                    plt.savefig(self.dir_name+'{}.png'
                        .format(str(counter).zfill(3)), bbox_inches='tight')
                    counter += 1
                    plt.close(fig)
            idxs = list(range(X_train.shape[0]))
            np.random.shuffle(idxs)
            X_train = X_train[idxs]
            Y_train = Y_train[idxs]
        plt.figure()
        g_loss_plot, = plt.plot(g_loss_overall, label='G Loss')
        d_loss_plot, = plt.plot(d_loss_overall, label='D Loss')

        plt.legend([g_loss_plot, d_loss_plot], ['G Loss', 'D Loss'])
        plt.savefig(dir_name+'plot_for_catgan.png')

    def discriminator(self, image, y=None, reuse=None):
        # if reuse:
        #     tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("Discriminator") as scope:
            if reuse: 
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), self.n_class, 'd_h3_lin')

            # SIGMOID OR SOFTMAX?
            assert(h4.get_shape().as_list() == [self.batch_size, self.n_class])
            return tf.nn.sigmoid(h4), h4

    def generator(self, z, y=None): 
        with tf.variable_scope("Generator") as scope: 
            s = self.output_size
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
                
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s16*s16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(self.z_, [-1, s16, s16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            if self.use_deconv: 
                self.h1, self.h1_w, self.h1_b = deconv2d(h0, 
                        [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))

                h2, self.h2_w, self.h2_b = deconv2d(h1,
                        [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, self.h3_w, self.h3_b = deconv2d(h2,
                        [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, self.h4_w, self.h4_b = deconv2d(h3,
                        [self.batch_size, s, s, self.c_dim], name='g_h4', with_w=True)
            else: 
                self.h1, self.h1_w, self.h1_b = resize_conv2d(h0, 
                        [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))

                h2, self.h2_w, self.h2_b = resize_conv2d(h1,
                        [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, self.h3_w, self.h3_b = resize_conv2d(h2,
                        [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, self.h4_w, self.h4_b = resize_conv2d(h3,
                        [self.batch_size, s, s, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def sampler(self, z, y=None):
        with tf.variable_scope("Generator") as scope: 

            scope.reuse_variables()

                
            s = self.output_size
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

                # project `z` and reshape
            h0 = tf.reshape(linear(z, self.gf_dim*8*s16*s16, 'g_h0_lin'),
                                [-1, s16, s16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            if self.use_deconv:
                h1 = deconv2d(h0, [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=False))

                h2 = deconv2d(h1, [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=False))

                h3 = deconv2d(h2, [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=False))

                h4 = deconv2d(h3, [self.batch_size, s, s, self.c_dim], name='g_h4')
            else: 
                h1 = resize_conv2d(h0, [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=False))

                h2 = resize_conv2d(h1, [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=False))

                h3 = resize_conv2d(h2, [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=False))

                h4 = resize_conv2d(h3, [self.batch_size, s, s, self.c_dim], name='g_h4')
                
            return tf.nn.tanh(h4)

       