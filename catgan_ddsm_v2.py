import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow.contrib.slim as slim
from dataset import load_ddsm_data
import os
import sys
import argparse
from collections import OrderedDict
from utils import line_writer

epsilon = 1e-6

def sample_z(m, n):
    sample = np.random.uniform(-1., 1., size=[m, n])
    sample = sample.astype(np.float32, copy=False)
    return sample

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

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)

def conv2d(input, output_shape, is_train, k_h=5, k_w=5, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input.get_shape()[-1], output_shape],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, 2, 2, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_shape], initializer=tf.constant_initializer(0.0))
        conv = lrelu(tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape()))
        bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True,
                                          decay=0.9, is_training=is_train,
                                          updates_collections=None)
    return bn

def deconv2d(input, deconv_info, is_train, name="deconv2d", stddev=0.02, activation_fn=None):
    with tf.variable_scope(name):
        output_shape = deconv_info[0]
        k = deconv_info[1]
        s = deconv_info[2]
        deconv = layers.conv2d_transpose(
            input, num_outputs=output_shape,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.zeros_initializer(),
            kernel_size=[k, k], stride=[s, s], padding='VALID'
        )
        if not activation_fn:
            deconv = tf.nn.relu(deconv)
            deconv = tf.contrib.layers.batch_norm(
                deconv, center=True, scale=True,  decay=0.9,
                is_training=is_train, updates_collections=None
            )
        else:
            deconv = activation_fn(deconv)
        return deconv

def deconv2d_new(input_, output_shape,
             k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False, activation_fn=None, is_train=True):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for versions of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if not activation_fn:
            deconv = tf.nn.relu(deconv)
            deconv = tf.contrib.layers.batch_norm(
                deconv, center=True, scale=True,  decay=0.9,
                is_training=is_train, updates_collections=None
            )
        else:
            deconv = activation_fn(deconv)
        #print 'deconv shape:', deconv.get_shape()
        #print 'w shape:', w.get_shape()
        if with_w:
            return deconv, w, biases
        else:
            return deconv

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def G(z, device_name, batch_size, n_class, is_train, deconv_info, scope='Generator'):
    image_shape = [batch_size, 224, 224, 3]
    with tf.variable_scope(scope) as scope:
        #log.warn(scope.name)
        s = 224
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        gf_dim = 64
        c_dim = 3
        z_1, h0_w, h0_b = linear(z, gf_dim*8*s16*s16, scope=scope, with_w=True)
        z_1 = tf.reshape(z_1, [-1, s16, s16, gf_dim * 8])
        z_1 = tf.nn.relu(z_1)
        z_1 = tf.contrib.layers.batch_norm(
                z_1, center=True, scale=True,  decay=0.9,
                is_training=is_train, updates_collections=None
            )
        g_1 = deconv2d_new(z_1, [batch_size, s8, s8, gf_dim*4], name='g_1_deconv', with_w=False, is_train=is_train)
        g_2 = deconv2d_new(g_1, [batch_size, s4, s4, gf_dim*2], name='g_2_deconv', with_w=False, is_train=is_train)
        g_3 = deconv2d_new(g_2, [batch_size, s2, s2, gf_dim*1], name='g_3_deconv', with_w=False, is_train=is_train)
        g_4 = deconv2d_new(g_3, [batch_size, s, s, c_dim], name='g_h4', with_w=False, activation_fn=tf.tanh, is_train=is_train)
        print(g_4.shape)
        output = g_4
        #print X.get_shape().as_list()
        assert output.get_shape().as_list() == image_shape, output.get_shape().as_list()
        return output

def D(img, device_name, batch_size, n_class, is_train, conv_info, scope='Discriminator', reuse=True):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        #if not reuse: log.warn(scope.name)
        with tf.device(device_name):
            d_1 = conv2d(img, conv_info[0], is_train, name='d_1_conv')
            d_1 = slim.dropout(d_1, keep_prob=0.5, is_training=is_train, scope='d_1_conv/')
        #if not reuse: log.info('{} {}'.format(scope.name, d_1))
            d_2 = conv2d(d_1, conv_info[1], is_train, name='d_2_conv')
            d_2 = slim.dropout(d_2, keep_prob=0.5, is_training=is_train, scope='d_2_conv/')
        #if not reuse: log.info('{} {}'.format(scope.name, d_2))
            d_3 = conv2d(d_2, conv_info[2], is_train, name='d_3_conv')
            d_3 = slim.dropout(d_3, keep_prob=0.5, is_training=is_train, scope='d_3_conv/')
        #if not reuse: log.info('{} {}'.format(scope.name, d_3))
            d_4 = slim.fully_connected(
                tf.reshape(d_3, [batch_size, -1]), n_class, scope='d_4_fc', activation_fn=None)
        #if not reuse: log.info('{} {}'.format(scope.name, d_4))
            output = d_4
            assert output.get_shape().as_list() == [batch_size, n_class]
            return tf.nn.softmax(output), output

# y has dim batch_size x num_classes
# entropy 1
def marginal_entropy(y):
    y_1 = tf.reduce_mean(y, axis=0) #1/N sum y_i
    y_2 = -y_1 * tf.log(y_1+epsilon)
    y_3 = tf.reduce_sum(y_2)
    return y_3

def entropy(y):
    #batch_size= K.int_shape(y)[0]
    y_1 = -y * tf.log(y+epsilon)
    y_2 = tf.reduce_sum(y_1,axis=1)
    y_3 = tf.reduce_mean(y_2,axis=0)
    return y_3

def main(args):
    parser = argparse.ArgumentParser(description = "Catgan ddsm")
    parser.add_argument("--batch_size",dest="batch_size",type=int,default=128)
    parser.add_argument("--z_dim",dest="z_dim",type=int,default=100)
    parser.add_argument("--h_dim",dest="h_dim",type=int,default=128)
    parser.add_argument("--lr",dest="lr",type=float,default=1e-3)
    parser.add_argument("--n_class",dest="n_class",type=int,default=10)
    parser.add_argument("--dataset", dest="dataset", type=str, default=None)
    parser.add_argument("--is_train", dest="is_train", type=bool, default=True)
    #parser.add_argument("--epsilon",dest="epsilon",type=float,default=1e-6)
    # cross entropy constant
    parser.add_argument("--ce_term",dest="ce_term",type=float,default=1)
    parser.add_argument("--dir_name", dest="dir_name", type=str, default=None)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=1)
    parser.add_argument("--device_name", dest="device_name", type=str, default=None)
    opts = parser.parse_args(args[1:])

    if opts.device_name == "gpu":
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"
    print("DEVICE NAME: ", device_name)
    batch_size = opts.batch_size
    z_dim = opts.z_dim
    h_dim = opts.h_dim
    lr = opts.lr
    n_class = opts.n_class
    dataset = opts.dataset
    is_train = opts.is_train
    #epsilon = opts.epsilon
    ce_term = opts.ce_term
    dir_name = opts.dir_name
    num_epochs = opts.num_epochs


    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    ddsm_path = '/dfs/scratch0/annhe/tanda_750_90_10_split/'
    labels_fl = 'mass_to_label.json'
    labels_path = os.path.join(ddsm_path,labels_fl)

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = load_ddsm_data(data_dir=ddsm_path, \
        label_json=ddsm_path+'/'+'mass_to_label.json', validation_set=True, segmentations=False, as_float=True, channels=3)
    X_train = X_train[0:1104,:]
    Y_train = Y_train[0:1104,:]
    conv_info = np.array([64, 128, 256])
    #deconv_info = np.array([[300, 3, 1], [100, 7, 2], [50, 5, 2], [25, 5, 2], [12, 6, 2], [3, 6, 2]])
    #deconv_info = np.array([[1000, 3, 1], [250, 7, 2], [100, 5, 2], [25, 5, 2], [12, 6, 2], [3, 6, 2]])
    deconv_info = np.array([[1000, 6, 2], [250, 6, 2], [100, 6, 3], [25, 8, 2], [12, 8, 2], [3, 3, 1]])
    
    X = tf.placeholder(
            name='image', dtype=tf.float32,
            shape=[batch_size, 224, 224, 3],
        )
    z = tf.placeholder(tf.float32, shape=[None, z_dim])
    y = tf.placeholder(
            name='label', dtype=tf.float32, shape=[batch_size, n_class],
        )

    with tf.device(device_name):    
        z = tf.random_uniform([batch_size, z_dim], minval=-1, maxval=1, dtype=tf.float32)

    #def G(z, device_name, batch_size, n_class, is_train, deconv_info, scope='Generator'):    
    G_sample = G(z, device_name, batch_size, n_class, is_train, deconv_info)

    D_real, D_real_logits = D(X, device_name, batch_size, n_class, is_train, conv_info, scope='Discriminator', reuse=False)
    D_fake, D_fake_logits = D(G_sample, device_name, batch_size, n_class, is_train, conv_info, scope='Discriminator', reuse=True)

    D_target = 1./batch_size
    G_target = 1./(batch_size*2)

    D_loss = -marginal_entropy(D_real) + entropy(D_real) - entropy(D_fake) + ce_term*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=D_real_logits))
    G_loss = -marginal_entropy(D_fake) + entropy(D_fake)

    all_vars = tf.trainable_variables()

    theta_D = [v for v in all_vars if v.name.startswith('Discriminator')]
    #log.warn("********* d_var ********** "); slim.model_analyzer.analyze_vars(d_var, print_info=True)

    theta_G = [v for v in all_vars if v.name.startswith(('Generator'))]
    #log.warn("********* g_var ********** "); slim.model_analyzer.analyze_vars(g_var, print_info=True)

    D_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(D_loss, var_list=theta_D))
    G_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(G_loss, var_list=theta_G))


    g_loss_overall = []
    d_loss_overall = []
    counter = 0
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        N_train = X_train.shape[0]
        n_batches = int(np.ceil(N_train / float(batch_size)))
        for it in range(num_epochs):
            for i, b in enumerate(range(0, N_train, batch_size)):
                X_batch = X_train[b : b + batch_size, :]
                Y_batch = Y_train[b : b + batch_size, :]
                if (X_batch.shape[0] != batch_size): continue

                _, D_loss_curr = sess.run(
                [D_solver, D_loss], feed_dict={X: X_batch, y: Y_batch}
                )

                _, G_loss_curr = sess.run(
                [G_solver, G_loss], feed_dict={X: X_batch, y: Y_batch}
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
                    line_writer(OrderedDict(info))
                    g_loss_overall.append(G_loss_curr)
                    d_loss_overall.append(D_loss_curr)
                    samples = sess.run(G_sample)
                    truncated_samples = samples[0:16,:]
                    fig = plot(truncated_samples)
                    plt.savefig(dir_name+'{}.png'
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
    return 0


if __name__ == '__main__':
    main(sys.argv)
