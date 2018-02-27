import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow.contrib.slim as slim
import os

def sample_z(m, n):
    sample = np.random.uniform(-1., 1., size=[m, n])
    sample = sample.astype(np.float32, copy=False)
    return sample

def plot(samples):
    fig = plt.figure(figsize=(4, 8))
    gs = gridspec.GridSpec(4, 8)
    gs.update(wspace=0.05, hspace=0.05)
    #print len(samples)
    for i, sample in enumerate(samples):
        #print i
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

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

def G(z, scope='Generator'):
    with tf.variable_scope(scope) as scope:
        #log.warn(scope.name)
        z = tf.reshape(z, [batch_size, 1, 1, -1])
        g_1 = deconv2d(z, deconv_info[0], is_train, name='g_1_deconv')
        #log.info('{} {}'.format(scope.name, g_1))
        g_2 = deconv2d(g_1, deconv_info[1], is_train, name='g_2_deconv')
        #log.info('{} {}'.format(scope.name, g_2))
        g_3 = deconv2d(g_2, deconv_info[2], is_train, name='g_3_deconv')
        #log.info('{} {}'.format(scope.name, g_3))
        g_4 = deconv2d(g_3, deconv_info[3], is_train, name='g_4_deconv', activation_fn=tf.tanh)
        #log.info('{} {}'.format(scope.name, g_4))
        output = g_4
        #print X.get_shape().as_list()
        assert output.get_shape().as_list() == image_shape, output.get_shape().as_list()
        return output

def D(img, scope='Discriminator', reuse=True):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        #if not reuse: log.warn(scope.name)
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

def entropy(y, epsilon):
    #batch_size= K.int_shape(y)[0]
    y_1 = -y * tf.log(y+epsilon)
    y_2 = tf.reduce_sum(y_1,axis=1)
    y_3 = tf.reduce_mean(y_2,axis=0)
    return y_3

def main(args):
    parser.add_argument("--batch_size",dest="batch_size",type=int,default=128)
    parser.add_argument("--z_dim",dest="z_dim",type=int,default=100)
    parser.add_argument("--h_dim",dest="h_dim",type=int,default=128)
    parser.add_argument("--lr",dest="lr",type=float,default=1e-3)
    parser.add_argument("--n_class",dest="n_class",type=int,default=10)
    parser.add_argument("--dataset", dest="dataset", type=str, default=None)
    parser.add_argument("--is_train", dest="is_train", type=bool, default=True)
    parser.add_argument("--epsilon",dest="epsilon",type=float,default=1e-6)
    # cross entropy constant
    parser.add_argument("--ce_term",dest="ce_term",type=float,default=1)
    parser.add_argument("--dir_name", dest="dir_name", type=str, default=None)

    opts = parser.parse_args(args[1:])

    batch_size = opts.batch_size
    z_dim = opts.z_dim
    h_dim = opts.h_dim
    lr = opts.lr
    n_clas = opts.n_class
    dataset = opts.dataset
    is_train = opts.is_train
    epsilon = opts.epsilon
    ce_term = opts.ce_term
    dir_name = opts.dir_name


    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if opts.dataset == 'mnist':
        mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
        x_train = mnist.train.images[:50,:]
        x_train = x_train.reshape([50,28,28,1])
        data_info = np.array([28, 28, 10, 1])
        conv_info = np.array([32, 64, 128])
        deconv_info = np.array([[100, 2, 1], [25, 3, 2], [6, 4, 2], [1, 6, 2]])
        image_shape = [32, 28, 28, 1]
    else:
        x_train = np.zeros((50,28,28,1))
        data_info = np.array([0, 0, 0])
        conv_info = np.array([0, 0, 0])
        deconv_info = np.array([0, 0, 0])
        image_shape = [0,0,0,0]

    z = tf.random_uniform([batch_size, nz], minval=-1, maxval=1, dtype=tf.float32)
    G_sample = G(z)

    D_real, D_real_logits = D(X, scope='Discriminator', reuse=False)
    D_fake, D_fake_logits = D(G_sample, scope='Discriminator', reuse=True)

    D_target = 1./mb_size
    G_target = 1./(mb_size*2)

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
    i = 0
    with tf.Session() as sess:
        for it in range(10000):
            X_mb, y_mb = mnist.train.next_batch(mb_size)
            X_mb = X_mb.reshape([mb_size, 28, 28, 1])

        _, D_loss_curr = sess.run(
            [D_solver, D_loss], feed_dict={X: X_mb, y: y_mb}
        )

        _, G_loss_curr = sess.run(
            [G_solver, G_loss], feed_dict={X: X_mb, y: y_mb}
        )

        if it % 100 == 0:
            print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'
                .format(it, D_loss_curr, G_loss_curr))
            g_loss_overall.append(G_loss_curr)
            d_loss_overall.append(D_loss_curr)
            samples = sess.run(G_sample)

            fig = plot(samples)
            plt.savefig(dir_name+'{}.png'
                    .format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

    plt.figure()
    g_loss_plot, = plt.plot(g_loss_overall, label='G Loss')
    d_loss_plot, = plt.plot(d_loss_overall, label='D Loss')

    plt.legend([g_loss_plot, d_loss_plot], ['G Loss', 'D Loss'])
    plt.savefig(dir_name+'plot_for_catgan_.png')
    return 0


if __name__ == '__main__':
    main(sys.argv)
