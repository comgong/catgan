import argparse
import os
import sys
from catgan_model import CATGAN
from dataset import load_ddsm_data
import tensorflow as tf


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


    # create CATGAN
    config_x = {'disc_lr':0.001, 'gen_lr':0.001, 'beta1':0.9}

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        catgan = CATGAN(sess, dir_name)
        catgan.train(config_x,X_train, Y_train, X_valid, Y_valid, X_test, Y_test)

    return 0

if __name__ == '__main__':
    main(sys.argv)
