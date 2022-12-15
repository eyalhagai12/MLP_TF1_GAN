import itertools
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import create_dataset, convert_batch_to_numpy
from visualize import show_train_hist, show_image_grid, init_vis_workspace, save_session, log

tf.compat.v1.disable_v2_behavior()

IMG_SHAPE = 128

# G(z)
def generator(x):
    # initializers
    w_init = tf.compat.v1.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    # 1st hidden layer
    w0 = tf.compat.v1.get_variable('G_w0', [x.get_shape()[1], 256], initializer=w_init)
    b0 = tf.compat.v1.get_variable('G_b0', [256], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)

    # 2nd hidden layer
    w1 = tf.compat.v1.get_variable('G_w1', [h0.get_shape()[1], 512], initializer=w_init)
    b1 = tf.compat.v1.get_variable('G_b1', [512], initializer=b_init)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)

    # 3rd hidden layer
    w2 = tf.compat.v1.get_variable('G_w2', [h1.get_shape()[1], 1024], initializer=w_init)
    b2 = tf.compat.v1.get_variable('G_b2', [1024], initializer=b_init)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    # 4rd hidden layer
    w3 = tf.compat.v1.get_variable('G_w3', [h2.get_shape()[1], 4096], initializer=w_init)
    b3 = tf.compat.v1.get_variable('G_b3', [4096], initializer=b_init)
    h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)

    # 5rd hidden layer
    w4 = tf.compat.v1.get_variable('G_w4', [h3.get_shape()[1], 8128], initializer=w_init)
    b4 = tf.compat.v1.get_variable('G_b4', [8128], initializer=b_init)
    h4 = tf.nn.relu(tf.matmul(h3, w4) + b4)

    # 6rd hidden layer
    w5 = tf.compat.v1.get_variable('G_w5', [h4.get_shape()[1], 4096], initializer=w_init)
    b5 = tf.compat.v1.get_variable('G_b5', [4096], initializer=b_init)
    h5 = tf.nn.relu(tf.matmul(h4, w5) + b5)

    # 7rd hidden layer
    w6 = tf.compat.v1.get_variable('G_w6', [h5.get_shape()[1], 1024], initializer=w_init)
    b6 = tf.compat.v1.get_variable('G_b6', [1024], initializer=b_init)
    h6 = tf.nn.relu(tf.matmul(h5, w6) + b6)

    # output hidden layer
    w7 = tf.compat.v1.get_variable('G_w7', [h6.get_shape()[1], 3 * IMG_SHAPE * IMG_SHAPE], initializer=w_init)
    b7 = tf.compat.v1.get_variable('G_b7', [3 * IMG_SHAPE * IMG_SHAPE], initializer=b_init)
    o = tf.nn.tanh(tf.matmul(h6, w7) + b7, name="gen")

    return o


# G(z)
def weaker_generator(x):
    # initializers
    w_init = tf.compat.v1.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    # 1st hidden layer
    w0 = tf.compat.v1.get_variable('G_w0', [x.get_shape()[1], 256], initializer=w_init)
    b0 = tf.compat.v1.get_variable('G_b0', [256], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)

    # 2nd hidden layer
    w1 = tf.compat.v1.get_variable('G_w1', [h0.get_shape()[1], 512], initializer=w_init)
    b1 = tf.compat.v1.get_variable('G_b1', [512], initializer=b_init)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)

    # 3rd hidden layer
    w2 = tf.compat.v1.get_variable('G_w2', [h1.get_shape()[1], 1024], initializer=w_init)
    b2 = tf.compat.v1.get_variable('G_b2', [1024], initializer=b_init)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    # output hidden layer
    w3 = tf.compat.v1.get_variable('G_w3', [h2.get_shape()[1], 3 * IMG_SHAPE * IMG_SHAPE], initializer=w_init)
    b3 = tf.compat.v1.get_variable('G_b3', [3 * IMG_SHAPE * IMG_SHAPE], initializer=b_init)
    o = tf.nn.tanh(tf.matmul(h2, w3) + b3, name="gen")

    return o

def linear_generator(x):
    # initializers
    w_init = tf.compat.v1.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    # input layer
    w0 = tf.compat.v1.get_variable('G_w0', [x.get_shape()[1], 3*IMG_SHAPE*IMG_SHAPE], initializer=w_init)
    b0 = tf.compat.v1.get_variable('G_b0', [3*IMG_SHAPE*IMG_SHAPE], initializer=b_init)
    o = tf.sigmoid(tf.matmul(x, w0) + b0, name="gen")

    return o


# D(x)
def discriminator(x, drop_out):
    # initializers
    w_init = tf.compat.v1.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    # 1st hidden layer
    w0 = tf.compat.v1.get_variable('D_w0', [x.get_shape()[1], 1024], initializer=w_init)
    b0 = tf.compat.v1.get_variable('D_b0', [1024], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)
    h0 = tf.nn.dropout(h0, drop_out)

    # 2nd hidden layer
    w1 = tf.compat.v1.get_variable('D_w1', [h0.get_shape()[1], 512], initializer=w_init)
    b1 = tf.compat.v1.get_variable('D_b1', [512], initializer=b_init)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)
    h1 = tf.nn.dropout(h1, drop_out)

    # 3rd hidden layer
    w2 = tf.compat.v1.get_variable('D_w2', [h1.get_shape()[1], 256], initializer=w_init)
    b2 = tf.compat.v1.get_variable('D_b2', [256], initializer=b_init)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    h2 = tf.nn.dropout(h2, drop_out)

    # 4th hidden layer
    w3 = tf.compat.v1.get_variable('D_w3', [h2.get_shape()[1], 128], initializer=w_init)
    b3 = tf.compat.v1.get_variable('D_b3', [128], initializer=b_init)
    h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)
    h3 = tf.nn.dropout(h3, drop_out)

    # 5th hidden layer
    w4 = tf.compat.v1.get_variable('D_w4', [h3.get_shape()[1], 64], initializer=w_init)
    b4 = tf.compat.v1.get_variable('D_b4', [64], initializer=b_init)
    h4 = tf.nn.relu(tf.matmul(h3, w4) + b4)
    h4 = tf.nn.dropout(h4, drop_out)

    # output layer
    w5 = tf.compat.v1.get_variable('D_w5', [h4.get_shape()[1], 1], initializer=w_init)
    b5 = tf.compat.v1.get_variable('D_b5', [3 * IMG_SHAPE * IMG_SHAPE], initializer=b_init)
    o = tf.sigmoid(tf.matmul(h4, w5) + b5, name="disc")

    return o


# D(x)
def weaker_discriminator(x, drop_out):
    # initializers
    w_init = tf.compat.v1.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    # 1st hidden layer
    w0 = tf.compat.v1.get_variable('D_w0', [x.get_shape()[1], 1024], initializer=w_init)
    b0 = tf.compat.v1.get_variable('D_b0', [1024], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)
    h0 = tf.nn.dropout(h0, drop_out)

    # 2nd hidden layer
    w1 = tf.compat.v1.get_variable('D_w1', [h0.get_shape()[1], 512], initializer=w_init)
    b1 = tf.compat.v1.get_variable('D_b1', [512], initializer=b_init)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)
    h1 = tf.nn.dropout(h1, drop_out)

    # 3rd hidden layer
    w2 = tf.compat.v1.get_variable('D_w2', [h1.get_shape()[1], 256], initializer=w_init)
    b2 = tf.compat.v1.get_variable('D_b2', [256], initializer=b_init)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    h2 = tf.nn.dropout(h2, drop_out)

    # output layer
    w3 = tf.compat.v1.get_variable('D_w3', [h2.get_shape()[1], 1], initializer=w_init)
    b3 = tf.compat.v1.get_variable('D_b3', [3 * IMG_SHAPE * IMG_SHAPE], initializer=b_init)
    o = tf.sigmoid(tf.matmul(h2, w3) + b3, name="disc")

    return o

def linear_discriminator(x, drop_out):

    # initializers
    w_init = tf.compat.v1.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    # input layer
    w0 = tf.compat.v1.get_variable('D_w0', [x.get_shape()[1], 1], initializer=w_init)
    b0 = tf.compat.v1.get_variable('D_b0', [1], initializer=b_init)
    o = tf.sigmoid(tf.matmul(x, w0) + b0, name= "disc")

    return o



def show_result(test_set, num_epoch, show=False, save=False, path='result.png'):
    test_images = sess.run(G_z, {z: np.array(test_set).reshape(-1, IMG_SHAPE * IMG_SHAPE), drop_out: 0.0})

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5 * 5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (IMG_SHAPE, IMG_SHAPE, 3)))

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

    # plot the train images
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5 * 5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_set[k], (IMG_SHAPE, IMG_SHAPE)))

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig('GAN_results/GAN_train' + str(epoch + 1) + '.png')

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    # check GPU usage
    print("GPU available: ", tf.test.is_gpu_available())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # check which device to use
    if tf.test.is_gpu_available():
        device = "/gpu:0"
    else:
        device = "/cpu:0"

    # training parameters
    batch_size = 128
    lr = 0.0002
    train_epoch = 5

    # define dataset
    bw_dataset = create_dataset("sample_data/bw", batch_size)
    rgb_dataset = create_dataset("sample_data/rgb", batch_size)

    x_train, x_test, y_train, y_test = train_test_split(bw_dataset, rgb_dataset, test_size=0.2, shuffle=True,
                                                        random_state=42)

    # init visualization workspace
    init_vis_workspace()

    # initiate models
    with tf.device(device):
        # networks : generator
        with tf.compat.v1.variable_scope('G'):
            z = tf.compat.v1.placeholder(tf.float32, shape=(None, IMG_SHAPE * IMG_SHAPE))
            G_z = generator(z)   #   EDITED!!

        # networks : discriminator
        with tf.compat.v1.variable_scope('D') as scope:
            drop_out = tf.compat.v1.placeholder(dtype=tf.float32, name='drop_out')
            x = tf.compat.v1.placeholder(tf.float32, shape=(None, 3 * IMG_SHAPE * IMG_SHAPE))
            D_real = discriminator(x, drop_out)      # EDITED!!!
            scope.reuse_variables()
            D_fake = discriminator(G_z, drop_out)    # EDITED!!!

        # loss for each network
        eps = 1e-2
        D_loss = 0.5 * tf.reduce_mean(
            -tf.compat.v1.log(D_real + eps) - tf.compat.v1.log(tf.ones_like(G_z) - D_fake + eps))
        G_loss = tf.reduce_mean(-tf.compat.v1.log(D_fake + eps)) + tf.losses.mean_absolute_error(x, G_z)

        # trainable variables for each network
        t_vars = tf.compat.v1.trainable_variables()
        D_vars = [var for var in t_vars if 'D_' in var.name]
        G_vars = [var for var in t_vars if 'G_' in var.name]

        # optimizer for each network
        D_optim = tf.compat.v1.train.AdamOptimizer(lr).minimize(D_loss, var_list=D_vars)
        G_optim = tf.compat.v1.train.AdamOptimizer(lr).minimize(G_loss, var_list=G_vars)

    # open session and initialize all variables
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # results save folder
    if not os.path.isdir('GAN_results'):
        os.mkdir('GAN_results')

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    # training-loop
    np.random.seed(int(time.time()))
    start_time = time.time()
    min_loss = math.inf
    for epoch in range(train_epoch):
        G_losses = []
        D_losses = []
        epoch_start_time = time.time()
        for idx, (x_batch, y_batch) in tqdm(enumerate(zip(x_train, y_train)), desc="Epoch {}".format(epoch)):
            # preprocess input and output
            x_batch = convert_batch_to_numpy(x_batch, True).reshape(-1, IMG_SHAPE * IMG_SHAPE)
            y_batch = convert_batch_to_numpy(y_batch, False).reshape(-1, 3 * IMG_SHAPE * IMG_SHAPE)

            # update discriminator
            loss_d_, _ = sess.run([D_loss, D_optim], {x: y_batch, z: x_batch, drop_out: 0.3})
            D_losses.append(loss_d_)

            # update generator
            loss_g_, _ = sess.run([G_loss, G_optim], {z: x_batch, x: y_batch, drop_out: 0.3})
            G_losses.append(loss_g_)

        # save session
        save_session(sess, "last")

        # save best model every epoch
        if np.mean(G_losses) < min_loss:
            min_loss = np.mean(G_losses)
            save_session(sess, "best")

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % (
            (epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))

        # log results
        log("--------------------------- Epoch {} ---------------------------".format(epoch))
        log("Discriminator Loss: {}".format(np.mean(D_losses)))
        log("Generator Loss: {}\n".format(np.mean(G_losses)))

        # save results
        show_image_grid(G_z, z, sess, (x_test, y_test), epoch, 4, 4)
        train_hist['D_losses'].append(np.mean(D_losses))
        train_hist['G_losses'].append(np.mean(G_losses))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (
        np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
    print("Training finish!... save training results")

    show_train_hist(train_hist, save=True)

    sess.close()
