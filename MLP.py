import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from urllib3.connectionpool import xrange

tf.compat.v1.disable_v2_behavior()

# setup work environment
RESULT_IMAGES_DIR = "result_images"
RESULT_LOSS_DIR = "result_losses"
CHECKPOINTS_DIR = "checkpoints"
LOG_DIR = "log_dir"

os.system("rm -r {} {} {} {}".format(RESULT_IMAGES_DIR, RESULT_LOSS_DIR, CHECKPOINTS_DIR, LOG_DIR))

if not os.path.isdir(RESULT_IMAGES_DIR):
    os.mkdir(RESULT_IMAGES_DIR)

if not os.path.isdir(RESULT_LOSS_DIR):
    os.mkdir(RESULT_LOSS_DIR)

if not os.path.isdir(CHECKPOINTS_DIR):
    os.mkdir(CHECKPOINTS_DIR)

if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

# create logger for logging results
logger = logging.getLogger("mlp_log")
log_path = os.path.join(LOG_DIR, "epochs_log.log")
file_handler = logging.FileHandler(log_path, "w+")
log_formatter = logging.Formatter("[%(asctime)s] - [%(levelname)s] - [%(name)s]: %(message)s")
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)


def log(level, message):
    """
    function to make logging easier

    :param level: the level of the log message (info, debug, warning, error)
    :param message: the message
    """
    logger.setLevel(level)
    logger.log(level, message)


def linear(inp, out_features: int, name: str, c: float = 0.001):
    """
    Create a linear layer

    :param inp: the input tensor in the computational graph
    :param out_features: the number of output features
    :param name: the name of the layer
    :param c: clipping value
    :return: A Tensor representing the linear layer operation in the graph
    """
    weights = tf.compat.v1.get_variable(name, dtype=tf.float32,
                                        shape=(inp.shape[-1], out_features),
                                        initializer=tf.compat.v1.truncated_normal_initializer(mean=0, stddev=c),
                                        trainable=True,
                                        use_resource=True,
                                        constraint=lambda x: tf.compat.v1.clip_by_value(x, -c, c))
    bias = tf.compat.v1.get_variable(name + "_bias", [out_features],
                                     initializer=tf.compat.v1.constant_initializer(0.),
                                     trainable=True,
                                     use_resource=True)
    return tf.matmul(inp, weights) + bias


def sigmoid(inp):
    """
    Create a Sigmoid activation function layer

    :param inp: the input tensor in the computational graph
    :return: A Tensor operation of the sigmoid function in the graph
    """
    return tf.compat.v1.sigmoid(inp)


def tanh(inp):
    """
    Create a Tanh activation function layer

    :param inp: the input tensor in the computational graph
    :return: A Tensor operation of the Tanh function in the graph
    """
    return tf.compat.v1.tanh(inp)


def lrelu(inp, alpha: float):
    """
    Create a Leaky ReLU activation function layer

    :param inp: the input tensor in the computational graph
    :param alpha: the slope of the function in the negative x values (usually 0 < alpha <= 1)
    :return: A Tensor operation of the leaky relu function in the graph
    """
    return tf.compat.v1.nn.leaky_relu(inp, alpha)


def relu(inp):
    """
    Create a Leaky ReLU activation function layer

    :param inp: the input tensor in the computational graph
    :return: A Tensor operation of the relu function in the graph
    """
    return tf.compat.v1.nn.relu(inp)


def dropout(inp, drop_out: float = 0.3):
    """
    Create a Dropout layer

    :param inp: the input tensor in the computational graph
    :param drop_out: the percentage of nodes to ignore
    :return: A Tensor operation of the dropout function in the graph
    """
    return tf.compat.v1.nn.dropout(inp, rate=drop_out)


def build_generator(inp_ph):
    """
    Build the generator model computational graph

    :param inp_ph: a placeholder Tensor for the input layer of that generator model in the graph
    :return: A Tensor that can be called to preform the operations of the generator computational graph
    """
    x = linear(inp_ph, 512, "gen_linear1")
    x = lrelu(x, 0.01)
    x = linear(x, 512, "gen_linear2")
    x = lrelu(x, 0.01)
    x = linear(x, 512, "gen_linear3")
    x = lrelu(x, 0.01)
    x = linear(x, 512, "gen_linear4")
    x = lrelu(x, 0.01)
    x = linear(x, 784, "gen_linear5")
    x = sigmoid(x)

    return x


def build_discriminator(inp_ph, drop_out=0.3):
    """
    Build the discriminator model's computational graph

    :param inp_ph: a placeholder Tensor for the input layer of that discriminator model in the graph
    :param drop_out: drop out rate
    :return: A Tensor that can be called to preform the operations of the discriminator computational graph
    """
    x = linear(inp_ph, 2048, "disc_linear1")
    x = lrelu(x, 0.01)
    x = linear(x, 1024, "disc_linear2")
    x = lrelu(x, 0.01)
    x = linear(x, 512, "disc_linear3")
    x = lrelu(x, 0.01)
    x = linear(x, 1, "disc_linear4")

    return x


def save_image_grid(images, labels, nrows, ncols, epoch):
    fig, axes = plt.subplots(nrows, ncols, figsize=(25, 25))
    for i in range(nrows):
        for j in range(ncols):
            axes[i][j].imshow(images[i * ncols + j], cmap="gray")
            label_ = "Score: %f.3" % labels[i * ncols + j]
            axes[i][j].set_title(label_)

    save_path = os.path.join(RESULT_IMAGES_DIR, "results_{}.png".format(epoch))
    plt.savefig(save_path)
    plt.tight_layout()
    plt.close(fig)


def save_loss_plots(generator_train_loss, discriminator_train_loss, epoch):
    # plot losses
    plt.title("Training Losses")
    plt.plot(generator_train_loss, label="generator loss", c="blue")
    plt.plot(discriminator_train_loss, label="discriminator loss", c="orange")
    plt.legend()

    # save figure
    save_path = os.path.join(RESULT_LOSS_DIR, "loss.png")
    plt.savefig(save_path)
    plt.tight_layout()
    plt.close()


def batches(l, batch_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in xrange(0, len(l), batch_size):
        if i + batch_size < len(l):
            yield l[i:i + batch_size]
        else:
            yield l[i:]


def show_image_example(x_train):
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))

    images = random.sample(list(x_train), 64)
    for i in range(8):
        for j in range(8):
            axes[i][j].imshow(images[i * 8 + j], cmap="gray")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # get dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    show_image_example(x_train)

    # quick preprocess of the input data
    x_train = x_train / 255.0  # normalize between 0 and 1
    x_test = x_test / 255.0

    # build models
    with tf.compat.v1.variable_scope("generator") as scope:
        generator_input = tf.compat.v1.placeholder(tf.float32, [None, 100])
        generator = build_generator(generator_input)

    drop_rate = 0.3
    with tf.compat.v1.variable_scope("discriminator") as scope:
        discriminator_input = tf.compat.v1.placeholder(tf.float32, [None, 784])
        discriminator = build_discriminator(discriminator_input, drop_rate)
        scope.reuse_variables()
        generator_discriminator = build_discriminator(generator, drop_rate)

    # define loss for each model
    # discriminator
    # eps = 1e-2  # used to not get log(0)
    # discriminator_loss = tf.compat.v1.reduce_mean(
    #     -tf.compat.v1.log(discriminator + eps) - tf.compat.v1.log(1 - generator_discriminator + eps))
    discriminator_loss = tf.compat.v1.reduce_mean(generator_discriminator - discriminator)
    generator_loss = tf.compat.v1.reduce_mean(-generator_discriminator)

    # generator
    # generator_loss = tf.compat.v1.reduce_mean(-tf.compat.v1.log(generator_discriminator + eps))

    # define optimizer for each model
    t_variables = tf.compat.v1.trainable_variables()
    D_vars = [var for var in t_variables if 'disc_' in var.name]
    G_vars = [var for var in t_variables if 'gen_' in var.name]
    discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(2e-4, beta1=0.5).minimize(discriminator_loss,
                                                                                         var_list=D_vars)
    generator_optimizer = tf.compat.v1.train.AdamOptimizer(2e-4, beta1=0.5).minimize(generator_loss,
                                                                                     var_list=G_vars)

    # start session and init variables
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # training
    fixed_random_noise = np.random.normal(0, 1, (64, 100))
    epochs = 200
    save_interval = 1
    batch_size = 128
    generator_train_losses = []
    discriminator_train_losses = []
    for epoch in range(epochs):
        # train
        g_train_losses = []
        d_train_losses = []
        for x, y in tqdm(zip(batches(x_train, batch_size), batches(y_train, batch_size)),
                         desc="Epoch {}".format(epoch + 1)):
            discriminator_feed_dict = {
                generator_input: list(np.random.rand(1, 100)),
                discriminator_input: list(x.reshape(-1, 784))
            }
            generator_feed_dict = {
                generator_input: list(np.random.rand(1, 100))
            }

            # get losses and optimize parameters
            discriminator_loss_value, _ = sess.run([discriminator_loss, discriminator_optimizer],
                                                   feed_dict=discriminator_feed_dict)
            generator_loss_value, _ = sess.run([generator_loss, generator_optimizer],
                                               feed_dict=generator_feed_dict)
            d_train_losses.append(discriminator_loss_value)
            g_train_losses.append(generator_loss_value)

        # save figures of losses
        generator_train_losses.append(np.mean(g_train_losses))
        discriminator_train_losses.append(np.mean(d_train_losses))
        log(logging.INFO, "Epoch {} - Discriminator Loss: {}, Generator Loss: {}".format(epoch + 1,
                                                                                         discriminator_train_losses[-1],
                                                                                         generator_train_losses[-1]))

        save_loss_plots(generator_train_losses, discriminator_train_losses, epoch)

        # save result images and model every interval of epochs
        if epoch % save_interval == 0 or epoch == epochs - 1:
            generator_output = sess.run(generator, feed_dict={generator_input: list(fixed_random_noise)})
            discriminator_output = sess.run(discriminator, feed_dict={discriminator_input: list(generator_output)})
            output_images = generator_output.reshape(64, 28, 28) * 255.0
            discriminator_output = discriminator_output.reshape(-1)
            # images.append(output_image)
            # labels.append(discriminator_output)
            save_image_grid(output_images, discriminator_output, 8, 8, epoch)
