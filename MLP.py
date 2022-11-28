import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from urllib3.connectionpool import xrange

tf.compat.v1.disable_v2_behavior()

# init work environment
RESULT_IMAGES_DIR = "result_images"
RESULT_LOSS_DIR = "result_losses"
CHECKPOINTS_DIR = "checkpoints"
TENSORBOARD_LOG_DIR = "tensorboard_log_dir"

if not os.path.isdir(RESULT_IMAGES_DIR):
    os.mkdir(RESULT_IMAGES_DIR)

if not os.path.isdir(RESULT_LOSS_DIR):
    os.mkdir(RESULT_LOSS_DIR)

if not os.path.isdir(CHECKPOINTS_DIR):
    os.mkdir(CHECKPOINTS_DIR)

if not os.path.isdir(TENSORBOARD_LOG_DIR):
    os.mkdir(TENSORBOARD_LOG_DIR)


def linear(inp, out_features: int, name: str):
    """
    Create a linear layer

    :param inp: the input tensor in the computational graph
    :param out_features: the number of output features
    :param name: the name of the layer
    :return: A Tensor representing the linear layer operation in the graph
    """
    weights = tf.compat.v1.get_variable(name, dtype=tf.float32,
                                        initializer=tf.random.uniform((inp.shape[-1], out_features), -1.0, 1.0),
                                        trainable=True,
                                        use_resource=True)
    bias = tf.compat.v1.get_variable(name + "_bias", [out_features],
                                     initializer=tf.constant_initializer(0.),
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


def dropout(inp, drop_out: float):
    """
    Create a Dropout layer

    :param inp: the input tensor in the computational graph
    :param drop_out: the percentage of nodes to ignore
    :return: A Tensor operation of the dropout function in the graph
    """
    return tf.compat.v1.nn.dropout(inp, rate=0.3)


def build_generator(inp_ph):
    """
    Build the generator model computational graph

    :param inp_ph: a placeholder Tensor for the input layer of that generator model in the graph
    :return: A Tensor that can be called to preform the operations of the generator computational graph
    """
    with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE) as scope:
        x = linear(inp_ph, 256, "gen_linear1")
        x = relu(x)
        x = linear(x, 512, "gen_linear2")
        x = relu(x)
        x = linear(x, 1024, "gen_linear3")
        x = relu(x)
        x = linear(x, 784, "gen_linear4")
        x = sigmoid(x)

    return x


def build_discriminator(inp_ph, drop_out=0.3):
    """
    Build the discriminator model computational graph

    :param inp_ph: a placeholder Tensor for the input layer of that discriminator model in the graph
    :param drop_out: drop out rate
    :return: A Tensor that can be called to preform the operations of the discriminator computational graph
    """
    with tf.compat.v1.variable_scope("discriminator", reuse=tf.compat.v1.AUTO_REUSE) as scope:
        x = linear(inp_ph, 128, "disc_linear1")
        x = relu(x)
        x = dropout(x, drop_out)
        x = linear(x, 1024, "disc_linear2")
        x = relu(x)
        x = dropout(x, drop_out)
        x = linear(x, 512, "disc_linear3")
        x = relu(x)
        x = dropout(x, drop_out)
        x = linear(x, 256, "disc_linear4")
        x = relu(x)
        x = dropout(x, drop_out)
        x = linear(x, 1, "disc_linear5")
        x = sigmoid(x)

    return x


def save_image_grid(images, nrows, ncols, epoch):
    fig, axes = plt.subplots(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            axes[i][j].imshow(images[i * ncols + j], cmap="gray")

    save_path = os.path.join(RESULT_IMAGES_DIR, "results_{}.png".format(epoch))
    plt.savefig(save_path)
    plt.close(fig)


def save_loss_plots(generator_train_loss, discriminator_train_loss, generator_valid_loss, discriminator_valid_loss,
                    epoch):
    # create figure
    fig, (generator_axis, discriminator_axis) = plt.subplots(1, 2)

    # plot losses
    generator_axis.set_title("Training Losses")
    generator_axis.plot(generator_train_loss, label="generator loss", c="blue")
    generator_axis.plot(discriminator_train_loss, label="discriminator loss", c="orange")
    generator_axis.legend()

    discriminator_axis.set_title("Validation Losses")
    discriminator_axis.plot(generator_valid_loss, label="generator loss", c="blue")
    discriminator_axis.plot(discriminator_valid_loss, label="discriminator loss", c="orange")
    discriminator_axis.legend()

    # save figure
    save_path = os.path.join(RESULT_LOSS_DIR, "losses_{}".format(epoch))
    plt.savefig(save_path)
    plt.close(fig)


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


if __name__ == '__main__':
    # check eager execution
    print("Executing Eagerly: ", tf.executing_eagerly())

    # get dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # quick preprocess of the input data
    x_train = ((x_train / 255.0) - 0.5) / 0.5

    # set placeholders for the models
    generator_input = tf.compat.v1.placeholder(tf.float32, [None, 100])
    discriminator_input = tf.compat.v1.placeholder(tf.float32, [None, 784])
    discriminator_real_label = tf.compat.v1.placeholder(tf.float32, [None, 1])
    discriminator_fake_label = tf.compat.v1.placeholder(tf.float32, [None, 1])
    generator_label = tf.compat.v1.placeholder(tf.float32, [None, 1])

    # build models
    generator = build_generator(generator_input)
    discriminator = build_discriminator(discriminator_input)
    generator_discriminator = build_discriminator(generator)

    # define loss for each model
    # discriminator
    # discriminator_real_loss = tf.compat.v1.reduce_mean(tf.losses.binary_crossentropy(discriminator_real_label,
    #                                                                                  discriminator))
    # discriminator_fake_loss = tf.compat.v1.reduce_mean(tf.losses.binary_crossentropy(discriminator_fake_label,
    #                                                                                  generator_discriminator))
    # discriminator_loss = discriminator_real_loss + discriminator_fake_loss
    # discriminator_loss = tf.compat.v1.reduce_mean(
    #     tf.compat.v1.log(discriminator) + tf.compat.v1.log(1 - generator_discriminator))

    # loss for each network
    eps = 1e-2
    discriminator_loss = tf.compat.v1.reduce_mean(
        -tf.compat.v1.log(discriminator + eps) - tf.compat.v1.log(1 - generator_discriminator + eps))
    generator_loss = tf.compat.v1.reduce_mean(-tf.compat.v1.log(generator_discriminator + eps))

    # generator
    # generator_loss = tf.losses.binary_crossentropy(generator_label,
    #                                                generator_discriminator)
    # generator_loss = tf.compat.v1.reduce_mean(tf.compat.v1.log(1 - generator_discriminator))

    # define optimizer for each model
    t_variables = tf.compat.v1.trainable_variables()
    D_vars = [var for var in t_variables if 'disc_' in var.name]
    G_vars = [var for var in t_variables if 'gen_' in var.name]
    discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(2e-4).minimize(discriminator_loss, var_list=D_vars)
    generator_optimizer = tf.compat.v1.train.AdamOptimizer(2e-4).minimize(generator_loss, var_list=G_vars)

    # start session and init variables
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # training
    fixed_random_noise = np.random.rand(1, 100)
    epochs = 100
    save_interval = 5
    batch_size = 100
    generator_train_losses = []
    discriminator_train_losses = []
    generator_valid_losses = []
    discriminator_valid_losses = []
    for epoch in range(epochs):
        # train
        g_train_losses = []
        d_train_losses = []
        for x, y in tqdm(zip(batches(x_train, batch_size), batches(y_train, batch_size)),
                         desc="Epoch {}".format(epoch + 1)):
            discriminator_feed_dict = {
                generator_input: list(np.random.rand(1, 100)),
                discriminator_input: list(x.reshape(-1, 784)),
                discriminator_fake_label: list(np.zeros((1, 1))),
                discriminator_real_label: list(np.ones((1, 1)))
            }
            generator_feed_dict = {
                generator_input: list(np.random.rand(1, 100)),
                generator_label: list(np.ones((1, 1)))
            }

            # get losses
            discriminator_loss_value = sess.run(discriminator_loss, feed_dict=discriminator_feed_dict)
            generator_loss_value = sess.run(generator_loss, feed_dict=generator_feed_dict)
            d_train_losses.append(discriminator_loss_value)
            g_train_losses.append(generator_loss_value)

            # train models
            sess.run(discriminator_optimizer, feed_dict=discriminator_feed_dict)
            sess.run(generator_optimizer, feed_dict=generator_feed_dict)

        # validate
        g_valid_losses = []
        d_valid_losses = []
        for x, y in tqdm(zip(batches(x_test, batch_size), batches(y_test, batch_size)),
                         desc="Validation Epoch".format(epoch + 1)):
            discriminator_feed_dict = {
                generator_input: list(np.random.rand(1, 100)),
                discriminator_input: list(x.reshape(-1, 784)),
                discriminator_fake_label: list(np.zeros((1, 1))),
                discriminator_real_label: list(np.ones((1, 1)))
            }
            generator_feed_dict = {
                generator_input: list(np.random.rand(1, 100)),
                generator_label: list(np.ones((1, 1)))
            }

            # get losses
            discriminator_loss_value = sess.run(discriminator_loss, feed_dict=discriminator_feed_dict)
            generator_loss_value = sess.run(generator_loss, feed_dict=generator_feed_dict)
            d_valid_losses.append(discriminator_loss_value)
            g_valid_losses.append(generator_loss_value)

        generator_train_losses.append(np.average(g_train_losses))
        discriminator_train_losses.append(np.average(d_train_losses))
        generator_valid_losses.append(np.average(g_valid_losses))
        discriminator_valid_losses.append(np.average(d_valid_losses))

        # save figures of losses
        save_loss_plots(generator_train_losses, discriminator_train_losses, generator_valid_losses,
                        discriminator_valid_losses, epoch)

        # save result images and model every interval of epochs
        if epoch % save_interval == 0 or epoch == epochs - 1:
            images = []
            for i in range(64):
                generator_output = sess.run(generator, feed_dict={generator_input: list(fixed_random_noise)})
                output_image = generator_output.reshape(28, 28) * 255.0
                images.append(output_image)

            save_image_grid(images, 8, 8, epoch)
