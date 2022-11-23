import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

tf.compat.v1.disable_v2_behavior()


# create a linear layer
def linear(inp, out_features: int):
    weights = tf.compat.v1.Variable(tf.random.uniform((inp.shape[-1], out_features,), -1.0, 1.0))
    return tf.matmul(inp, weights)


# create a sigmoid layer
def sigmoid(inp):
    return tf.compat.v1.sigmoid(inp)


# create lrelu layer
def lrelu(inp, alpha):
    return tf.compat.v1.nn.leaky_relu(inp, alpha)


def build_generator(inp_ph):
    x = linear(inp_ph, 128)
    x = lrelu(x, 0.2)
    x = linear(x, 784)
    x = sigmoid(x)

    return x


def build_discriminator(inp_ph, reuse=True):
    with tf.compat.v1.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        x = linear(inp_ph, 128)
        x = lrelu(x, 0.2)
        x = linear(x, 1)
        x = sigmoid(x)

    return x


if __name__ == '__main__':
    # get dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # set placeholders for the models
    generator_input = tf.compat.v1.placeholder(tf.float32, [None, 784])
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
    discriminator_real_loss = tf.compat.v1.reduce_mean(tf.losses.binary_crossentropy(discriminator_real_label,
                                                                                     discriminator))
    discriminator_fake_loss = tf.compat.v1.reduce_mean(tf.losses.binary_crossentropy(discriminator_fake_label,
                                                                                     generator_discriminator))
    discriminator_loss = discriminator_real_loss + discriminator_fake_loss
    # discriminator_loss = tf.compat.v1.reduce_mean(
    #     tf.compat.v1.log(discriminator) + tf.compat.v1.log(1 - generator_discriminator))

    # generator
    generator_loss = tf.losses.binary_crossentropy(generator_label,
                                                   generator_discriminator)
    # generator_loss = tf.compat.v1.reduce_mean(tf.compat.v1.log(1 - generator_discriminator))

    # define optimizer for each model
    discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(discriminator_loss)
    generator_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(generator_loss)

    # start session and init variables
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # training
    epochs = 100
    epoch_loss_show_interval = 1
    for epoch in range(epochs):
        for x, y in tqdm(zip(x_train, y_train), desc="Epoch {}".format(epoch + 1)):
            discriminator_feed_dict = {
                generator_input: list(np.random.rand(1, 784)),
                discriminator_input: list(x.reshape(-1, 784) / 255.0),
                discriminator_fake_label: list(np.zeros((1, 1))),
                discriminator_real_label: list(np.ones((1, 1)))
            }
            generator_feed_dict = {
                generator_input: list(np.random.rand(1, 784)),
                generator_label: list(np.ones((1, 1)))
            }

            # train models
            sess.run(discriminator_optimizer, feed_dict=discriminator_feed_dict)
            sess.run(generator_optimizer, feed_dict=generator_feed_dict)

        if epoch % epoch_loss_show_interval == 0 or epoch == epochs - 1:
            generator_loss_value = sess.run(generator_loss, generator_feed_dict)
            discriminator_loss_value = sess.run(discriminator_loss, discriminator_feed_dict)
            print("Epoch: {}\n".format(epoch + 1) +
                  "\tGenerator loss: {}\n".format(generator_loss_value) +
                  "\tDiscriminator loss: {}".format(discriminator_loss_value))

    out = sess.run(generator, feed_dict={generator_input: list(np.random.rand(1, 784))})

    out = out.reshape(28, 28) * 255.0
    plt.imshow(out)
    plt.show()
