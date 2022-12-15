import os
import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dataset import convert_batch_to_numpy

# dirs
RESULT_DIR: str = ""
RUN_DIR: str = ""
IMAGES_DIR: str = ""
CHECKPOINT_DIR: str = ""
LOSSES_DIR: str = ""
LOG_DIR: str = ""

# image shape
IMG_SHAPE = 128

# log
LOG_NAME = "train_log"

def get_logger() -> logging.Logger:
    return logging.getLogger(LOG_NAME)

def log(message):
    get_logger().info(message)

def show_train_hist(hist, show=False, save=False):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.clf()
    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        path = os.path.join(LOSSES_DIR, "discriminator_and_generator_loss.png")
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def init_vis_workspace():
    """
    create all the directories for visualization
    """
    global RESULT_DIR, RUN_DIR, IMAGES_DIR, CHECKPOINT_DIR, LOSSES_DIR
    # create main dir
    RESULT_DIR = "Results"
    if not os.path.isdir(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    # create dir for current run
    RUN_DIR = os.path.join(RESULT_DIR, "run_{}".format(len(os.listdir(RESULT_DIR))))
    if not os.path.isdir(RUN_DIR):
        os.mkdir(RUN_DIR)

    # create dir in the current run for result images
    IMAGES_DIR = os.path.join(RUN_DIR, "result_images")
    if not os.path.isdir(IMAGES_DIR):
        os.mkdir(IMAGES_DIR)

    # create dir in the current run for checkpoints of the model
    CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")
    if not os.path.isdir(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)

    # create dir in the current run for the loss graphs of the model
    LOSSES_DIR = os.path.join(RUN_DIR, "losses")
    if not os.path.isdir(LOSSES_DIR):
        os.mkdir(LOSSES_DIR)
    
    # create dir in the current run for the log of the model
    LOG_DIR = os.path.join(RUN_DIR, "logs")
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)

    # create log 
    logger = logging.getLogger(LOG_NAME)
    log_file_name = "train_log.log"
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, log_file_name), "w+")
    log_format = logging.Formatter("[%(asctime)s] - [%(levelname)s] - [%(name)s]: %(message)s")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

def save_session(session, name):
    # create dir for that session
    save_dir = os.path.join(CHECKPOINT_DIR, name)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # save session
    save_path = os.path.join(save_dir, name)
    saver = tf.compat.v1.train.Saver()
    saver.save(session, save_path)


def show_image_grid(generator, gen_placeholder, session, test_set, epoch, n_rows, n_cols, save=True, show=False):
    """
    Show image grid and save that image grid
    :param generator: Tensor - the generator tensor
    :param gen_placeholder: Tensor - the placeholder for the generator input
    :param session: Session - the session for the calculations
    :param test_set: Tuple - a tuple containing (x_test, y_test)
    :param epoch: Int current epoch
    :param n_rows: Int - number of rows
    :param n_cols: Int - number of columns
    :param save: Boolean - will save figure if True
    :param show: Boolean - will show figure if True
    """
    for x, y in zip(*test_set):
        # get images
        x_batch = convert_batch_to_numpy(x, True).reshape(-1,IMG_SHAPE * IMG_SHAPE)
        y_batch = convert_batch_to_numpy(y, False)

        # get output
        out = session.run(generator, feed_dict={gen_placeholder: x_batch})
        assert out.shape[0] >= n_cols * n_rows  # assert that we can fill the figure properly

        # create figure
        fig, axes = plt.subplots(n_rows, 2 * n_cols, figsize=(15, 15))

        # show real images
        for i in range(n_rows):
            for j in range(n_cols):
                axes[i][j].imshow(y_batch[i * n_cols + j])

        # show generated images
        for i in range(n_rows):
            for j in range(n_cols, 2 * n_cols):
                axes[i][j].imshow((out[i * n_cols + (j - n_cols)].reshape(IMG_SHAPE,IMG_SHAPE, 3) * 255.0).astype(np.uint8))

        # save and show figure
        fig.tight_layout()
        if save:
            save_path = os.path.join(IMAGES_DIR, "epoch_{}.png".format(epoch + 1))
            fig.savefig(save_path)

        if show:
            fig.show()

        plt.close(fig)
        break


if __name__ == '__main__':
    pass
