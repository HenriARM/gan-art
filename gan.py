import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt

import glob
import imageio
# import numpy as np
# import PIL
import time
import numpy as np
# from discriminator import Discriminator, get_discriminator
# from generator import Generator, get_generator

from dcgandiscriminator import get_discriminator
from dcgangenerator import get_generator


# =========================  GLOBAL VARIABLES  ================================================
tf.debugging.set_log_device_placement(False)

EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

IMAGE_H = 64
IMAGE_W = 64
IMAGE_C = 3

CE = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# ========================= DATA EXTRACTION ================================================

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    return tf.image.resize(img, [IMAGE_H, IMAGE_W])


def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, file_path


def load_dataset():
    root = pathlib.Path('./datasets/fauvism64-transformed')
    dataset = tf.data.Dataset.list_files(file_pattern=str(root / '*.png'), shuffle=True)
    dataset = dataset.map(map_func=process_path)
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    return dataset


# dataset = None
dataset = load_dataset()


# =========================  GAN FUNCTIONS  ================================================

def discriminator_loss(real_output, fake_output):
    """
    :param real_output:
    :param fake_output:
    :return:

    Discriminator is basically a binary classifier, which should return 0 for fake image and 1 for real image
    Best loss function for it would be Cross-Entropy
    binary CE = y_i * log(p(y_i)) + y_i * log(1 - p(y_i)),
    y_i - true distribution of i-th class (one-hot vector (1,0) or (0,1))
    """
    real_loss = CE(y_true=tf.ones_like(real_output), y_pred=real_output)
    fake_loss = CE(y_true=tf.zeros_like(fake_output), y_pred=fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    """

    :param fake_output:
    :return:

    The generator loss function measure how well the generator was able to trick the discriminator:
    If discriminator will be tricked successfully, it will return 1 (real image)
    """
    return CE(tf.ones_like(fake_output), fake_output)


def train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images):
    noise = tf.random.normal([BATCH_SIZE, 1, 1, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        """
        https://www.tensorflow.org/api_docs/python/tf/GradientTape
        GradientTape used to calculate gradient
        e.x. with tf.GradientTape() as g:
              g.watch(x)
              y = x * x
          dy_dx = g.gradient(y, x)
        """
        # TODO: add checks tf.debugging.assert_equal(logits.shape, (32, 10))

        # TODO: training = True
        generated_images = generator(noise)

        real_output = discriminator(images)
        fake_output = discriminator(generated_images)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # backpropogation
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# =========================  IMAGE  ================================================


def generate_and_save_images(model, epoch, noise):
    generated_images = model(noise, training=False)
    plt.figure(figsize=(4, 4))
    for i in range(generated_images.shape[0]):
        plt.subplot(4, 4, i + 1)

        img = generated_images[i] * 127.5 + 127.5
        # img = generated_images[i] * 255
        img = img.numpy().astype(np.uint8)  # matplotlib understand images with values 0..255 only as int, 0..1 as float
        plt.imshow(img)
        plt.axis('off')
    plt.savefig('./images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def generate_gif():
    anim_file = './dcgan.gif'
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('./images/image*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


# ==================================================================================

def main():
    # Manual Chekpoint configuration
    # generator = Generator()
    # discriminator = Discriminator()
    generator = get_generator()
    discriminator = get_discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Checkpoints capture the exact value of all parameters (tf.Variable objects) used by a model.
    checkpoint_dir = './training_checkpoints'
    checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0),
                                     generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    # Training
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    noise_dim = 100

    # We will reuse this seed overtime (so it's easier) to visualize progress in the animated GIF)
    # its actually used to print each time during training same noise, to see how generator learns
    seed = tf.random.normal([16, 1, 1, noise_dim])
    # seed = tf.random.normal([16, noise_dim])

    # used to continue training
    last_epoch = int(checkpoint.epoch.numpy())
    # TODO: add check for epoch < EPOCHS
    for epoch in range(last_epoch, EPOCHS):
        start = time.time()
        for image_batch, label in dataset:
            # normalize the images to [0, 1]
            # images = image_batch / 255

            # normalize the images to [-1, 1]
            images = (image_batch - 127.5) / 127.5
            train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images)

        # Save Generator Architecture with weights and optimizers
        # Since we assign only one constant path - each epoch model will be overwritten
        generator.save('./models/model')

        # increase epoch
        checkpoint.epoch.assign_add(1)
        save_path = manager.save()
        print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

        generate_and_save_images(generator, epoch, seed)
        print('Time for epoch {} is {} sec'.format(epoch, time.time() - start))


if __name__ == '__main__':
    # main()
    generate_gif()
    # try:
    #     with tf.device('/device:GPU:0'):
    #         main()
    # except RuntimeError as e:
    #     print(e)
