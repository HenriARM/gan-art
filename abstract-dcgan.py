import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, Flatten, \
    Dropout
import matplotlib.pyplot as plt

# import glob
# import imageio
# import numpy as np
import os
# import PIL
import time
from IPython import display


BATCH_SIZE = 32
ce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# train_dataset = tf.data.Dataset.list_files(file_pattern='./dataset/*.jpg').batch(BATCH_SIZE)

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(directory='./dataset-transformed-full',
                                                                    image_size=(28, 28),
                                                                    batch_size=BATCH_SIZE)


def make_generator_model() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        # can use model.build(input_shape), before running model.summary() or training
        Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)),
        # Dense(32 * 32 * 256, use_bias=False, input_shape=(100,)),
        BatchNormalization(),
        LeakyReLU(),

        Reshape((7, 7, 256)),
        # Reshape((32, 32, 256)),

        Conv2DTranspose(filters=128, kernel_size=5, strides=1, padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(filters=3, kernel_size=5, strides=2, padding='same', use_bias=False, activation='tanh'),
    ])


def make_discriminator_model() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        # When padding='some' and stride=2, calculate padding as if input=output and stride=1.
        # Then use that padding with stride 2, and you will get exactly output 2x smaller than input
        Conv2D(filters=64, kernel_size=5, strides=2, padding='same', input_shape=[28, 28, 3]),
        LeakyReLU(),
        Dropout(0.3),

        Conv2D(filters=128, kernel_size=5, strides=2, padding='same'),
        LeakyReLU(),
        Dropout(0.3),

        Flatten(),
        Dense(1)
    ])


def discriminator_loss(real_output, fake_output):
    # discriminator is a binary classifier, which should return for fake image 0, and 1 for real
    # best loss function for it would be Cross-Entropy
    # binary CE = y_i * log(p(y_i)) + y_i * log(1 - p(y_i)),
    # y_i - true distribution of i-th class (one-hot vector (1,0) or (0,1))
    real_loss = ce(y_true=tf.ones_like(real_output), y_pred=real_output)
    fake_loss = ce(y_true=tf.zeros_like(fake_output), y_pred=fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    return ce(tf.ones_like(fake_output), fake_output)


generator = make_generator_model()
discriminator = make_discriminator_model()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)


    # https://www.tensorflow.org/api_docs/python/tf/GradientTape
    # GradientTape used to calculate gradient
    # e.x. with tf.GradientTape() as g:
    #       g.watch(x)
    #       y = x * x
    #   dy_dx = g.gradient(y, x)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # backpropogation
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def generate_and_save_images(model, epoch, noise):
    generated_images = model(noise, training=False)
    plt.figure(figsize=(4, 4))
    for i in range(generated_images.shape[0]):
        plt.subplot(4, 4, i + 1)

        img = generated_images[i] * 127.5 + 127.5
        img = img.numpy().astype(int)  # 0..255 only as int, 0..1 as float
        plt.imshow(img)
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# # Display a single image using the epoch number
# def display_image(epoch_no):
#   return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


def main():
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    EPOCHS = 20
    noise_dim = 100

    # We will reuse this seed overtime (so it's easier) to visualize progress in the animated GIF)
    # its actually used to print each time during training same noise, to see how generator learns
    seed = tf.random.normal([16, noise_dim])

    for epoch in range(EPOCHS):
        start = time.time()
        for image_batch in train_dataset:
            # dataset returns empty y_train as second tuple variable, omit
            image_batch = image_batch[0]
            # normalize the images to [0, 1]
            # image_batch /= 255
            image_batch = (image_batch - 128) / 128
            train_step(image_batch)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every epoch
        checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, EPOCHS, seed)


if __name__ == '__main__':
    main()
