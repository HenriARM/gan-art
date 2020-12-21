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

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# Normalize the images to [-1, 1]
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = 60000
BATCH_SIZE = 256
ce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# TODO: use Bilinear Upscaling as Evalds told insted of Transpose
# TODO: find fastes solution how to make 2D picture from random noise


def make_generator_model() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        # can use model.build(input_shape), before running model.summary() or training
        Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)),
        BatchNormalization(),
        LeakyReLU(),

        Reshape((7, 7, 256)),

        Conv2DTranspose(filters=128, kernel_size=5, strides=1, padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(filters=1, kernel_size=5, strides=2, padding='same', use_bias=False, activation='tanh'),
    ])


def make_discriminator_model() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        # TODO: padding some + stride = 2, just calculate the padding as input=output same and stride=1.
        #  Then use that padding with stride 2, and you will get exactly output 2x smaller than input
        Conv2D(filters=64, kernel_size=5, strides=2, padding='same', input_shape=[28, 28, 1]),
        LeakyReLU(),
        Dropout(0.3),

        Conv2D(filters=128, kernel_size=5, strides=2, padding='same'),
        LeakyReLU(),
        Dropout(0.3),

        Flatten(),
        Dense(1)
    ])


def discriminator_loss(real_output, fake_output):
    # TODO: why we compare real images with one and fake ones with zero
    real_loss = ce(y_true=tf.ones_like(real_output), y_pred=real_output)
    fake_loss = ce(y_true=tf.zeros_like(fake_output), y_pred=fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    return ce(tf.ones_like(fake_output), fake_output)

# TODO: dont like that these vars are global
generator = make_generator_model()
discriminator = make_discriminator_model()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    # TODO: noise dim - 100
    noise = tf.random.normal([BATCH_SIZE, 100])

    # TODO: GradientTape
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # TODO: understand this
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        # TODO: why we add 127? maybe 256 x prediction
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# # Display a single image using the epoch number
# def display_image(epoch_no):
#   return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


def main():
    # # TODO: add comments on each line
    # noise = tf.random.normal([1, 100])
    # generator = make_generator_model()
    # # generator.summary()
    # # run call() of Sequential model.
    # # Don't need to run model on training mode, since not interested in layer training and gradient storing
    # generated_image = generator.call(inputs=noise, training=False)
    # # plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    # # plt.show()
    #
    # discriminator = make_discriminator_model()
    # discriminator.summary()
    # decision = discriminator.call(generated_image)
    # print(decision)


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
    # TODO
    # train(train_dataset, EPOCHS)
    # display_image(EPOCHS)
    main()
