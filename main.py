import time

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# Source: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


generator = make_generator_model()
discriminator = make_discriminator_model()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 1
seed = tf.random.normal([num_examples_to_generate, noise_dim])

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def generate_and_save_image(generator, file_name, test_input):
  predictions = generator(test_input, training=False)

  for i in range(predictions.shape[0]):
      plt.subplot(1, 1, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig(file_name)


def train(dataset, epochs, checkpoint, checkpoint_prefix, output_dir: str):
  for epoch in range(epochs):
    start = time.time()

    step = 0
    for image_batch in dataset:
      step += 1
      train_step(image_batch)
      print('Train step {0} of {1} done'.format(step, len(dataset)))

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    # Save image at epoch
    generate_and_save_image(generator, output_dir + 'epoch-{0}.png'.format(epoch), seed)

    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


def load(checkpoint, checkpoint_dir) -> None:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def save(checkpoint_prefix) -> tf.train.Checkpoint:
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    checkpoint.save(file_prefix = checkpoint_prefix)
    return checkpoint


def generate(output_dir: str, checkpoint: tf.train.Checkpoint, checkpoint_prefix: str, do_train=False):
    if do_train:
        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        train(train_dataset, EPOCHS, checkpoint, checkpoint_prefix, output_dir)

    generate_and_save_image(generator, output_dir + 'final.png', seed)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GAN')
    parser.add_argument('--train', type=bool, nargs=1, help='do training')
    parser.add_argument('--checkpoint', type=str, nargs=1, help='path to checkpoints')
    parser.add_argument('--output', type=str, nargs=1, help='output file name')
    args = parser.parse_args()

    checkpoint_prefix = args.checkpoint[0]
    generate(args.output[0], save(checkpoint_prefix), checkpoint_prefix, args.train[0])
