from math import sqrt

import matplotlib.pyplot as plt
import tensorflow as tf
import time

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def generate_and_save_image(generator, file_name, test_input, figures, cmap):
  predictions = generator(test_input, training=False)

  figsize = int(sqrt(figures))
  plt.figure(figsize=(figsize, figsize))

  for i in range(predictions.shape[0]):
      print(predictions[0, :, :])
      plt.subplot(figsize, figsize, i+1)
      plt.imshow((predictions[i, :, :] + 1) / 2, cmap=cmap)
      plt.axis('off')

  plt.savefig(file_name)


def load(checkpoint, checkpoint_dir) -> None:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def build_checkpoint(gan) -> tf.train.Checkpoint:
    return tf.train.Checkpoint(generator_optimizer=gan.generator_optimizer,
                                     discriminator_optimizer=gan.discriminator_optimizer,
                                     generator=gan.generator,
                                     discriminator=gan.discriminator)


def generate(output_dir: str, gan, checkpoint_prefix: str, do_train=False):
    checkpoint = build_checkpoint(gan)

    if do_train:
        @tf.function
        def train_step(images):
            noise = tf.random.normal([gan.BATCH_SIZE, gan.noise_dim])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = gan.generator(noise, training=True)

                real_output = gan.discriminator(images, training=True)
                fake_output = gan.discriminator(generated_images, training=True)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, gan.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, gan.discriminator.trainable_variables)

            gan.generator_optimizer.apply_gradients(zip(gradients_of_generator, gan.generator.trainable_variables))
            gan.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, gan.discriminator.trainable_variables))

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
                generate_and_save_image(gan.generator, output_dir + 'epoch-{0}.png'.format(epoch), gan.seed, gan.num_examples_to_generate, gan.cmap)

                print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        train(gan.train_dataset, gan.EPOCHS, checkpoint, checkpoint_prefix, output_dir)
    else:
        load(checkpoint, checkpoint_prefix)
    generate_and_save_image(gan.generator, output_dir + 'final.png', gan.seed, gan.num_examples_to_generate, gan.cmap)
