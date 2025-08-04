import tensorflow as tf
from tensorflow.keras import layers, Model

class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = layers.Conv1D(64, kernel_size=1, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(128, kernel_size=1, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv1D(64, kernel_size=1, padding='same', activation='relu')
        self.bn3 = layers.BatchNormalization()
        self.output_layer = layers.Conv1D(64, kernel_size=1, padding='same', activation='linear')

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        return self.output_layer(x)
        
class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = layers.Conv1D(64, kernel_size=1, activation='relu', padding='same')
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')  # Binary classifier

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.global_pool(x)
        x = self.dense1(x)
        return self.output_layer(x)

class GANPurifier(Model):
    def __init__(self, generator, discriminator, lambda_adv=0.5):
        super(GANPurifier, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.lambda_adv = lambda_adv

        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.l1 = tf.keras.losses.MeanAbsoluteError()
        self.d_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.g_optimizer = tf.keras.optimizers.Adam(1e-4)

    def compile(self, **kwargs):
        super().compile(**kwargs)

    def train_step(self, data):
        x_clean, x_adv = data

        with tf.GradientTape(persistent=True) as tape:
            x_fake = self.generator(x_adv, training=True)
            real_pred = self.discriminator(x_clean, training=True)
            fake_pred = self.discriminator(x_fake, training=True)

            # Discriminator Loss
            d_loss_real = self.bce(tf.ones_like(real_pred), real_pred)
            d_loss_fake = self.bce(tf.zeros_like(fake_pred), fake_pred)
            d_loss = d_loss_real + d_loss_fake

            # Generator Loss
            recon_loss = self.l1(x_clean, x_fake)
            g_adv_loss = self.bce(tf.ones_like(fake_pred), fake_pred)
            g_loss = recon_loss + self.lambda_adv * g_adv_loss

            # Cosine Similarity (as positive similarity, not loss)
            x_clean_flat = tf.squeeze(x_clean, axis=1)
            x_fake_flat = tf.squeeze(x_fake, axis=1)
            cos_sim = tf.reduce_mean(
                tf.keras.losses.cosine_similarity(x_clean_flat, x_fake_flat)
            ) * -1.0  # invert to get actual similarity

        # Apply gradients
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "reconstruction_loss": recon_loss,
            "cosine_similarity": cos_sim
        }


