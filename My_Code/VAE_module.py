import tensorflow as tf
from tensorflow.keras import layers, Model

class VAE(Model):
    def __init__(self, input_dim, latent_dim=32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder network
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(latent_dim * 2),  # outputs mean and log variance
        ])

        # Decoder network
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid'),  # sigmoid for binary data
        ])

    def encode(self, x):
        z = self.encoder(x)
        z_mean, z_log_var = tf.split(z, num_or_size_splits=2, axis=1)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return eps * tf.exp(z_log_var * 0.5) + z_mean

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decode(z)
        return x_recon

    def compute_loss(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decode(z)

        # Binary crossentropy reconstruction loss summed over features
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(x, x_recon), axis=1
            )
        )

        # KL divergence
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )

        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss

    def train_step(self, data):
        x = tf.cast(data, tf.float32)
        with tf.GradientTape() as tape:
            total_loss, recon_loss, kl_loss = self.compute_loss(x)
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.compiled_metrics.update_state(x, self.call(x))

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data):
        x = tf.cast(data, tf.float32)
        total_loss, recon_loss, kl_loss = self.compute_loss(x)

        self.compiled_metrics.update_state(x, self.call(x))

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
