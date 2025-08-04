import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from transformer import BasicTransformer
from classification import LastTokenClassificationHead

class PGDAdversarialTrainer(tf.keras.Model):
    def __init__(self, base_model, epsilon_range=(0.01, 0.1), alpha=0.01, num_iter=7):
        super().__init__()
        self.base_model = base_model
        self.epsilon_range = epsilon_range
        self.alpha = alpha
        self.num_iter = num_iter
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.metric = tf.keras.metrics.BinaryAccuracy()

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def pgd_attack_batch(self, x, y):
        x_adv = tf.identity(x)
        y = tf.expand_dims(tf.cast(y, tf.float32), axis=-1)
        epsilon = tf.random.uniform([], *self.epsilon_range)
        for _ in range(self.num_iter):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                preds = self.base_model(x_adv, training=True)
                loss = self.loss_fn(y, preds)
            grad = tape.gradient(loss, x_adv)
            x_adv += self.alpha * tf.sign(grad)
            x_adv = tf.clip_by_value(x_adv, x - epsilon, x + epsilon)
            x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
        return x_adv

    def train_step(self, data):
        x, y = data
        x_adv = self.pgd_attack_batch(x, y)

        x_combined = tf.concat([x, x_adv], axis=0)
        y_combined = tf.concat([y, y], axis=0)
        y_combined = tf.expand_dims(tf.cast(y_combined, tf.float32), axis=-1)

        with tf.GradientTape() as tape:
            preds = self.base_model(x_combined, training=True)
            loss = self.loss_fn(y_combined, preds)

        grads = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        self.metric.update_state(y_combined, preds)
        return {"loss": loss, "accuracy": self.metric.result()}

    def test_step(self, data):
        x, y = data
        y = tf.expand_dims(tf.cast(y, tf.float32), axis=-1)
        preds = self.base_model(x, training=False)
        loss = self.loss_fn(y, preds)
        self.metric.update_state(y, preds)
        return {"loss": loss, "accuracy": self.metric.result()}

    def call(self, inputs):
        return self.base_model(inputs)

class PGDTransformerClassifier:
    def __init__(self, input_dim, transformer_config, pgd_config, mlp_units=[128], dropout_rate=0.1):
        self.input_dim = input_dim
        self.transformer_config = transformer_config
        self.pgd_config = pgd_config
        self.mlp_units = mlp_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()

    def _build_model(self):
        input_layer = Input(shape=(1, self.input_dim), name="record_input")
        transformer = BasicTransformer(**self.transformer_config)
        x = transformer.apply(input_layer, training=True)

        classification_head = LastTokenClassificationHead()
        x = classification_head.apply(x)

        for i, units in enumerate(self.mlp_units):
            x = Dense(units, activation="relu", name=f"mlp_{i}")(x)
            x = Dropout(self.dropout_rate)(x)

        output_layer = Dense(1, activation="sigmoid", name="output")(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        return model

    def compile_and_wrap(self, optimizer):
        self.adv_model = PGDAdversarialTrainer(
            self.model,
            **self.pgd_config
        )
        self.adv_model.compile(optimizer=optimizer)

    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=64, epochs=10, patience=3):
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_data = (X_val, y_val) if X_val is not None else None
        callbacks = [EarlyStopping(patience=patience, restore_best_weights=True)]
        self.adv_model.fit(
            train_ds,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks
        )

    def evaluate(self, X_test, y_test):
        return self.adv_model.evaluate(X_test, y_test)

    def predict(self, X):
        return self.adv_model.predict(X)

    def summary(self):
        self.model.summary()
