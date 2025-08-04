import tensorflow as tf
from transformer import BasicTransformer
from classification import LastTokenClassificationHead
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


class TransformerBinaryClassifier:
    def __init__(
        self,
        input_shape,
        n_layers=2,
        internal_size=128,
        n_heads=4,
        mlp_units=[128],
        dropout_rate=0.1,
        learning_rate=1e-3
    ):
        self.input_shape = input_shape
        self.n_layers = n_layers
        self.internal_size = internal_size
        self.n_heads = n_heads
        self.mlp_units = mlp_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        # Input layer
        input_layer = Input(shape=self.input_shape, name="record_input")

        # Transformer
        transformer = BasicTransformer(
            n_layers=self.n_layers,
            internal_size=self.internal_size,
            n_heads=self.n_heads,
            verbose=False
        )
        x = transformer.apply(input_layer, training=True)

        # Classification head (pooling)
        classification_head = LastTokenClassificationHead()
        x = classification_head.apply(x)

        # MLP head
        for i, units in enumerate(self.mlp_units):
            x = Dense(units, activation="relu", name=f"classification_mlp_{i}_{units}")(x)
            x = Dropout(self.dropout_rate)(x)

        # Output layer
        output_layer = Dense(1, activation="sigmoid", name="binary_output")(x)

        # Model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
            jit_compile=True
        )
        return model

    def summary(self):
        return self.model.summary()

    def train(
        self,
        X_train, y_train,
        X_val=None, y_val=None,
        batch_size=32,
        epochs=20,
        patience=5,
        verbose=1
    ):
        callbacks = []
        if X_val is not None and y_val is not None:
            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=0
            )
            callbacks.append(early_stop)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        return history

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)

    def get_model(self):
        return self.model
