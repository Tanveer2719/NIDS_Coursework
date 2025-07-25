import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

class Attack:

    @staticmethod
    def evaluate_metrics(y_true, y_pred):
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        far = fp / (fp + tn + 1e-10)

        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'far': far
        }

    @staticmethod
    def fgsm_attack_tf(model, inputs, labels, epsilon=0.05):
        inputs = tf.convert_to_tensor(inputs)
        labels = tf.expand_dims(tf.convert_to_tensor(labels, dtype=tf.float32), axis=-1)

        with tf.GradientTape() as tape:
            tape.watch(inputs)
            predictions = model(inputs, training=False)
            loss = tf.keras.losses.binary_crossentropy(labels, predictions)

        gradients = tape.gradient(loss, inputs)
        if gradients is None:
            raise ValueError("Gradient computation failed.")

        perturbation = epsilon * tf.sign(gradients)
        adv_inputs = inputs + perturbation
        adv_inputs = tf.clip_by_value(adv_inputs, clip_value_min=0.0, clip_value_max=1.0)

        return adv_inputs

    @classmethod
    def perform_fgsm(cls, model, x_df, y_df, epsilon=0.05):
        x_tf = tf.convert_to_tensor(x_df, dtype=tf.float32)
        y_tf = tf.convert_to_tensor(y_df, dtype=tf.float32)

        x_adv = cls.fgsm_attack_tf(model, x_tf, y_tf, epsilon=epsilon)

        y_pred_clean = (model.predict(x_tf) > 0.5).astype(int).flatten()
        y_pred_adv = (model.predict(x_adv) > 0.5).astype(int).flatten()
        y_true = y_df.flatten()

        clean_metrics = cls.evaluate_metrics(y_true, y_pred_clean)
        adv_metrics = cls.evaluate_metrics(y_true, y_pred_adv)

        return {
            'epsilon': epsilon,
            'clean': clean_metrics,
            'adversarial': adv_metrics
        }

    @classmethod
    def perform_fgsm_batch(cls, model, x_df, y_df, epsilon_list):
        x_tf = tf.convert_to_tensor(x_df, dtype=tf.float32)
        y_tf = tf.convert_to_tensor(y_df, dtype=tf.float32)

        f1_scores = []

        for eps in epsilon_list:
            x_adv = cls.fgsm_attack_tf(model, x_tf, y_tf, epsilon=eps)
            y_pred_adv = (model.predict(x_adv) > 0.5).astype(int).flatten()
            f1 = f1_score(y_df, y_pred_adv)
            f1_scores.append((eps, f1))

        return f1_scores
