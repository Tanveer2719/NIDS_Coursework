import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# ========================
# 🔧 Helper Functions
# ========================


def fgsm_attack_tf(model, inputs, labels, epsilon=0.05, verbose=True):
    if verbose:
        print("Converting inputs and labels to tensors...")

    inputs = tf.convert_to_tensor(inputs)
    labels = tf.expand_dims(tf.convert_to_tensor(labels, dtype=tf.float32), axis=-1)

    if verbose:
        print(f"Inputs shape: {inputs.shape}")
        print(f"Labels shape: {labels.shape}")
        print("Starting FGSM attack...")

    with tf.GradientTape() as tape:
        tape.watch(inputs)

        if verbose:
            print("Running forward pass through the model...")
        predictions = model(inputs, training=False)

        if verbose:
            print("Calculating binary crossentropy loss...")
        loss = tf.keras.losses.binary_crossentropy(labels, predictions)

    if verbose:
        print("Computing gradients of loss w.r.t. inputs...")
    gradients = tape.gradient(loss, inputs)

    if gradients is None:
        raise ValueError("Gradient computation failed. Check model/trainable inputs.")

    if verbose:
        print("Generating perturbations using sign of gradients...")
    signed_grad = tf.sign(gradients)
    perturbation = epsilon * signed_grad

    if verbose:
        print(f"Applying perturbation with epsilon = {epsilon}")
    adv_inputs = inputs + perturbation

    if verbose:
        print("Clipping adversarial inputs to range [0.0, 1.0]")
    adv_inputs = tf.clip_by_value(adv_inputs, clip_value_min=0.0, clip_value_max=1.0)

    if verbose:
        max_change = tf.reduce_max(tf.abs(adv_inputs - inputs)).numpy()
        print(f"Adversarial samples created. Max feature delta: {max_change:.4f}")

    return adv_inputs

def pgd_attack_tf(model, x, y, epsilon=0.05, alpha=0.01, num_iter=10, clip_min=0.0, clip_max=1.0):
    x_adv = tf.identity(x)
    y = tf.expand_dims(tf.cast(y, tf.float32), axis=-1)

    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            pred = model(x_adv, training=False)
            loss = tf.keras.losses.binary_crossentropy(y, pred)
        grad = tape.gradient(loss, x_adv)
        x_adv = x_adv + alpha * tf.sign(grad)
        x_adv = tf.clip_by_value(x_adv, x - epsilon, x + epsilon)
        x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)

    return x_adv


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


class FGSM:

    def __init__(self, model):
        self.model = model

    def perform_fgsm(self, x_df, y_df, epsilon=0.05,verbose=True):
        x_tf = tf.convert_to_tensor(x_df, dtype=tf.float32)
        y_tf = tf.convert_to_tensor(y_df, dtype=tf.float32)

        # Generate adversarial examples
        x_adv = fgsm_attack_tf(self.model, x_tf, y_tf, epsilon=epsilon,verbose=verbose)

        # Predictions on clean and adversarial inputs
        y_pred_clean = (self.model.predict(x_tf) > 0.5).astype(int).flatten()
        y_pred_adv = (self.model.predict(x_adv) > 0.5).astype(int).flatten()
        
        y_true = y_tf.numpy()

        # Evaluate metrics
        clean_metrics = evaluate_metrics(y_true, y_pred_clean)
        adv_metrics = evaluate_metrics(y_true, y_pred_adv)

        return {
            'epsilon': epsilon,
            'clean': clean_metrics,
            'adversarial': adv_metrics
        }

    def perform_fgsm_batch(self, x_df, y_df, epsilon_list,verbose=True):
        x_tf = tf.convert_to_tensor(x_df, dtype=tf.float32)
        y_tf = tf.convert_to_tensor(y_df, dtype=tf.float32)

        results = []

        for eps in epsilon_list:
            print(f"\n🔁 Epsilon = {eps}")

            x_adv = fgsm_attack_tf(self.model, x_tf, y_tf, epsilon=eps,verbose=verbose)

            y_pred_adv = (self.model.predict(x_adv) > 0.5).astype(int).flatten()
            y_true = y_tf.numpy()

            adv_metrics = evaluate_metrics(y_true, y_pred_adv)

            results.append({
                'epsilon': eps,
                'adversarial': adv_metrics
            })

        return results


# ========================
# PGD Attack Class
# ========================   
class PGD:
    def __init__(self, model, clip_min=0.0, clip_max=1.0, alpha_ratio=0.2):
        self.model = model
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.alpha_ratio = alpha_ratio

    def perform_pgd(self, x_df, y_df, epsilon=0.05, alpha=None, num_iter=10):
        x_tf = tf.convert_to_tensor(x_df, dtype=tf.float32)
        y_tf = tf.convert_to_tensor(y_df, dtype=tf.float32)

        if alpha is None:
            alpha = epsilon * self.alpha_ratio

        x_adv = pgd_attack_tf(self.model, x_tf, y_tf,
                              epsilon=epsilon, alpha=alpha, num_iter=num_iter,
                              clip_min=self.clip_min, clip_max=self.clip_max)

        y_pred_clean = (self.model.predict(x_tf) > 0.5).astype(int).flatten()
        y_pred_adv = (self.model.predict(x_adv) > 0.5).astype(int).flatten()
        y_true = y_tf.numpy()

        clean_metrics = evaluate_metrics(y_true, y_pred_clean)
        adv_metrics = evaluate_metrics(y_true, y_pred_adv)

        return {
            'epsilon': epsilon,
            'clean': clean_metrics,
            'adversarial': adv_metrics
        }

    def perform_pgd_batch(self, x_df, y_df, epsilon_list, alpha_ratio=0.2, num_iter=10, verbose=True):
        x_tf = tf.convert_to_tensor(x_df, dtype=tf.float32)
        y_tf = tf.convert_to_tensor(y_df, dtype=tf.float32)

        results = []

        for eps in epsilon_list:
            if verbose:
                print(f"\n🔁 Epsilon = {eps}")

            alpha = eps * alpha_ratio

            x_adv = pgd_attack_tf(
                self.model, x_tf, y_tf,
                epsilon=eps,
                alpha=alpha,
                num_iter=num_iter,
                clip_min=self.clip_min,
                clip_max=self.clip_max
            )

            y_pred_adv = (self.model.predict(x_adv) > 0.5).astype(int).flatten()
            y_true = y_tf.numpy()

            adv_metrics = evaluate_metrics(y_true, y_pred_adv)

            results.append({
                'epsilon': eps,
                'adversarial': adv_metrics
            })

        return results


