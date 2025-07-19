import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from boruta import BorutaPy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf


class SimpleBinaryPSO:
    def __init__(self, n_particles, dimensions, max_iter, c1=1.5, c2=1.5, w=0.7):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w = w

        # Initialization
        self.X = np.random.randint(0, 2, size=(n_particles, dimensions))  # Binary positions
        self.V = np.random.uniform(low=-1, high=1, size=(n_particles, dimensions))  # Velocities
        self.pbest = self.X.copy()
        self.pbest_scores = np.full(n_particles, -np.inf)
        self.gbest = None
        self.gbest_score = -np.inf

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def update_particles(self):
        r1 = np.random.rand(self.n_particles, self.dimensions)
        r2 = np.random.rand(self.n_particles, self.dimensions)

        cognitive = self.c1 * r1 * (self.pbest - self.X)
        social = self.c2 * r2 * (self.gbest - self.X)
        self.V = self.w * self.V + cognitive + social

        prob = self.sigmoid(self.V)
        self.X = (np.random.rand(self.n_particles, self.dimensions) < prob).astype(int)

    def optimize(self, fitness_func, verbose=False):
        for it in range(self.max_iter):
            if verbose:
                print(f"\n=== Iteration {it + 1}/{self.max_iter} ===")
            for i in range(self.n_particles):
                score = fitness_func(self.X[i])
                if verbose:
                    print(f"Particle {i + 1}/{self.n_particles}: Score = {score:.4f}", end='')

                if score > self.pbest_scores[i]:
                    if verbose:
                        print(f"  --> New personal best! Previous: {self.pbest_scores[i]:.4f}")
                    self.pbest_scores[i] = score
                    self.pbest[i] = self.X[i].copy()
                else:
                    if verbose:
                        print()

                if score > self.gbest_score:
                    if verbose:
                        print(f"*** New global best score found: {score:.4f} (Particle {i + 1}) ***")
                    self.gbest_score = score
                    self.gbest = self.X[i].copy()

            self.update_particles()

            if verbose:
                print(f"After iteration {it + 1}, global best score: {self.gbest_score:.4f}")

        return self.gbest

class PSOFeatureSelector:
    def __init__(self, classifier=None):
        self.classifier = classifier or RandomForestClassifier(random_state=42)

    def fitness_function(self, feature_mask, X, y, verbose=False):
        if np.sum(feature_mask) == 0:
            return 0
        X_selected = X[:, feature_mask.astype(bool)]
        scorer = make_scorer(f1_score, average='macro')
        try:
            scores = cross_val_score(self.classifier, X_selected, y, cv=3, scoring=scorer, n_jobs=1)
            return scores.mean()
        except Exception as e:
            if verbose:
                print(f"Exception in fitness function: {e}")
            return 0

    def run_pso(self, df, target_column, n_particles=20, iters=10, verbose=False):
        X_df = df.drop(columns=[target_column])
        y = df[target_column]
        X = X_df.values

        def wrapped_fitness(mask):
            return self.fitness_function(mask, X, y, verbose)

        pso = SimpleBinaryPSO(n_particles=n_particles, dimensions=X.shape[1], max_iter=iters)
        best_mask = pso.optimize(wrapped_fitness, verbose=verbose).astype(bool)
        selected_features = X_df.columns[best_mask]

        print(f"\n✅ Selected {best_mask.sum()} features out of {len(best_mask)}")
        print("Selected features:", selected_features.tolist())
        return best_mask, selected_features

class KBestFeatureSelector:
    def __init__(self, k=20, classifier=None):
        self.k = k
        self.classifier = classifier or RandomForestClassifier(random_state=42)

    def select_features(self, df, target_column, verbose=False):
        X_df = df.drop(columns=[target_column])
        y = df[target_column]

        if verbose:
            print(f"\n🔍 Starting SelectKBest feature selection (k = {self.k}) using mutual_info_classif...")

        selector = SelectKBest(score_func=mutual_info_classif, k=self.k)
        selector.fit(X_df, y)

        selected_mask = selector.get_support()
        selected_features = X_df.columns[selected_mask]

        if verbose:
            print(f"✅ Selected top {self.k} features out of {X_df.shape[1]}")
            print("📌 Selected features:")
            for i, feat in enumerate(selected_features, 1):
                print(f"  {i:2d}. {feat}")

        return selected_mask, selected_features

    def evaluate_selected(self, df, target_column, selected_mask, verbose=False):
        X_df = df.drop(columns=[target_column])
        y = df[target_column]
        X_selected = X_df.iloc[:, selected_mask]

        scorer = make_scorer(f1_score, average='macro')
        scores = cross_val_score(self.classifier, X_selected, y, cv=3, scoring=scorer, n_jobs=1)

        if verbose:
            print("\n🧪 Evaluating selected features with cross-validated F1 score (macro)...")
            print(f"📊 F1 scores from each fold: {np.round(scores, 4).tolist()}")
            print(f"📈 Mean F1 Score: {scores.mean():.4f}")

        return scores.mean()

class AutoencoderFeatureSelector:
    def __init__(self, encoding_dim=40, classifier=None, epochs=100, batch_size=64, verbose=True):
        self.encoding_dim = encoding_dim
        self.classifier = classifier or RandomForestClassifier(random_state=42)
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.encoder = None
        self.feature_ranks = None

    def _build_autoencoder(self, input_dim):
        if self.verbose:
            print(f"🔧 Building autoencoder (input_dim = {input_dim}, encoding_dim = {self.encoding_dim})...")

        input_layer = Input(shape=(input_dim,))
        encoded = Dense(128, activation='relu')(input_layer)
        encoded = Dense(self.encoding_dim, activation='relu', name="encoded_layer")(encoded)

        decoded = Dense(128, activation='relu')(encoded)
        decoded = Dense(input_dim, activation='linear')(decoded)

        autoencoder = Model(inputs=input_layer, outputs=decoded)
        encoder = Model(inputs=input_layer, outputs=encoded)

        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        if self.verbose:
            autoencoder.summary()

        return autoencoder, encoder

    def _rank_original_features(self, encoder, feature_names):
        # Get weights from input -> first hidden layer
        first_layer_weights = encoder.layers[1].get_weights()[0]  # Shape: (num_features, 128)
        importances = np.linalg.norm(first_layer_weights, axis=1)  # L2 norm per feature

        ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])

        if self.verbose:
            print("\n📊 Feature importances from encoder (top 10):")
            for i, (name, score) in enumerate(ranked[:10]):
                print(f"  {i+1}. {name}: {score:.4f}")

        return ranked

    def select_features(self, df, target_column, top_k_features=None):
        X_df = df.drop(columns=[target_column])
        y = df[target_column]
        feature_names = X_df.columns.tolist()

        if self.verbose:
            print(f"\n🧹 Input features shape: {X_df.shape}, Target shape: {y.shape}")
            print("🚫 Skipping normalization (already preprocessed)...")

        X = X_df.values.astype(np.float32)

        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

        autoencoder, encoder = self._build_autoencoder(X.shape[1])
        self.encoder = encoder

        early_stop = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
        )

        history = autoencoder.fit(
            X_train, X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(X_val, X_val),
            callbacks=[early_stop],
            verbose=self.verbose
        )

        if self.verbose:
            print("\n✅ Autoencoder training complete. Encoding data...")

        # Get encoded data
        X_encoded = encoder.predict(X, verbose=self.verbose)

        # Get feature importance ranking
        ranked_features = self._rank_original_features(encoder, feature_names)
        self.feature_ranks = ranked_features

        if top_k_features is not None:
            top_features = [name for name, _ in ranked_features[:top_k_features]]
            X_top = X_df[top_features].values
            if self.verbose:
                print(f"\n✅ Selected Top-{top_k_features} original features based on encoder importances.")
            return top_features, X_top, y.values

        # Return the full embedding if top_k not requested
        encoded_feature_names = [f'encoded_{i}' for i in range(X_encoded.shape[1])]
        return encoded_feature_names, X_encoded, y.values

    def evaluate_selected(self, X, y, verbose=True):
        if verbose:
            print("\n🧪 Evaluating selected features with cross-validated F1 score (macro)...")

        scorer = make_scorer(f1_score, average='macro')
        scores = cross_val_score(self.classifier, X, y, cv=3, scoring=scorer, n_jobs=-1)

        if verbose:
            print(f"📊 F1 scores from each fold: {np.round(scores, 4).tolist()}")
            print(f"📈 Mean F1 Score: {scores.mean():.4f}")

        return scores.mean()

class BorutaFeatureSelector:
    def __init__(self, max_iter=100, classifier=None, verbose=True):
        self.max_iter = max_iter
        self.classifier = classifier or RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
        self.verbose = verbose
        self.selected_features_ = None

    def select_features(self, df, target_column, top_k_features=None):
        X_df = df.drop(columns=[target_column])
        y = df[target_column]
        feature_names = X_df.columns.tolist()

        if self.verbose:
            print(f"\n🧹 Input features shape: {X_df.shape}, Target shape: {y.shape}")
            print("🚫 Skipping normalization (already preprocessed)...")

        X = X_df.values
        y = y.values

        rf = self.classifier
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2 if self.verbose else 0, random_state=42, max_iter=self.max_iter)

        if self.verbose:
            print("\n🚀 Starting Boruta feature selection...")

        feat_selector.fit(X, y)

        support = feat_selector.support_
        ranking = feat_selector.ranking_

        selected_features = [feature for feature, keep in zip(feature_names, support) if keep]
        ranked_features = sorted(zip(feature_names, ranking), key=lambda x: x[1])

        if self.verbose:
            print("\n✅ Boruta selection complete.")
            print("\n📊 Top 10 features (lowest rank):")
            for i, (name, rank) in enumerate(ranked_features[:10]):
                print(f"  {i+1}. {name}: Rank {rank}")

        self.selected_features_ = selected_features

        if top_k_features is not None:
            selected_features = selected_features[:top_k_features]

        X_top = X_df[selected_features].values
        return selected_features, X_top, y

    def evaluate_selected(self, X, y, verbose=True):
        if verbose:
            print("\n🧪 Evaluating selected features with cross-validated F1 score (macro)...")

        scorer = make_scorer(f1_score, average='macro')
        scores = cross_val_score(self.classifier, X, y, cv=3, scoring=scorer, n_jobs=-1)

        if verbose:
            print(f"📊 F1 scores from each fold: {np.round(scores, 4).tolist()}")
            print(f"📈 Mean F1 Score: {scores.mean():.4f}")

        return scores.mean()
