import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

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
