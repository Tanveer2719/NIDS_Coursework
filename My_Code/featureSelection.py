from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
import pyswarms as ps
import numpy as np
import pandas as pd

class PSOFeatureSelector:
    def __init__(self, classifier=None):
        self.classifier = classifier or RandomForestClassifier(random_state=42)

    def __fitness_function(self, feature_mask: np.ndarray, X: np.ndarray, y: np.ndarray, verbose: bool=False, idx: int=None) -> float:
        if np.sum(feature_mask) == 0:
            if verbose:
                print(f"[Particle {idx}] No features selected. Score: 0")
            return 0

        X_selected = X[:, feature_mask.astype(bool)]
        scorer = make_scorer(f1_score, average='macro')

        try:
            scores = cross_val_score(self.classifier, X_selected, y, cv=5, scoring=scorer)
            mean_score = scores.mean()
        except Exception as e:
            if verbose:
                print(f"[Particle {idx}] Exception: {e}")
            return 0

        if verbose:
            print(f"[Particle {idx}] Features: {np.sum(feature_mask)} | Score: {mean_score:.4f}")
        return mean_score

    def __pso_feature_selection(self, X: np.ndarray, y: np.ndarray, n_particles: int=30, iters: int=50, verbose: bool=False):
        dim = X.shape[1]

        optimizer = ps.single.BinaryPSO(
            n_particles=n_particles,
            dimensions=dim,
            options={'c1': 1.5, 'c2': 1.5, 'w': 0.7}
        )

        def objective_func(particles):
            scores = []
            for i, particle in enumerate(particles):
                score = self.fitness_function(particle, X, y, verbose=verbose, idx=i)
                scores.append(-score)  # Minimize negative score
            return np.array(scores)

        best_cost, best_pos = optimizer.optimize(objective_func, iters=iters, verbose=verbose)
        return best_pos.astype(bool)

    def run_pso(self, df: pd.DataFrame, target_column: str, n_particles: int=30, iters: int=50, verbose: bool=False):
        """
        df: DataFrame containing both features and target.
        target_column: Name of the target column.
        """
        X_df = df.drop(columns=[target_column])
        y = df[target_column]

        X = X_df.values

        if verbose:
            print(f"\nRunning PSO on {X.shape[1]} features and {len(np.unique(y))} target classes.\n")

        selected_mask = self.pso_feature_selection(X, y, n_particles=n_particles, iters=iters, verbose=verbose)
        selected_features = X_df.columns[selected_mask]

        print(f"\n Selected {selected_mask.sum()} features out of {len(selected_mask)}")
        print("Selected features:")
        print(selected_features.tolist())

        return selected_mask, selected_features
