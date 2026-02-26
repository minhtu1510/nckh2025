"""
Stacking Ensemble for IDS Defense

Two-layer architecture:
- Layer 1: Base models (MLP, SVM, RF, KNN)
- Layer 2: Meta-learner (Logistic Regression) learns optimal combination

Advantages over fixed weights:
- Learns optimal combination from data
- Adapts to different attack types
- Higher accuracy (typically +3-5%)
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import joblib


class StackingEnsemble:
    """
    Stacking ensemble with meta-learning
    
    Usage:
        ensemble = StackingEnsemble(base_models)
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
    """
    
    def __init__(self, base_models, meta_model=None, val_split=0.2, random_state=42):
        """
        Args:
            base_models: Dict of {name: model} for base models
            meta_model: Meta-learner (default: LogisticRegression)
            val_split: Fraction of data for meta-learning
            random_state: Random seed
        """
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight='balanced'  # Handle imbalanced data
        )
        self.val_split = val_split
        self.random_state = random_state
        self.is_fitted = False
        
    def fit(self, X, y, verbose=True):
        """
        Train stacking ensemble
        
        Process:
        1. Split data into train/validation
        2. Train base models on train set
        3. Generate meta-features from validation set
        4. Train meta-learner on meta-features
        """
        if verbose:
            print("\n" + "="*80)
            print("TRAINING STACKING ENSEMBLE".center(80))
            print("="*80)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.val_split, 
            random_state=self.random_state,
            stratify=y
        )
        
        if verbose:
            print(f"\nData split:")
            print(f"  Train: {len(X_train):,} samples")
            print(f"  Validation: {len(X_val):,} samples")
        
        # Train base models
        if verbose:
            print(f"\n[LAYER 1] Training {len(self.base_models)} base models...")
        
        for name, model in self.base_models.items():
            if verbose:
                print(f"  Training {name}...", end=" ")
            
            # Handle neural network (MLP)
            if hasattr(model, 'fit') and hasattr(model, 'predict'):
                if name == 'mlp':
                    # MLP needs epochs, batch_size
                    model.fit(
                        X_train, y_train,
                        epochs=50,
                        batch_size=128,
                        validation_split=0.1,
                        verbose=0
                    )
                else:
                    # Sklearn models
                    model.fit(X_train, y_train)
            
            if verbose:
                print("✓")
        
        # Generate meta-features
        if verbose:
            print(f"\n[LAYER 2] Generating meta-features from validation set...")
        
        meta_features = self._get_meta_features(X_val, verbose=verbose)
        
        if verbose:
            print(f"  Meta-features shape: {meta_features.shape}")
        
        # Train meta-learner
        if verbose:
            print(f"\nTraining meta-learner (Logistic Regression)...")
        
        self.meta_model.fit(meta_features, y_val)
        
        # Evaluate on validation
        val_acc = self.meta_model.score(meta_features, y_val)
        
        if verbose:
            print(f"  ✓ Meta-learner trained!")
            print(f"  Validation accuracy: {val_acc:.4f}")
            print(f"\n{'='*80}")
            print("✅ STACKING ENSEMBLE READY!".center(80))
            print(f"{'='*80}\n")
        
        self.is_fitted = True
        return self
    
    def _get_meta_features(self, X, verbose=False):
        """
        Get predictions from base models as meta-features
        
        Returns:
            meta_features: (n_samples, n_base_models) array
        """
        meta_features = []
        
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                # For probabilistic models (MLP, RF, KNN, SVM with probability=True)
                preds = model.predict_proba(X)[:, 1]  # Probability of class 1
            elif hasattr(model, 'decision_function'):
                # For SVM without probability
                preds = model.decision_function(X)
                # Normalize to [0, 1] range
                preds = 1 / (1 + np.exp(-preds))  # Sigmoid
            else:
                # Fallback: hard predictions
                preds = model.predict(X).astype(float)
            
            meta_features.append(preds)
            
            if verbose:
                print(f"    {name}: mean={preds.mean():.3f}, std={preds.std():.3f}")
        
        return np.column_stack(meta_features)
    
    def predict(self, X):
        """
        Predict using stacking ensemble
        
        Returns:
            predictions: Binary predictions (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X):
        """
        Predict probabilities using stacking ensemble
        
        Returns:
            probabilities: (n_samples, 2) array
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict_proba(meta_features)
    
    def get_base_predictions(self, X):
        """
        Get individual predictions from all base models
        
        Useful for analysis and debugging
        
        Returns:
            dict: {model_name: predictions}
        """
        predictions = {}
        for name, model in self.base_models.items():
            predictions[name] = model.predict(X)
        return predictions
    
    def save(self, output_dir):
        """Save ensemble to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save base models
        for name, model in self.base_models.items():
            if hasattr(model, 'save'):
                # Keras model
                model.save(output_dir / f'{name}_model.keras')
            else:
                # Sklearn model
                joblib.dump(model, output_dir / f'{name}_model.pkl')
        
        # Save meta-model
        joblib.dump(self.meta_model, output_dir / 'meta_model.pkl')
        
        # Save config
        config = {
            'val_split': self.val_split,
            'random_state': self.random_state,
            'base_model_names': list(self.base_models.keys())
        }
        joblib.dump(config, output_dir / 'config.pkl')
        
        print(f"✓ Saved stacking ensemble to {output_dir}")
    
    @classmethod
    def load(cls, model_dir, base_models):
        """
        Load ensemble from disk
        
        Args:
            model_dir: Directory containing saved models
            base_models: Dict of base models (must provide structure)
        
        Returns:
            Loaded StackingEnsemble
        """
        model_dir = Path(model_dir)
        
        # Load meta-model
        meta_model = joblib.load(model_dir / 'meta_model.pkl')
        
        # Load config
        config = joblib.load(model_dir / 'config.pkl')
        
        # Create ensemble
        ensemble = cls(
            base_models=base_models,
            meta_model=meta_model,
            val_split=config['val_split'],
            random_state=config['random_state']
        )
        ensemble.is_fitted = True
        
        print(f"✓ Loaded stacking ensemble from {model_dir}")
        return ensemble


def create_stacking_ensemble(input_dim, random_state=42):
    """
    Convenience function to create stacking ensemble with default models
    
    Args:
        input_dim: Number of input features
        random_state: Random seed
    
    Returns:
        StackingEnsemble ready to train
    """
    from models.advanced.mlp import create_mlp_model
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    
    # Match Exp1 config for fair comparison
    base_svm = LinearSVC(
        C=1.0,
        max_iter=3000,
        random_state=random_state,
        verbose=0
    )
    
    base_models = {
        'mlp': create_mlp_model(
            input_dim=input_dim,
            hidden_layers=[128, 64, 32],
            dropout_rate=0.3,
            learning_rate=0.001
        ),
        # LinearSVC + Calibration (matches Exp1)
        'svm': CalibratedClassifierCV(base_svm, cv=3),
        'rf': RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            random_state=random_state,
            n_jobs=-1
        ),
        'knn': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        )
    }
    
    return StackingEnsemble(base_models, random_state=random_state)


if __name__ == '__main__':
    # Example usage
    print("Testing Stacking Ensemble...")
    
    from sklearn.datasets import make_classification
    
    # Generate dummy data
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        random_state=42
    )
    
    # Create ensemble
    ensemble = create_stacking_ensemble(input_dim=50)
    
    # Train
    ensemble.fit(X, y)
    
    # Predict
    predictions = ensemble.predict(X[:10])
    probabilities = ensemble.predict_proba(X[:10])
    
    print(f"\nPredictions (first 10): {predictions}")
    print(f"Probabilities (first 10):\n{probabilities}")
    
    print("\n✓ Test passed!")


def create_stacking_ensemble_gan_optimized(input_dim, random_state=42):
    """
    GAN-optimized stacking ensemble.

    RAW GAN performance: MLP=0.949, KNN=0.914, RF=0.861, SVM=0.787
    Removes RF+SVM (weak on GAN), uses dual MLP + dual KNN variants.
    Expected GAN F1 improvement: 0.822 → ~0.93+
    """
    from models.advanced.mlp import create_mlp_model
    from sklearn.neighbors import KNeighborsClassifier

    base_models = {
        'mlp_deep': create_mlp_model(
            input_dim=input_dim,
            hidden_layers=[256, 128, 64],
            dropout_rate=0.3,
            learning_rate=0.001
        ),
        'mlp_wide': create_mlp_model(
            input_dim=input_dim,
            hidden_layers=[128, 64, 32],
            dropout_rate=0.2,
            learning_rate=0.0005
        ),
        'knn_5': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        ),
        'knn_11': KNeighborsClassifier(
            n_neighbors=11,
            weights='distance',
            n_jobs=-1
        ),
    }

    return StackingEnsemble(base_models, random_state=random_state)
