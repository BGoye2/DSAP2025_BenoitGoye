"""
Model Training for GINI Coefficient Prediction
================================================
This script trains five tree-based regression models:
- Decision Tree: Simple, interpretable baseline model
- Random Forest: Ensemble of decision trees with bagging
- Gradient Boosting: Sequential ensemble with boosting
- XGBoost: Optimized gradient boosting with regularization
- LightGBM: Fast gradient boosting with leaf-wise growth

Features:
- Train/test data splitting
- Optional hyperparameter tuning via GridSearchCV
- Cross-validation for robust performance estimation
- Model persistence via .pkl files

Input: output/processed_data.csv, output/feature_names.csv
Output: Trained models saved to output/trained_models.pkl

Note: Models are always saved to ensure single training per pipeline run.
      Other scripts load these models instead of retraining.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

from config.constants import TARGET_VARIABLE
import lightgbm as lgb
from typing import Dict, Tuple
import joblib
import hashlib
from datetime import datetime
import warnings

from config.constants import (
    DEFAULT_RANDOM_SEED,
    PROCESSED_DATA_PATH,
    FEATURE_NAMES_PATH,
    MODELS_PATH,
    DEFAULT_HYPERPARAMETERS,
    HYPERPARAMETER_GRIDS,
    CV_FOLDS,
    CV_SCORING,
    SECTION_SEPARATOR
)

warnings.filterwarnings('ignore')


class ModelTrainer:
    """Train tree-based models for GINI coefficient prediction"""

    def __init__(self, data_path: str = PROCESSED_DATA_PATH):
        """
        Initialize trainer

        Parameters:
        -----------
        data_path : str
            Path to processed data
        """
        self.data_path = data_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.models = {}
        self.data_hash = None

    @staticmethod
    def _compute_data_hash(X: np.ndarray, y: np.ndarray) -> str:
        """
        Compute SHA256 hash of training data for validation

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector

        Returns:
        --------
        str
            Hexadecimal hash string
        """
        # Combine X and y into single array for hashing
        combined = np.concatenate([X.flatten(), y.flatten()])
        hash_obj = hashlib.sha256(combined.tobytes())
        return hash_obj.hexdigest()

    def load_and_split_data(self, test_size: float = 0.2,
                           random_state: int = DEFAULT_RANDOM_SEED) -> None:
        """
        Load data and split into train/test sets

        Parameters:
        -----------
        test_size : float
            Proportion of data for testing
        random_state : int
            Random seed for reproducibility
        """
        print("Loading processed data...")
        data = pd.read_csv(self.data_path)

        # Load feature names
        feature_names_df = pd.read_csv(FEATURE_NAMES_PATH)
        self.feature_names = feature_names_df['feature'].tolist()

        # Separate features and target
        X = data[self.feature_names].values
        y = data[TARGET_VARIABLE].values

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Compute hash of training data for validation
        self.data_hash = self._compute_data_hash(self.X_train, self.y_train)

        print(f"\nData split complete:")
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Data hash: {self.data_hash[:16]}...")

    def _train_with_grid_search(self, model, param_grid: Dict, model_key: str,
                               model_name: str) -> None:
        """
        Train a model using GridSearchCV for hyperparameter tuning

        Parameters:
        -----------
        model
            Model instance to tune
        param_grid : Dict
            Hyperparameter grid for search
        model_key : str
            Internal key for storing model (e.g., 'decision_tree')
        model_name : str
            Display name for logging (e.g., 'Decision Tree')
        """
        print("Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            model, param_grid, cv=CV_FOLDS, scoring=CV_SCORING,
            n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)
        print(f"\nBest parameters: {grid_search.best_params_}")
        self.models[model_key] = grid_search.best_estimator_

    def _train_with_defaults(self, model, model_key: str, model_name: str) -> None:
        """
        Train a model using default hyperparameters

        Parameters:
        -----------
        model
            Configured model instance
        model_key : str
            Internal key for storing model
        model_name : str
            Display name for logging
        """
        print("Training with default parameters...")
        model.fit(self.X_train, self.y_train)
        self.models[model_key] = model

    def _evaluate_with_cv(self, model_key: str, model_name: str) -> float:
        """
        Perform cross-validation and print results

        Parameters:
        -----------
        model_key : str
            Internal key of trained model
        model_name : str
            Display name for logging

        Returns:
        --------
        float
            Cross-validated RMSE
        """
        cv_scores = cross_val_score(
            self.models[model_key], self.X_train, self.y_train,
            cv=CV_FOLDS, scoring=CV_SCORING, n_jobs=-1
        )
        cv_rmse = np.sqrt(-cv_scores.mean())
        print(f"✓ {model_name} trained | CV RMSE: {cv_rmse:.3f}")
        return cv_rmse

    @staticmethod
    def _print_section_header(title: str) -> None:
        """Print formatted section header"""
        print(f"\n{SECTION_SEPARATOR}")
        print(title)
        print(SECTION_SEPARATOR)

    def train_decision_tree(self, tune_hyperparameters: bool = False) -> None:
        """
        Train Decision Tree Regressor

        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        """
        self._print_section_header("TRAINING DECISION TREE REGRESSOR")

        if tune_hyperparameters:
            dt = DecisionTreeRegressor(random_state=DEFAULT_RANDOM_SEED)
            self._train_with_grid_search(
                dt, HYPERPARAMETER_GRIDS['decision_tree'],
                'decision_tree', 'Decision Tree'
            )
        else:
            dt = DecisionTreeRegressor(**DEFAULT_HYPERPARAMETERS['decision_tree'])
            self._train_with_defaults(dt, 'decision_tree', 'Decision Tree')

        self._evaluate_with_cv('decision_tree', 'Decision Tree')

    def train_random_forest(self, tune_hyperparameters: bool = False) -> None:
        """
        Train Random Forest Regressor

        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        """
        self._print_section_header("TRAINING RANDOM FOREST REGRESSOR")

        if tune_hyperparameters:
            rf = RandomForestRegressor(random_state=DEFAULT_RANDOM_SEED, n_jobs=-1)
            self._train_with_grid_search(
                rf, HYPERPARAMETER_GRIDS['random_forest'],
                'random_forest', 'Random Forest'
            )
        else:
            rf = RandomForestRegressor(**DEFAULT_HYPERPARAMETERS['random_forest'])
            self._train_with_defaults(rf, 'random_forest', 'Random Forest')

        self._evaluate_with_cv('random_forest', 'Random Forest')

    def train_gradient_boosting(self, tune_hyperparameters: bool = False) -> None:
        """
        Train Gradient Boosting Regressor

        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        """
        self._print_section_header("TRAINING GRADIENT BOOSTING REGRESSOR")

        if tune_hyperparameters:
            gb = GradientBoostingRegressor(random_state=DEFAULT_RANDOM_SEED)
            self._train_with_grid_search(
                gb, HYPERPARAMETER_GRIDS['gradient_boosting'],
                'gradient_boosting', 'Gradient Boosting'
            )
        else:
            gb = GradientBoostingRegressor(**DEFAULT_HYPERPARAMETERS['gradient_boosting'])
            self._train_with_defaults(gb, 'gradient_boosting', 'Gradient Boosting')

        self._evaluate_with_cv('gradient_boosting', 'Gradient Boosting')

    def train_xgboost(self, tune_hyperparameters: bool = False) -> None:
        """
        Train XGBoost Regressor

        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        """
        self._print_section_header("TRAINING XGBOOST REGRESSOR")

        if tune_hyperparameters:
            xgboost = xgb.XGBRegressor(
                random_state=DEFAULT_RANDOM_SEED,
                n_jobs=-1,
                tree_method='hist'
            )
            self._train_with_grid_search(
                xgboost, HYPERPARAMETER_GRIDS['xgboost'],
                'xgboost', 'XGBoost'
            )
        else:
            xgboost = xgb.XGBRegressor(**DEFAULT_HYPERPARAMETERS['xgboost'])
            self._train_with_defaults(xgboost, 'xgboost', 'XGBoost')

        self._evaluate_with_cv('xgboost', 'XGBoost')

    def train_lightgbm(self, tune_hyperparameters: bool = False) -> None:
        """
        Train LightGBM Regressor

        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        """
        self._print_section_header("TRAINING LIGHTGBM REGRESSOR")

        if tune_hyperparameters:
            lightgbm = lgb.LGBMRegressor(
                random_state=DEFAULT_RANDOM_SEED,
                n_jobs=-1,
                verbose=-1
            )
            self._train_with_grid_search(
                lightgbm, HYPERPARAMETER_GRIDS['lightgbm'],
                'lightgbm', 'LightGBM'
            )
        else:
            lightgbm = lgb.LGBMRegressor(**DEFAULT_HYPERPARAMETERS['lightgbm'])
            self._train_with_defaults(lightgbm, 'lightgbm', 'LightGBM')

        self._evaluate_with_cv('lightgbm', 'LightGBM')

    def train_all_models(self, tune_hyperparameters: bool = False) -> None:
        """
        Train all models at once

        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning (slower but better)
        """
        print("\nTraining all models...")
        self.train_decision_tree(tune_hyperparameters)
        self.train_random_forest(tune_hyperparameters)
        self.train_gradient_boosting(tune_hyperparameters)
        self.train_xgboost(tune_hyperparameters)
        self.train_lightgbm(tune_hyperparameters)
        print("\n✓ All models trained successfully")

    def save_models(self, filepath: str = MODELS_PATH) -> None:
        """
        Save models and data splits to disk with metadata

        Parameters:
        -----------
        filepath : str
            Path to save models (default: output/trained_models.pkl)
        """
        # Create metadata
        metadata = {
            'version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'data_hash': self.data_hash,
            'n_train': len(self.X_train),
            'n_test': len(self.X_test),
            'n_features': len(self.feature_names),
            'model_names': list(self.models.keys()),
            'sklearn_version': joblib.__version__,
            'data_source': self.data_path
        }

        # Save models, data, and metadata to pickle file
        save_data = {
            'models': self.models,
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'feature_names': self.feature_names,
            'metadata': metadata
        }

        joblib.dump(save_data, filepath)
        print(f"\n✓ Models saved to {filepath}")
        print(f"  Version: {metadata['version']}")
        print(f"  Timestamp: {metadata['timestamp']}")
        print(f"  Data hash: {metadata['data_hash'][:16]}...")


def main():
    """
    Main execution function for model training
    """
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train GINI prediction models')
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning (slower but better results)'
    )
    args = parser.parse_args()

    # Initialize trainer
    trainer = ModelTrainer(PROCESSED_DATA_PATH)

    # Load and split data
    trainer.load_and_split_data(test_size=0.2)

    # Train all models with optional hyperparameter tuning
    trainer.train_all_models(tune_hyperparameters=args.tune)

    # Always save models to disk
    trainer.save_models()

    print(f"\n{SECTION_SEPARATOR}")
    print("MODEL TRAINING COMPLETE!")
    print(SECTION_SEPARATOR)
    print(f"\nTrained {len(trainer.models)} models:")
    for model_name in trainer.models.keys():
        print(f"  • {model_name.replace('_', ' ').title()}")

    print("\nModels saved and ready for use by downstream scripts")
    print("Other scripts will load these models instead of retraining")

    return trainer


if __name__ == "__main__":
    main()
