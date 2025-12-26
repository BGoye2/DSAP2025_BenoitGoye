"""
Prediction Script - On-Demand Model Training and Inference

This script provides an interface for making GINI coefficient predictions on new data.
Models are trained on-demand from processed data rather than loaded from saved files.

Key Features:
- On-demand model training: No .pkl files needed, trains fresh models when first called
- Single prediction: Predict GINI for one observation (country-year)
- Batch prediction: Predict for multiple observations from CSV
- Ensemble prediction: Get predictions from all models and compute weighted average
- Automatic feature alignment: Ensures input data matches training feature order

Supported Models:
- Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM

Usage Example:
    predictor = GINIPredictor()
    # Single prediction
    gini_pred = predictor.predict_single(features_dict, model_name='XGBoost')
    # Ensemble prediction
    predictions = predictor.predict_ensemble(features_dict)
    # Batch prediction from CSV
    predictor.predict_from_csv('new_data.csv', 'predictions.csv')

Input: output/processed_data.csv, output/feature_names.csv, user-provided new data
Output: GINI coefficient predictions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class GINIPredictor:
    """Make predictions by training models on-demand"""

    def __init__(self, data_path: str = 'output/processed_data.csv'):
        """
        Initialize predictor

        Parameters:
        -----------
        data_path : str
            Path to processed training data
        """
        self.data_path = data_path
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.X_train = None
        self.y_train = None
        self._is_trained = False

    def _load_and_prepare_data(self) -> None:
        """Load training data and prepare it"""
        if self._is_trained:
            return

        print("Loading training data...")
        data = pd.read_csv(self.data_path)

        # Load feature names
        try:
            feature_names_df = pd.read_csv('output/feature_names.csv')
            self.feature_names = feature_names_df['feature'].tolist()
        except FileNotFoundError:
            # If feature_names.csv doesn't exist, infer from data
            self.feature_names = [col for col in data.columns if col != 'SI.POV.GINI']

        # Separate features and target
        self.X_train = data[self.feature_names].values
        self.y_train = data['SI.POV.GINI'].values

        print(f"Loaded {len(self.X_train)} training samples with {len(self.feature_names)} features")

    def _train_models(self) -> None:
        """Train all models"""
        if self._is_trained:
            return

        self._load_and_prepare_data()

        print("\nTraining models on-demand...")

        # Train Decision Tree
        print("  - Training Decision Tree...")
        self.models['decision_tree'] = DecisionTreeRegressor(
            max_depth=10, min_samples_split=20, random_state=42
        )
        self.models['decision_tree'].fit(self.X_train, self.y_train)

        # Train Random Forest
        print("  - Training Random Forest...")
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        )
        self.models['random_forest'].fit(self.X_train, self.y_train)

        # Train Gradient Boosting
        print("  - Training Gradient Boosting...")
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        )
        self.models['gradient_boosting'].fit(self.X_train, self.y_train)

        # Train XGBoost
        print("  - Training XGBoost...")
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1
        )
        self.models['xgboost'].fit(self.X_train, self.y_train)

        # Train LightGBM
        print("  - Training LightGBM...")
        self.models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1, verbose=-1
        )
        self.models['lightgbm'].fit(self.X_train, self.y_train)

        self._is_trained = True
        print("✓ All models trained and ready\n")

    def predict_single(self, features: Dict[str, float],
                      model_name: str = 'random_forest') -> float:
        """
        Predict GINI coefficient for a single observation

        Parameters:
        -----------
        features : dict
            Dictionary of feature values
        model_name : str
            Name of model to use ('decision_tree', 'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm')

        Returns:
        --------
        float
            Predicted GINI coefficient
        """
        # Ensure models are trained
        if not self._is_trained:
            self._train_models()

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Choose from: {list(self.models.keys())}")

        # Create feature vector in correct order
        X = np.array([features.get(feat, np.nan) for feat in self.feature_names]).reshape(1, -1)

        # Make prediction
        prediction = self.models[model_name].predict(X)[0]

        return prediction

    def predict_batch(self, data: pd.DataFrame,
                     model_name: str = 'random_forest') -> np.ndarray:
        """
        Predict GINI coefficient for multiple observations

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing features
        model_name : str
            Name of model to use

        Returns:
        --------
        np.ndarray
            Array of predicted GINI coefficients
        """
        # Ensure models are trained
        if not self._is_trained:
            self._train_models()

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")

        # Extract features in correct order
        X = data[self.feature_names].values

        # Make predictions
        predictions = self.models[model_name].predict(X)

        return predictions

    def predict_ensemble(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Get predictions from all models and compute ensemble

        Parameters:
        -----------
        features : dict
            Dictionary of feature values

        Returns:
        --------
        dict
            Dictionary with predictions from each model plus ensemble average
        """
        # Ensure models are trained
        if not self._is_trained:
            self._train_models()

        predictions = {}

        for model_name in self.models.keys():
            predictions[model_name] = self.predict_single(features, model_name)

        # Simple average ensemble
        predictions['ensemble_mean'] = np.mean(list(predictions.values()))

        # Weighted ensemble (giving more weight to advanced models)
        predictions['ensemble_weighted'] = (
            0.05 * predictions.get('decision_tree', 0) +
            0.20 * predictions.get('random_forest', 0) +
            0.20 * predictions.get('gradient_boosting', 0) +
            0.275 * predictions.get('xgboost', 0) +
            0.275 * predictions.get('lightgbm', 0)
        )

        return predictions

    def predict_from_csv(self, input_file: str, output_file: str,
                        model_name: str = 'random_forest') -> pd.DataFrame:
        """
        Make predictions from CSV file

        Parameters:
        -----------
        input_file : str
            Path to input CSV file
        output_file : str
            Path to save predictions
        model_name : str
            Name of model to use

        Returns:
        --------
        pd.DataFrame
            DataFrame with predictions
        """
        print(f"Loading data from {input_file}...")
        data = pd.read_csv(input_file)

        print(f"Making predictions with {model_name}...")
        predictions = self.predict_batch(data, model_name)

        # Add predictions to dataframe
        data['predicted_gini'] = predictions

        # Save results
        data.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

        return data


def example_usage():
    """Example of how to use the predictor"""

    # Initialize predictor
    print("="*60)
    print("GINI PREDICTOR - ON-DEMAND TRAINING")
    print("="*60)
    print("\nNote: Models will be trained automatically when making predictions")
    print("No .pkl files needed!\n")

    try:
        predictor = GINIPredictor()

        # Example 1: Predict for a single country/year
        print("\n" + "="*60)
        print("EXAMPLE 1: Single Prediction")
        print("="*60)

        # Load some actual data to use as an example
        try:
            data = pd.read_csv('output/processed_data.csv')
            feature_names_df = pd.read_csv('output/feature_names.csv')
            feature_names = feature_names_df['feature'].tolist()

            # Get first row as example
            example_features = data[feature_names].iloc[0].to_dict()

            # Get predictions from all models
            predictions = predictor.predict_ensemble(example_features)
            print("\nPredicted GINI coefficients:")
            for model_name, pred in predictions.items():
                print(f"  {model_name:20s}: {pred:.2f}")
        except FileNotFoundError:
            print("Note: processed_data.csv not found. Run the pipeline first.")

        # Example 2: Predict from CSV
        print("\n" + "="*60)
        print("EXAMPLE 2: Batch Prediction from CSV")
        print("="*60)
        print("To predict from a CSV file:")
        print("  predictor.predict_from_csv('new_data.csv', 'predictions.csv')")
        print("\nModels will be trained automatically when you call prediction methods!")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run the pipeline first to generate processed_data.csv:")
        print("  python main.py")


def main():
    """Main execution"""
    example_usage()


if __name__ == "__main__":
    main()
