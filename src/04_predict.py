"""
GINI Coefficient Predictions
=============================
Make GINI coefficient predictions using trained machine learning models.
Supports both CLI batch predictions and programmatic usage for integration.

CSV File Requirements:
    - Must contain all required World Bank indicator columns used in training
    - Column names must match World Bank indicator codes (e.g., 'NY.GDP.PCAP.PP.CD')
    - No missing values allowed in feature columns
    - Standard CSV format with comma-separated values

Command-Line Usage:
    python src/04_predict.py --csv input.csv                                    # Basic prediction
    python src/04_predict.py --csv input.csv --model random_forest              # Specify model
    python src/04_predict.py --csv data.csv --output results.csv --model xgboost

Programmatic Usage:
    from src.04_predict import GINIPredictor

    # Initialize and load models
    predictor = GINIPredictor()

    # Single prediction
    features = {'NY.GDP.PCAP.PP.CD': 15000, 'SL.UEM.TOTL.ZS': 8.5, ...}
    gini = predictor.predict_single(features, model_name='xgboost')

    # Batch prediction
    data = pd.read_csv('data.csv')
    predictions = predictor.predict_batch(data, model_name='random_forest')

    # Ensemble prediction (all models with statistics)
    results = predictor.predict_ensemble(features)
    print(f"Best estimate: {results['ensemble_weighted']:.2f}")

For more information, run: python src/04_predict.py --help
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Union, Optional
import warnings
warnings.filterwarnings('ignore')


class GINIPredictor:
    """Make predictions using trained GINI models"""

    def __init__(self, models_path: str = 'output/trained_models.pkl', verbose: bool = True):
        """
        Initialize predictor

        Parameters:
        -----------
        models_path : str
            Path to trained models file
        verbose : bool
            Whether to print status messages
        """
        self.models = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.verbose = verbose

        # Load pre-trained models
        if not os.path.exists(models_path):
            raise FileNotFoundError(
                f"Trained models not found at {models_path}\n"
                "Please run 03_model_training.py first to train and save models."
            )

        self._load_from_pickle(models_path)

    def _load_from_pickle(self, filepath: str):
        """Load models from pickle file"""
        if self.verbose:
            print(f"Loading pre-trained models from {filepath}...")

        try:
            save_data = joblib.load(filepath)

            self.models = save_data['models']
            self.X_train = save_data['X_train']
            self.X_test = save_data['X_test']
            self.y_train = save_data['y_train']
            self.y_test = save_data['y_test']
            self.feature_names = save_data['feature_names']

            if self.verbose:
                print(f"SUCCESS: Loaded {len(self.models)} pre-trained models")
                print(f"         Models: {', '.join(self.models.keys())}")
                print(f"         Features: {len(self.feature_names)}")
        except Exception as e:
            print(f"ERROR: Error loading models: {e}")
            raise

    def get_available_models(self) -> List[str]:
        """
        Get list of available model names

        Returns:
        --------
        list
            List of available model names
        """
        return list(self.models.keys())

    def get_required_features(self) -> List[str]:
        """
        Get list of required feature names

        Returns:
        --------
        list
            List of required World Bank indicator codes
        """
        return list(self.feature_names)

    def predict_single(self, features: Dict[str, float],
                      model_name: str = 'xgboost') -> float:
        """
        Predict GINI coefficient for a single observation

        Parameters:
        -----------
        features : dict
            Dictionary of feature values (feature_name: value)
            Example: {'NY.GDP.PCAP.PP.CD': 15000, 'SL.UEM.TOTL.ZS': 8.5, ...}
        model_name : str
            Name of model to use (default: 'xgboost')
            Options: 'decision_tree', 'random_forest', 'gradient_boosting',
                     'xgboost', 'lightgbm'

        Returns:
        --------
        float
            Predicted GINI coefficient

        Raises:
        -------
        RuntimeError
            If models are not loaded
        ValueError
            If model_name is invalid or features are missing

        Example:
        --------
        >>> predictor = GINIPredictor()
        >>> features = {
        ...     'NY.GDP.PCAP.PP.CD': 15000,
        ...     'SL.UEM.TOTL.ZS': 8.5,
        ...     # ... all other required features
        ... }
        >>> gini = predictor.predict_single(features, model_name='xgboost')
        >>> print(f"Predicted GINI: {gini:.2f}")
        """
        # Validate model selection
        if model_name not in self.models:
            available = ', '.join(self.models.keys())
            raise ValueError(
                f"ERROR: Model '{model_name}' not available.\n"
                f"       Available models: {available}\n"
                f"       Use get_available_models() to see all options."
            )

        # Validate input type
        if not isinstance(features, dict):
            raise TypeError(
                f"ERROR: Expected features to be a dictionary, got {type(features).__name__}\n"
                f"       Example: {{'NY.GDP.PCAP.PP.CD': 15000, 'SL.UEM.TOTL.ZS': 8.5}}"
            )

        # Create feature vector in correct order
        X = np.array([features.get(feat, np.nan) for feat in self.feature_names]).reshape(1, -1)

        # Check for missing features
        if np.isnan(X).any():
            missing = [self.feature_names[i] for i, val in enumerate(X[0]) if np.isnan(val)]
            raise ValueError(
                f"ERROR: Missing {len(missing)} required feature(s):\n"
                f"       {', '.join(missing[:5])}" +
                (f"\n       ... and {len(missing)-5} more" if len(missing) > 5 else "") +
                f"\n       Use get_required_features() to see all required features."
            )

        # Make prediction
        prediction = self.models[model_name].predict(X)[0]

        return float(prediction)

    def predict_batch(self, data: Union[pd.DataFrame, List[Dict[str, float]]],
                     model_name: str = 'xgboost') -> np.ndarray:
        """
        Predict GINI coefficient for multiple observations

        Parameters:
        -----------
        data : pd.DataFrame or list of dict
            DataFrame with feature columns OR list of feature dictionaries
        model_name : str
            Name of model to use (default: 'xgboost')

        Returns:
        --------
        np.ndarray
            Array of predicted GINI coefficients

        Raises:
        -------
        RuntimeError
            If models are not loaded
        ValueError
            If model_name is invalid or required features are missing

        Example:
        --------
        >>> # Using DataFrame
        >>> data = pd.read_csv('data.csv')
        >>> predictions = predictor.predict_batch(data, model_name='random_forest')

        >>> # Using list of dictionaries
        >>> data = [
        ...     {'NY.GDP.PCAP.PP.CD': 15000, 'SL.UEM.TOTL.ZS': 8.5, ...},
        ...     {'NY.GDP.PCAP.PP.CD': 20000, 'SL.UEM.TOTL.ZS': 6.2, ...}
        ... ]
        >>> predictions = predictor.predict_batch(data, model_name='xgboost')
        """
        # Validate model selection
        if model_name not in self.models:
            available = ', '.join(self.models.keys())
            raise ValueError(
                f"ERROR: Model '{model_name}' not available.\n"
                f"       Available models: {available}"
            )

        # Convert list of dicts to DataFrame if needed
        if isinstance(data, list):
            if not data:
                raise ValueError("ERROR: Input list is empty")
            if not all(isinstance(item, dict) for item in data):
                raise TypeError("ERROR: All items in list must be dictionaries")
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"ERROR: Expected DataFrame or list of dicts, got {type(data).__name__}"
            )

        # Check for missing feature columns
        missing_cols = set(self.feature_names) - set(data.columns)
        if missing_cols:
            raise ValueError(
                f"ERROR: Input data is missing {len(missing_cols)} required feature(s):\n"
                f"       {', '.join(list(missing_cols)[:5])}" +
                (f"\n       ... and {len(missing_cols)-5} more" if len(missing_cols) > 5 else "") +
                f"\n       Use get_required_features() to see all required features."
            )

        # Extract features in correct order
        X = data[self.feature_names].values

        # Check for NaN values
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            if self.verbose:
                print(f"WARNING: Found {nan_count} NaN values in input data")

        # Make predictions
        predictions = self.models[model_name].predict(X)

        return predictions

    def predict_ensemble(self, features: Dict[str, float],
                        weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Get predictions from all models and compute ensemble averages

        Parameters:
        -----------
        features : dict
            Dictionary of feature values
        weights : dict, optional
            Custom weights for weighted ensemble (must sum to 1.0)
            Example: {'decision_tree': 0.1, 'random_forest': 0.3, ...}
            If None, uses default performance-based weights

        Returns:
        --------
        dict
            Dictionary containing:
            - Individual model predictions (e.g., 'xgboost': 35.2)
            - 'ensemble_mean': Simple average of all models
            - 'ensemble_weighted': Weighted average (performance-based or custom)
            - 'ensemble_std': Standard deviation across models
            - 'ensemble_min': Minimum prediction
            - 'ensemble_max': Maximum prediction

        Example:
        --------
        >>> features = {'NY.GDP.PCAP.PP.CD': 15000, ...}
        >>> results = predictor.predict_ensemble(features)
        >>> print(f"XGBoost: {results['xgboost']:.2f}")
        >>> print(f"Ensemble (weighted): {results['ensemble_weighted']:.2f}")
        >>> print(f"Prediction range: {results['ensemble_min']:.2f} - {results['ensemble_max']:.2f}")

        >>> # With custom weights
        >>> custom_weights = {
        ...     'decision_tree': 0.05,
        ...     'random_forest': 0.25,
        ...     'gradient_boosting': 0.25,
        ...     'xgboost': 0.225,
        ...     'lightgbm': 0.225
        ... }
        >>> results = predictor.predict_ensemble(features, weights=custom_weights)
        """
        # Validate custom weights if provided
        if weights is not None:
            if not isinstance(weights, dict):
                raise TypeError(f"ERROR: Weights must be a dictionary, got {type(weights).__name__}")

            # Check all models are present
            missing_models = set(self.models.keys()) - set(weights.keys())
            if missing_models:
                raise ValueError(
                    f"ERROR: Weights missing for models: {', '.join(missing_models)}\n"
                    f"       All models must have weights: {', '.join(self.models.keys())}"
                )

            # Check weights sum to 1.0 (with small tolerance)
            weight_sum = sum(weights.values())
            if not (0.99 <= weight_sum <= 1.01):
                raise ValueError(
                    f"ERROR: Weights must sum to 1.0, got {weight_sum:.4f}\n"
                    f"       Adjust weights so they sum to exactly 1.0"
                )

        predictions = {}

        # Get prediction from each model
        for model_name in self.models.keys():
            predictions[model_name] = self.predict_single(features, model_name)

        # Compute ensemble statistics
        pred_values = list(predictions.values())

        predictions['ensemble_mean'] = float(np.mean(pred_values))
        predictions['ensemble_std'] = float(np.std(pred_values))
        predictions['ensemble_min'] = float(np.min(pred_values))
        predictions['ensemble_max'] = float(np.max(pred_values))

        # Weighted ensemble
        if weights is not None:
            # Use custom weights
            predictions['ensemble_weighted'] = float(sum(
                predictions[model] * weights[model]
                for model in self.models.keys()
            ))
        else:
            # Default performance-based weights
            # Based on typical performance: DT=5%, RF=20%, GB=20%, XGB=27.5%, LGBM=27.5%
            default_weights = {
                'decision_tree': 0.05,
                'random_forest': 0.20,
                'gradient_boosting': 0.20,
                'xgboost': 0.275,
                'lightgbm': 0.275
            }
            predictions['ensemble_weighted'] = float(sum(
                predictions.get(model, 0) * weight
                for model, weight in default_weights.items()
                if model in predictions
            ))

        return predictions

    def predict_from_csv(self, input_file: str, output_file: str = None,
                        model_name: str = 'xgboost') -> pd.DataFrame:
        """
        Make predictions from CSV file and save results

        Parameters:
        -----------
        input_file : str
            Path to input CSV file with features
        output_file : str
            Path to save predictions CSV (default: input_file with '_predictions' suffix)
        model_name : str
            Name of model to use for predictions

        Returns:
        --------
        pd.DataFrame
            DataFrame with original data plus 'predicted_gini' column
        """
        # Auto-generate output filename if not provided
        if output_file is None:
            output_file = input_file.replace('.csv', '_predictions.csv')

        if self.verbose:
            print(f"Loading data from {input_file}...")

        try:
            data = pd.read_csv(input_file)
            if self.verbose:
                print(f"Loaded {len(data)} rows")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"ERROR: Input file not found: {input_file}\n"
                f"       Please check the file path and try again."
            )
        except Exception as e:
            raise Exception(f"ERROR: Error reading CSV file: {e}")

        # Validate model selection
        if model_name not in self.models:
            available = ', '.join(self.models.keys())
            raise ValueError(
                f"ERROR: Model '{model_name}' not available.\n"
                f"       Available models: {available}"
            )

        # Check for missing feature columns
        missing_cols = set(self.feature_names) - set(data.columns)
        if missing_cols:
            raise ValueError(
                f"ERROR: Input data is missing {len(missing_cols)} required feature(s):\n"
                f"       {', '.join(list(missing_cols)[:5])}" +
                (f"... and {len(missing_cols)-5} more" if len(missing_cols) > 5 else "")
            )

        if self.verbose:
            print(f"Making predictions with {model_name}...")

        # Extract features in correct order
        X = data[self.feature_names].values

        # Check for NaN values
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            if self.verbose:
                print(f"WARNING: Found {nan_count} NaN values in input data")

        # Make predictions
        predictions = self.models[model_name].predict(X)

        # Add predictions to dataframe
        data['predicted_gini'] = predictions

        # Save results
        try:
            data.to_csv(output_file, index=False)
            if self.verbose:
                print(f"Predictions saved to {output_file}")
                print(f"Statistics:")
                print(f"  Predictions made: {len(predictions)}")
                print(f"  GINI range: {predictions.min():.2f} - {predictions.max():.2f}")
                print(f"  Mean GINI: {predictions.mean():.2f}")
                print(f"  Median GINI: {np.median(predictions):.2f}")
        except Exception as e:
            raise Exception(f"ERROR: Error saving predictions: {e}")

        return data


def print_help():
    """Print help message"""
    print("\n" + "="*70)
    print("GINI Coefficient Predictor - Help")
    print("="*70)
    print("\nDESCRIPTION:")
    print("  Make GINI coefficient predictions using trained machine learning models")
    print("  Supports both command-line and programmatic usage")
    print("\nCOMMAND-LINE USAGE:")
    print("  python src/04_predict.py [OPTIONS]")
    print("\nOPTIONS:")
    print("  --csv FILE          Input CSV file with feature data")
    print("  --output FILE       Output CSV file (default: INPUT_predictions.csv)")
    print("  --model MODEL       Model to use (default: xgboost)")
    print("                      Options: decision_tree, random_forest,")
    print("                               gradient_boosting, xgboost, lightgbm")
    print("  --help, -h          Show this help message")
    print("\nCOMMAND-LINE EXAMPLES:")
    print("  # Basic prediction")
    print("  python src/04_predict.py --csv data.csv")
    print("\n  # Specify output file and model")
    print("  python src/04_predict.py --csv data.csv --output results.csv --model random_forest")
    print("\nPROGRAMMATIC USAGE:")
    print("  from src.04_predict import GINIPredictor")
    print("  ")
    print("  # Initialize predictor")
    print("  predictor = GINIPredictor()")
    print("  ")
    print("  # Single prediction")
    print("  features = {'NY.GDP.PCAP.PP.CD': 15000, 'SL.UEM.TOTL.ZS': 8.5, ...}")
    print("  gini = predictor.predict_single(features, model_name='xgboost')")
    print("  ")
    print("  # Batch prediction")
    print("  data = pd.read_csv('data.csv')")
    print("  predictions = predictor.predict_batch(data, model_name='random_forest')")
    print("  ")
    print("  # Ensemble prediction")
    print("  results = predictor.predict_ensemble(features)")
    print("  print(f\"Weighted ensemble: {results['ensemble_weighted']:.2f}\")")
    print("\nHELPER METHODS:")
    print("  predictor.get_available_models()      # List available models")
    print("  predictor.get_required_features()     # List required feature columns")
    print("\nREQUIREMENTS:")
    print("  • Input CSV must contain all required World Bank indicator columns")
    print("  • Column names must match indicator codes (e.g., 'NY.GDP.PCAP.PP.CD')")
    print("  • No missing values allowed in feature columns")
    print("  • Models loaded from output/.temp_models.pkl or trained on-demand")
    print("="*70 + "\n")


def main():
    """Main execution function"""
    import sys

    # Check for help flag
    if '--help' in sys.argv or '-h' in sys.argv:
        print_help()
        return

    # Check for CSV flag
    if '--csv' not in sys.argv:
        print("ERROR: Missing required argument --csv")
        print("\nUsage: python src/04_predict.py --csv input.csv")
        print("Run with --help for more information\n")
        return

    # Parse arguments
    try:
        csv_idx = sys.argv.index('--csv')
        if csv_idx + 1 >= len(sys.argv):
            print("ERROR: --csv flag requires a file path")
            print("Usage: python src/04_predict.py --csv input.csv\n")
            return
        input_file = sys.argv[csv_idx + 1]
    except ValueError:
        print("ERROR: Invalid arguments")
        print("Run with --help for usage information\n")
        return

    # Optional output file
    output_file = None
    if '--output' in sys.argv:
        out_idx = sys.argv.index('--output')
        if out_idx + 1 < len(sys.argv):
            output_file = sys.argv[out_idx + 1]

    # Optional model selection
    model_name = 'xgboost'
    if '--model' in sys.argv:
        model_idx = sys.argv.index('--model')
        if model_idx + 1 < len(sys.argv):
            model_name = sys.argv[model_idx + 1]

    # Run predictions
    print("\n" + "="*70)
    print("GINI Coefficient Prediction")
    print("="*70 + "\n")

    try:
        predictor = GINIPredictor()
        predictor.predict_from_csv(input_file, output_file, model_name)
        print("\nPrediction completed successfully!\n")

    except FileNotFoundError as e:
        print(f"\n{e}\n")
    except ValueError as e:
        print(f"\n{e}\n")
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}\n")
        import traceback
        if '--debug' in sys.argv:
            traceback.print_exc()


if __name__ == "__main__":
    main()
