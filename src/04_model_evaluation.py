"""
Model Evaluation for GINI Coefficient Prediction
=================================================
This script evaluates trained models and generates performance metrics
and visualizations.

Features:
- Comprehensive performance metrics (RMSE, MAE, R²)
- Cross-validation scores
- Model comparison table
- Feature importance plots
- Prediction vs actual plots
- Residual analysis plots

Input: Trained models from output/trained_models.pkl
Output: output/model_comparison.csv, output/figures/*.png (visualizations)
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from config.constants import FIGURES_DIR, MODELS_PATH, OUTPUT_DIR
from config.feature_names import get_display_name

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Evaluate and visualize trained GINI models"""

    def __init__(self, models_path: str = None):
        """
        Initialize evaluator

        Parameters:
        -----------
        models_path : str
            Path to trained models file
        """
        if models_path is None:
            models_path = MODELS_PATH
        self.models = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.results = {}

        # Load pre-trained models
        if not os.path.exists(models_path):
            raise FileNotFoundError(
                f"Trained models not found at {models_path}\n"
                "Please run 03_model_training.py first to train and save models."
            )

        self._load_from_pickle(models_path)

    def _load_from_pickle(self, filepath: str):
        """Load models from pickle file"""
        print(f"Loading pre-trained models from {filepath}...")
        save_data = joblib.load(filepath)

        self.models = save_data['models']
        self.X_train = save_data['X_train']
        self.X_test = save_data['X_test']
        self.y_train = save_data['y_train']
        self.y_test = save_data['y_test']
        self.feature_names = save_data['feature_names']

        print(f"✓ Loaded {len(self.models)} pre-trained models")

    def evaluate_model(self, model_name: str):
        """
        Evaluate a single model

        Parameters:
        -----------
        model_name : str
            Name of the model to evaluate
        """
        model = self.models[model_name]

        # Make predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        # Calculate metrics
        train_metrics = {
            'rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'mae': mean_absolute_error(self.y_train, y_train_pred),
            'r2': r2_score(self.y_train, y_train_pred)
        }

        test_metrics = {
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'mae': mean_absolute_error(self.y_test, y_test_pred),
            'r2': r2_score(self.y_test, y_test_pred)
        }

        # Cross-validation score (parallelized)
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=5, scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        cv_rmse = np.sqrt(-cv_scores.mean())

        # Store results
        self.results[model_name] = {
            'train': train_metrics,
            'test': test_metrics,
            'cv_rmse': cv_rmse,
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred
            }
        }

        # Print results
        print(f"\n{model_name.upper()} RESULTS:")
        print("-" * 60)
        print(f"Training Set:")
        print(f"  RMSE: {train_metrics['rmse']:.3f}")
        print(f"  MAE:  {train_metrics['mae']:.3f}")
        print(f"  R²:   {train_metrics['r2']:.3f}")
        print(f"\nTest Set:")
        print(f"  RMSE: {test_metrics['rmse']:.3f}")
        print(f"  MAE:  {test_metrics['mae']:.3f}")
        print(f"  R²:   {test_metrics['r2']:.3f}")
        print(f"\nCross-Validation RMSE: {cv_rmse:.3f}")

    def evaluate_all_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*60)
        print("EVALUATING ALL MODELS")
        print("="*60)

        for model_name in self.models.keys():
            self.evaluate_model(model_name)

    def compare_models(self):
        """
        Compare all trained models

        Returns:
        --------
        pd.DataFrame
            Comparison of model performances
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)

        # Evaluate all models if not already done
        if not self.results:
            self.evaluate_all_models()

        comparison_data = []

        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Train RMSE': results['train']['rmse'],
                'Test RMSE': results['test']['rmse'],
                'Train R²': results['train']['r2'],
                'Test R²': results['test']['r2'],
                'CV RMSE': results['cv_rmse'],
                'Overfit Gap': results['train']['rmse'] - results['test']['rmse']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.round(3)

        print("\n", comparison_df.to_string(index=False))

        # Save comparison
        output_path = os.path.join(OUTPUT_DIR, 'model_comparison.csv')
        comparison_df.to_csv(output_path, index=False)
        print(f"\n✓ Comparison saved to: {output_path}")

        return comparison_df

    def plot_feature_importance(self, top_n: int = 20):
        """
        Plot feature importance for tree-based models in a two-column layout

        Parameters:
        -----------
        top_n : int
            Number of top features to display
        """
        n_models = len(self.models)

        # Create two-column layout
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))

        # Flatten axes array for easier indexing
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes

        for idx, (model_name, model) in enumerate(self.models.items()):
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:top_n]

                # Map feature codes to human-readable names
                feature_display_names = [
                    get_display_name(self.feature_names[i]) for i in indices
                ]

                axes[idx].barh(range(top_n), importances[indices], color='steelblue')
                axes[idx].set_yticks(range(top_n))
                axes[idx].set_yticklabels(feature_display_names, fontsize=10)
                axes[idx].set_xlabel('Importance', fontsize=12)
                axes[idx].set_title(
                    f'{model_name.replace("_", " ").title()}\nTop {top_n} Features',
                    fontsize=13,
                    fontweight='bold'
                )
                axes[idx].invert_yaxis()
                axes[idx].grid(axis='x', alpha=0.3)

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        output_path = os.path.join(FIGURES_DIR, 'feature_importance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Feature importance plot saved to: {output_path}")
        plt.close()

    def plot_predictions(self):
        """Plot actual vs predicted values for all models"""
        # Evaluate all models if not already done
        if not self.results:
            self.evaluate_all_models()

        n_models = len(self.results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))

        # Flatten axes array for easier indexing
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes

        for idx, (model_name, results) in enumerate(self.results.items()):
            y_true = self.y_test
            y_pred = results['predictions']['test']

            axes[idx].scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
            axes[idx].plot([y_true.min(), y_true.max()],
                          [y_true.min(), y_true.max()],
                          'r--', lw=2, label='Perfect Prediction')

            axes[idx].set_xlabel('Actual GINI')
            axes[idx].set_ylabel('Predicted GINI')
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\n'
                               f'R² = {results["test"]["r2"]:.3f}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        output_path = os.path.join(FIGURES_DIR, 'predictions_plot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Predictions plot saved to: {output_path}")
        plt.close()

    def plot_residuals(self):
        """Plot residuals for all models"""
        # Evaluate all models if not already done
        if not self.results:
            self.evaluate_all_models()

        n_models = len(self.results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))

        # Flatten axes array for easier indexing
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes

        for idx, (model_name, results) in enumerate(self.results.items()):
            y_true = self.y_test
            y_pred = results['predictions']['test']
            residuals = y_true - y_pred

            axes[idx].scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
            axes[idx].axhline(y=0, color='r', linestyle='--', lw=2)
            axes[idx].set_xlabel('Predicted GINI')
            axes[idx].set_ylabel('Residuals')
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nResidual Plot')
            axes[idx].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        output_path = os.path.join(FIGURES_DIR, 'residuals_plot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Residuals plot saved to: {output_path}")
        plt.close()


def main():
    """Main execution function"""

    # Initialize evaluator
    evaluator = ModelEvaluator()

    # Evaluate all models
    evaluator.evaluate_all_models()

    # Compare models
    comparison = evaluator.compare_models()

    # Generate visualizations
    print("\nGenerating visualizations...")
    evaluator.plot_feature_importance(top_n=15)
    evaluator.plot_predictions()
    evaluator.plot_residuals()

    print("\n" + "="*60)
    print("MODEL EVALUATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  - {os.path.join(OUTPUT_DIR, 'model_comparison.csv')}")
    print(f"  - {os.path.join(FIGURES_DIR, 'feature_importance.png')}")
    print(f"  - {os.path.join(FIGURES_DIR, 'predictions_plot.png')}")
    print(f"  - {os.path.join(FIGURES_DIR, 'residuals_plot.png')}")


if __name__ == "__main__":
    main()
