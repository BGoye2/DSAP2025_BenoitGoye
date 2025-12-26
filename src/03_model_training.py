"""
Machine Learning Models for GINI Coefficient Prediction
Implements Decision Tree, Random Forest, and Gradient Boosting models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')


class GINIPredictor:
    """Train and evaluate tree-based models for GINI prediction"""
    
    def __init__(self, data_path: str = 'output/processed_data.csv'):
        """
        Initialize predictor
        
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
        self.scaler = StandardScaler()
        
        self.models = {}
        self.results = {}
        
    def load_and_split_data(self, test_size: float = 0.2, 
                           random_state: int = 42,
                           scale_features: bool = True) -> None:
        """
        Load data and split into train/test sets
        
        Parameters:
        -----------
        test_size : float
            Proportion of data for testing
        random_state : int
            Random seed for reproducibility
        scale_features : bool
            Whether to standardize features
        """
        print("Loading processed data...")
        data = pd.read_csv(self.data_path)
        
        # Load feature names
        feature_names_df = pd.read_csv('output/feature_names.csv')
        self.feature_names = feature_names_df['feature'].tolist()
        
        # Separate features and target
        X = data[self.feature_names].values
        y = data['SI.POV.GINI'].values
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features if requested
        if scale_features:
            print("Scaling features...")
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\nData split complete:")
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Number of features: {len(self.feature_names)}")
    
    def train_decision_tree(self, tune_hyperparameters: bool = True) -> None:
        """
        Train Decision Tree Regressor
        
        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        """
        print("\n" + "="*60)
        print("TRAINING DECISION TREE REGRESSOR")
        print("="*60)
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            param_grid = {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None]
            }
            
            dt = DecisionTreeRegressor(random_state=42)
            grid_search = GridSearchCV(
                dt, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"\nBest parameters: {grid_search.best_params_}")
            self.models['decision_tree'] = grid_search.best_estimator_
        else:
            print("Training with default parameters...")
            self.models['decision_tree'] = DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42
            )
            self.models['decision_tree'].fit(self.X_train, self.y_train)
        
        # Evaluate
        self._evaluate_model('decision_tree')
    
    def train_random_forest(self, tune_hyperparameters: bool = True) -> None:
        """
        Train Random Forest Regressor
        
        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST REGRESSOR")
        print("="*60)
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"\nBest parameters: {grid_search.best_params_}")
            self.models['random_forest'] = grid_search.best_estimator_
        else:
            print("Training with default parameters...")
            self.models['random_forest'] = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            self.models['random_forest'].fit(self.X_train, self.y_train)
        
        # Evaluate
        self._evaluate_model('random_forest')
    
    def train_gradient_boosting(self, tune_hyperparameters: bool = True) -> None:
        """
        Train Gradient Boosting Regressor
        
        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        """
        print("\n" + "="*60)
        print("TRAINING GRADIENT BOOSTING REGRESSOR")
        print("="*60)
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            gb = GradientBoostingRegressor(random_state=42)
            grid_search = GridSearchCV(
                gb, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"\nBest parameters: {grid_search.best_params_}")
            self.models['gradient_boosting'] = grid_search.best_estimator_
        else:
            print("Training with default parameters...")
            self.models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.9,
                random_state=42
            )
            self.models['gradient_boosting'].fit(self.X_train, self.y_train)
        
        # Evaluate
        self._evaluate_model('gradient_boosting')
    
    def train_xgboost(self, tune_hyperparameters: bool = True) -> None:
        """
        Train XGBoost Regressor
        
        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        """
        print("\n" + "="*60)
        print("TRAINING XGBOOST REGRESSOR")
        print("="*60)
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7, 9],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2]
            }
            
            xgboost = xgb.XGBRegressor(
                random_state=42,
                n_jobs=-1,
                tree_method='hist'
            )
            grid_search = GridSearchCV(
                xgboost, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"\nBest parameters: {grid_search.best_params_}")
            self.models['xgboost'] = grid_search.best_estimator_
        else:
            print("Training with default parameters...")
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=3,
                subsample=0.9,
                colsample_bytree=0.9,
                gamma=0.1,
                random_state=42,
                n_jobs=-1,
                tree_method='hist'
            )
            self.models['xgboost'].fit(self.X_train, self.y_train)
        
        # Evaluate
        self._evaluate_model('xgboost')
    
    def train_lightgbm(self, tune_hyperparameters: bool = True) -> None:
        """
        Train LightGBM Regressor
        
        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        """
        print("\n" + "="*60)
        print("TRAINING LIGHTGBM REGRESSOR")
        print("="*60)
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7, 9, -1],
                'num_leaves': [31, 50, 70, 100],
                'min_child_samples': [20, 30, 50],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            }
            
            lightgbm = lgb.LGBMRegressor(
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            grid_search = GridSearchCV(
                lightgbm, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"\nBest parameters: {grid_search.best_params_}")
            self.models['lightgbm'] = grid_search.best_estimator_
        else:
            print("Training with default parameters...")
            self.models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=50,
                min_child_samples=20,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            self.models['lightgbm'].fit(self.X_train, self.y_train)
        
        # Evaluate
        self._evaluate_model('lightgbm')
    
    def _evaluate_model(self, model_name: str) -> None:
        """
        Evaluate a trained model
        
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
        
        # Cross-validation score
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train, 
            cv=5, scoring='neg_mean_squared_error'
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
    
    def compare_models(self) -> pd.DataFrame:
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
        comparison_df.to_csv('output/model_comparison.csv', index=False)
        print("\nComparison saved to: model_comparison.csv")
        
        return comparison_df
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """
        Plot feature importance for tree-based models
        
        Parameters:
        -----------
        top_n : int
            Number of top features to display
        """
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6))
        
        # Handle single model case
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:top_n]
                
                axes[idx].barh(range(top_n), importances[indices])
                axes[idx].set_yticks(range(top_n))
                axes[idx].set_yticklabels([self.feature_names[i] for i in indices])
                axes[idx].set_xlabel('Importance')
                axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nTop {top_n} Features')
                axes[idx].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('output/feature_importance.png', dpi=300, bbox_inches='tight')
        print("\nFeature importance plot saved to: feature_importance.png")
        plt.close()
    
    def plot_predictions(self) -> None:
        """Plot actual vs predicted values for all models"""
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
        plt.savefig('output/predictions_plot.png', dpi=300, bbox_inches='tight')
        print("Predictions plot saved to: predictions_plot.png")
        plt.close()
    
    def plot_residuals(self) -> None:
        """Plot residuals for all models"""
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
        plt.savefig('output/residuals_plot.png', dpi=300, bbox_inches='tight')
        print("Residuals plot saved to: residuals_plot.png")
        plt.close()
    
    def save_models(self) -> None:
        """Models are kept in memory only - no disk saving"""
        print("\nModels trained and ready in memory (not saved to disk)")
        print("Note: Models will be retrained when needed for predictions")


def main():
    """Main execution function"""
    
    # Initialize predictor
    predictor = GINIPredictor('output/processed_data.csv')
    
    # Load and split data
    predictor.load_and_split_data(test_size=0.2, scale_features=False)
    
    # Train models (set tune_hyperparameters=True for better results, but slower)
    predictor.train_decision_tree(tune_hyperparameters=False)
    predictor.train_random_forest(tune_hyperparameters=False)
    predictor.train_gradient_boosting(tune_hyperparameters=False)
    predictor.train_xgboost(tune_hyperparameters=False)
    predictor.train_lightgbm(tune_hyperparameters=False)
    
    # Compare models
    comparison = predictor.compare_models()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    predictor.plot_feature_importance(top_n=15)
    predictor.plot_predictions()
    predictor.plot_residuals()
    
    # Note: Models not saved to disk
    print("\nModels trained successfully (kept in memory, not saved as .pkl files)")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - output/model_comparison.csv")
    print("  - output/feature_importance.png")
    print("  - output/predictions_plot.png")
    print("  - output/residuals_plot.png")
    print("\nNote: Models are NOT saved as .pkl files.")
    print("They will be retrained on-demand when needed for predictions.")


if __name__ == "__main__":
    main()
