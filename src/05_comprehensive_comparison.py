"""
Comprehensive Model Comparison and Statistical Analysis

This script performs in-depth evaluation and comparison of all trained models,
going beyond basic metrics to provide statistical rigor and actionable insights.

Analysis Components:

1. Comprehensive Metrics:
   - Training and test performance (RMSE, MAE, R², MAPE)
   - Cross-validation scores with confidence intervals
   - Residual analysis (mean, std, skewness, kurtosis)
   - Maximum and median absolute errors

2. Statistical Comparison:
   - Paired t-tests comparing absolute errors between models
   - Wilcoxon signed-rank tests (non-parametric alternative)
   - Identifies statistically significant performance differences

3. Segmentation Analysis:
   - Performance breakdown by GINI ranges (low, moderate, high inequality)
   - Identifies where each model excels or struggles

4. Visualization:
   - Side-by-side metric comparisons
   - Prediction scatter plots for all models
   - Residual distributions and Q-Q plots
   - Box plots of error distributions

5. Summary Report:
   - Model rankings by multiple criteria
   - Best model recommendations
   - Detailed performance statistics

Input: output/trained_models.pkl
Output: comprehensive_metrics.csv, statistical_comparison.csv, segment_performance.csv,
        output/figures/comprehensive_comparison.png, output/figures/error_analysis.png,
        output/figures/segment_performance.png, model_comparison_report.txt
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_validate

from config.constants import FIGURES_DIR, MODELS_PATH, OUTPUT_DIR, PROCESSED_DATA_PATH

warnings.filterwarnings('ignore')


class ModelComparator:
    """Comprehensive comparison of trained models"""

    def __init__(self, models=None, X_train=None, X_test=None, y_train=None, y_test=None,
                 feature_names=None, models_path=None):
        """
        Initialize comparator

        Parameters:
        -----------
        models : dict, optional
            Pre-trained models from GINIPredictor (keys: model names, values: fitted models)
            If provided, skips loading from file
        X_train : np.ndarray, optional
            Training features from GINIPredictor
        X_test : np.ndarray, optional
            Test features from GINIPredictor
        y_train : np.ndarray, optional
            Training target from GINIPredictor
        y_test : np.ndarray, optional
            Test target from GINIPredictor
        feature_names : list, optional
            Feature names from GINIPredictor
        models_path : str
            Path to trained models file
        """
        if models_path is None:
            models_path = MODELS_PATH
        self.predictions = {}
        self.metrics = {}

        # Priority 1: Use pre-trained models if provided directly
        if models is not None and X_train is not None:
            print("Using pre-trained models from GINIPredictor...")
            self.models = self._convert_model_names(models)
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.feature_names = feature_names
            print(f"✓ Loaded {len(self.models)} pre-trained models")

        # Priority 2: Load from models file
        else:
            if not os.path.exists(models_path):
                raise FileNotFoundError(
                    f"Trained models not found at {models_path}\n"
                    "Please run 03_model_training.py first to train and save models."
                )
            print(f"Loading pre-trained models from {models_path}...")
            self._load_from_pickle(models_path)

    def _load_from_pickle(self, filepath):
        """Load models and data from pickle file"""
        save_data = joblib.load(filepath)

        # Convert model names to expected format
        self.models = self._convert_model_names(save_data['models'])
        self.X_train = save_data['X_train']
        self.X_test = save_data['X_test']
        self.y_train = save_data['y_train']
        self.y_test = save_data['y_test']
        self.feature_names = save_data['feature_names']

        print(f"✓ Loaded {len(self.models)} pre-trained models")

    def _convert_model_names(self, models):
        """Convert model keys from script 03 format to script 05 format"""
        name_mapping = {
            'decision_tree': 'Decision Tree',
            'random_forest': 'Random Forest',
            'gradient_boosting': 'Gradient Boosting',
            'xgboost': 'XGBoost',
            'lightgbm': 'LightGBM'
        }
        return {name_mapping.get(k, k.title()): v for k, v in models.items()}
    
    def calculate_comprehensive_metrics(self):
        """Calculate comprehensive metrics for all models"""
        print("\n" + "="*80)
        print("CALCULATING COMPREHENSIVE METRICS")
        print("="*80)
        
        for model_name, model in self.models.items():
            print(f"\nAnalyzing {model_name}...")
            
            # Make predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            self.predictions[model_name] = {
                'train': y_train_pred,
                'test': y_test_pred
            }
            
            # Calculate various metrics
            train_metrics = self._calculate_metrics(self.y_train, y_train_pred)
            test_metrics = self._calculate_metrics(self.y_test, y_test_pred)
            
            # Cross-validation metrics (parallelized)
            cv_results = cross_validate(
                model, self.X_train, self.y_train,
                cv=5,
                scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
                return_train_score=True,
                n_jobs=-1  # Parallel cross-validation
            )
            
            cv_metrics = {
                'cv_rmse_mean': np.sqrt(-cv_results['test_neg_mean_squared_error'].mean()),
                'cv_rmse_std': np.sqrt(-cv_results['test_neg_mean_squared_error'].std()),
                'cv_mae_mean': -cv_results['test_neg_mean_absolute_error'].mean(),
                'cv_mae_std': -cv_results['test_neg_mean_absolute_error'].std(),
                'cv_r2_mean': cv_results['test_r2'].mean(),
                'cv_r2_std': cv_results['test_r2'].std(),
            }
            
            # Store all metrics
            self.metrics[model_name] = {
                'train': train_metrics,
                'test': test_metrics,
                'cv': cv_metrics,
                'overfit_gap': train_metrics['rmse'] - test_metrics['rmse']
            }
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate all regression metrics"""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'r2': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred),
            'max_error': np.max(np.abs(y_true - y_pred)),
            'median_abs_error': np.median(np.abs(y_true - y_pred)),
        }
        
        # Additional statistics
        residuals = y_true - y_pred
        metrics['residuals_mean'] = np.mean(residuals)
        metrics['residuals_std'] = np.std(residuals)
        metrics['residuals_skewness'] = stats.skew(residuals)
        metrics['residuals_kurtosis'] = stats.kurtosis(residuals)
        
        return metrics
    
    def create_metrics_table(self):
        """Create comprehensive metrics comparison table"""
        print("\n" + "="*80)
        print("METRICS COMPARISON TABLE")
        print("="*80)
        
        rows = []
        for model_name, metrics in self.metrics.items():
            rows.append({
                'Model': model_name,
                'Train RMSE': metrics['train']['rmse'],
                'Test RMSE': metrics['test']['rmse'],
                'Train MAE': metrics['train']['mae'],
                'Test MAE': metrics['test']['mae'],
                'Test MAPE (%)': metrics['test']['mape'],
                'Train R²': metrics['train']['r2'],
                'Test R²': metrics['test']['r2'],
                'CV RMSE': metrics['cv']['cv_rmse_mean'],
                'CV R² (mean)': metrics['cv']['cv_r2_mean'],
                'CV R² (std)': metrics['cv']['cv_r2_std'],
                'Overfit Gap': metrics['overfit_gap'],
                'Max Error': metrics['test']['max_error'],
                'Median Abs Error': metrics['test']['median_abs_error']
            })
        
        df = pd.DataFrame(rows)
        df = df.round(3)

        print("\n", df.to_string(index=False))

        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, 'comprehensive_metrics.csv')
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved to: {output_path}")
        
        return df
    
    def statistical_comparison(self):
        """Perform statistical tests to compare models"""
        print("\n" + "="*80)
        print("STATISTICAL COMPARISON")
        print("="*80)
        
        # Prepare data for comparison
        model_names = list(self.models.keys())
        
        # Paired t-tests on residuals
        print("\nPaired t-tests (comparing absolute errors):")
        print("-" * 80)
        
        results = []
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                abs_error1 = np.abs(self.y_test - self.predictions[model1]['test'])
                abs_error2 = np.abs(self.y_test - self.predictions[model2]['test'])
                
                t_stat, p_value = stats.ttest_rel(abs_error1, abs_error2)
                
                results.append({
                    'Comparison': f"{model1} vs {model2}",
                    't-statistic': t_stat,
                    'p-value': p_value,
                    'Significant (α=0.05)': 'Yes' if p_value < 0.05 else 'No',
                    'Better Model': model1 if t_stat < 0 else model2
                })
        
        comparison_df = pd.DataFrame(results)
        print(comparison_df.to_string(index=False))
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        print("\n\nWilcoxon Signed-Rank Tests:")
        print("-" * 80)
        
        wilcoxon_results = []
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                abs_error1 = np.abs(self.y_test - self.predictions[model1]['test'])
                abs_error2 = np.abs(self.y_test - self.predictions[model2]['test'])
                
                stat, p_value = stats.wilcoxon(abs_error1, abs_error2)
                
                wilcoxon_results.append({
                    'Comparison': f"{model1} vs {model2}",
                    'Statistic': stat,
                    'p-value': p_value,
                    'Significant (α=0.05)': 'Yes' if p_value < 0.05 else 'No'
                })
        
        wilcoxon_df = pd.DataFrame(wilcoxon_results)
        print(wilcoxon_df.to_string(index=False))

        # Save statistical results
        output_path = os.path.join(OUTPUT_DIR, 'statistical_comparison.csv')
        comparison_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved to: {output_path}")
        
        return comparison_df, wilcoxon_df
    
    def analyze_by_segments(self):
        """Analyze model performance across different GINI ranges"""
        print("\n" + "="*80)
        print("PERFORMANCE BY GINI SEGMENTS")
        print("="*80)
        
        # Define GINI segments
        segments = [
            (0, 30, 'Low Inequality (0-30)'),
            (30, 40, 'Moderate Inequality (30-40)'),
            (40, 50, 'High Inequality (40-50)'),
            (50, 100, 'Very High Inequality (50+)')
        ]
        
        segment_results = []
        
        for lower, upper, label in segments:
            mask = (self.y_test >= lower) & (self.y_test < upper)
            n_samples = mask.sum()
            
            if n_samples == 0:
                continue
            
            print(f"\n{label} ({n_samples} samples):")
            print("-" * 80)
            
            for model_name, model in self.models.items():
                y_true_seg = self.y_test[mask]
                y_pred_seg = self.predictions[model_name]['test'][mask]
                
                rmse = np.sqrt(mean_squared_error(y_true_seg, y_pred_seg))
                mae = mean_absolute_error(y_true_seg, y_pred_seg)
                r2 = r2_score(y_true_seg, y_pred_seg)
                
                segment_results.append({
                    'Segment': label,
                    'Model': model_name,
                    'N Samples': n_samples,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R²': r2
                })
                
                print(f"  {model_name:20s}: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")

        segment_df = pd.DataFrame(segment_results)
        output_path = os.path.join(OUTPUT_DIR, 'segment_performance.csv')
        segment_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved to: {output_path}")
        
        return segment_df
    
    def plot_comprehensive_comparison(self):
        """Create comprehensive visualization comparing all models"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        model_names = list(self.models.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(model_names)]
        
        # 1. RMSE Comparison (Train vs Test)
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(len(model_names))
        width = 0.35
        train_rmse = [self.metrics[m]['train']['rmse'] for m in model_names]
        test_rmse = [self.metrics[m]['test']['rmse'] for m in model_names]
        
        ax1.bar(x - width/2, train_rmse, width, label='Train', alpha=0.8)
        ax1.bar(x + width/2, test_rmse, width, label='Test', alpha=0.8)
        ax1.set_ylabel('RMSE')
        ax1.set_title('RMSE Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. R² Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        train_r2 = [self.metrics[m]['train']['r2'] for m in model_names]
        test_r2 = [self.metrics[m]['test']['r2'] for m in model_names]
        
        ax2.bar(x - width/2, train_r2, width, label='Train', alpha=0.8)
        ax2.bar(x + width/2, test_r2, width, label='Test', alpha=0.8)
        ax2.set_ylabel('R² Score')
        ax2.set_title('R² Score Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. MAE Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        train_mae = [self.metrics[m]['train']['mae'] for m in model_names]
        test_mae = [self.metrics[m]['test']['mae'] for m in model_names]
        
        ax3.bar(x - width/2, train_mae, width, label='Train', alpha=0.8)
        ax3.bar(x + width/2, test_mae, width, label='Test', alpha=0.8)
        ax3.set_ylabel('MAE')
        ax3.set_title('MAE Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Prediction Scatter Plots (Combined)
        ax4 = fig.add_subplot(gs[1, :])
        for i, model_name in enumerate(model_names):
            y_pred = self.predictions[model_name]['test']
            ax4.scatter(self.y_test, y_pred, alpha=0.5, label=model_name, 
                       color=colors[i], s=30)
        
        min_val, max_val = self.y_test.min(), self.y_test.max()
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, 
                label='Perfect Prediction')
        ax4.set_xlabel('Actual GINI')
        ax4.set_ylabel('Predicted GINI')
        ax4.set_title('Predictions: All Models')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Residual Distribution
        ax5 = fig.add_subplot(gs[2, 0])
        for i, model_name in enumerate(model_names):
            residuals = self.y_test - self.predictions[model_name]['test']
            ax5.hist(residuals, bins=30, alpha=0.5, label=model_name, color=colors[i])
        
        ax5.set_xlabel('Residuals')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Residual Distribution')
        ax5.axvline(x=0, color='red', linestyle='--', lw=2)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Box Plot of Absolute Errors
        ax6 = fig.add_subplot(gs[2, 1])
        abs_errors = [np.abs(self.y_test - self.predictions[m]['test']) 
                     for m in model_names]
        bp = ax6.boxplot(abs_errors, labels=model_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax6.set_ylabel('Absolute Error')
        ax6.set_title('Error Distribution (Box Plot)')
        ax6.set_xticklabels(model_names, rotation=45, ha='right')
        ax6.grid(True, alpha=0.3)
        
        # 7. Cross-Validation Scores
        ax7 = fig.add_subplot(gs[2, 2])
        cv_r2_means = [self.metrics[m]['cv']['cv_r2_mean'] for m in model_names]
        cv_r2_stds = [self.metrics[m]['cv']['cv_r2_std'] for m in model_names]
        
        ax7.bar(x, cv_r2_means, yerr=cv_r2_stds, alpha=0.8, 
               color=colors, capsize=5)
        ax7.set_ylabel('R² Score')
        ax7.set_title('Cross-Validation R² (mean ± std)')
        ax7.set_xticks(x)
        ax7.set_xticklabels(model_names, rotation=45, ha='right')
        ax7.grid(True, alpha=0.3)

        output_path = os.path.join(FIGURES_DIR, 'comprehensive_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved to: {output_path}")
        plt.close()
    
    def plot_error_analysis(self):
        """Detailed error analysis plots"""
        model_names = list(self.models.keys())
        n_models = len(model_names)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(2*n_rows, n_cols, figsize=(6*n_cols, 5*2*n_rows))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:n_models]
        
        # Flatten axes for easier indexing
        if n_models == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        elif n_cols == 1:
            axes = axes.reshape(2*n_rows, 1)
        
        # 1. Absolute Error vs Actual GINI
        for i, model_name in enumerate(model_names):
            row = i // n_cols
            col = i % n_cols
            
            abs_error = np.abs(self.y_test - self.predictions[model_name]['test'])
            axes[row, col].scatter(self.y_test, abs_error, alpha=0.6, 
                             color=colors[i], s=30)
            axes[row, col].set_xlabel('Actual GINI')
            axes[row, col].set_ylabel('Absolute Error')
            axes[row, col].set_title(f'{model_name}\nError vs Actual')
            axes[row, col].grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(self.y_test, abs_error, 1)
            p = np.poly1d(z)
            axes[row, col].plot(self.y_test, p(self.y_test), "r--", alpha=0.8, lw=2)
        
        # 2. Q-Q Plot for Residuals
        for i, model_name in enumerate(model_names):
            row = n_rows + i // n_cols
            col = i % n_cols
            
            residuals = self.y_test - self.predictions[model_name]['test']
            stats.probplot(residuals, dist="norm", plot=axes[row, col])
            axes[row, col].set_title(f'{model_name}\nQ-Q Plot')
            axes[row, col].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_models, n_cols * n_rows):
            row_idx = i // n_cols
            col_idx = i % n_cols
            axes[row_idx, col_idx].set_visible(False)
            axes[n_rows + row_idx, col_idx].set_visible(False)

        plt.tight_layout()
        output_path = os.path.join(FIGURES_DIR, 'error_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {output_path}")
        plt.close()
    
    def plot_segment_performance(self):
        """Plot performance across GINI segments"""
        segment_path = os.path.join(OUTPUT_DIR, 'segment_performance.csv')
        segment_df = pd.read_csv(segment_path)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics_to_plot = ['RMSE', 'MAE', 'R²']
        for idx, metric in enumerate(metrics_to_plot):
            pivot = segment_df.pivot(index='Segment', columns='Model', values=metric)
            pivot.plot(kind='bar', ax=axes[idx], rot=45)
            axes[idx].set_title(f'{metric} by GINI Segment')
            axes[idx].set_ylabel(metric)
            axes[idx].legend(title='Model')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(FIGURES_DIR, 'segment_performance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {output_path}")
        plt.close()
    
    def generate_summary_report(self):
        """Generate a comprehensive text summary report"""
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE MODEL COMPARISON REPORT")
        report.append("="*80)
        report.append(f"\nGenerated: {pd.Timestamp.now()}")
        report.append(f"Dataset: {PROCESSED_DATA_PATH}")
        report.append(f"Number of models: {len(self.models)}")
        report.append(f"Test set size: {len(self.y_test)}")
        
        # Best model by each metric
        report.append("\n" + "="*80)
        report.append("BEST MODELS BY METRIC")
        report.append("="*80)
        
        metric_names = {
            'rmse': 'RMSE (lower is better)',
            'mae': 'MAE (lower is better)',
            'r2': 'R² (higher is better)',
            'mape': 'MAPE (lower is better)'
        }
        
        for metric_key, metric_label in metric_names.items():
            if metric_key == 'r2':
                best_model = max(self.metrics.items(), 
                               key=lambda x: x[1]['test'][metric_key])
            else:
                best_model = min(self.metrics.items(), 
                               key=lambda x: x[1]['test'][metric_key])
            
            report.append(f"\n{metric_label}:")
            report.append(f"  Best: {best_model[0]} = {best_model[1]['test'][metric_key]:.3f}")
        
        # Model rankings
        report.append("\n" + "="*80)
        report.append("OVERALL MODEL RANKING (by Test R²)")
        report.append("="*80)
        
        ranked = sorted(self.metrics.items(), 
                       key=lambda x: x[1]['test']['r2'], 
                       reverse=True)
        
        for rank, (model_name, metrics) in enumerate(ranked, 1):
            report.append(f"\n{rank}. {model_name}")
            report.append(f"   Test R²: {metrics['test']['r2']:.4f}")
            report.append(f"   Test RMSE: {metrics['test']['rmse']:.4f}")
            report.append(f"   Test MAE: {metrics['test']['mae']:.4f}")
            report.append(f"   Overfit Gap: {metrics['overfit_gap']:.4f}")
        
        # Detailed statistics
        report.append("\n" + "="*80)
        report.append("DETAILED MODEL STATISTICS")
        report.append("="*80)
        
        for model_name, metrics in self.metrics.items():
            report.append(f"\n{model_name}:")
            report.append("-" * 80)
            report.append(f"  Training Performance:")
            report.append(f"    RMSE: {metrics['train']['rmse']:.4f}")
            report.append(f"    MAE: {metrics['train']['mae']:.4f}")
            report.append(f"    R²: {metrics['train']['r2']:.4f}")
            
            report.append(f"\n  Test Performance:")
            report.append(f"    RMSE: {metrics['test']['rmse']:.4f}")
            report.append(f"    MAE: {metrics['test']['mae']:.4f}")
            report.append(f"    MAPE: {metrics['test']['mape']:.2f}%")
            report.append(f"    R²: {metrics['test']['r2']:.4f}")
            report.append(f"    Max Error: {metrics['test']['max_error']:.4f}")
            report.append(f"    Median Abs Error: {metrics['test']['median_abs_error']:.4f}")
            
            report.append(f"\n  Cross-Validation:")
            report.append(f"    RMSE: {metrics['cv']['cv_rmse_mean']:.4f} ± {metrics['cv']['cv_rmse_std']:.4f}")
            report.append(f"    R²: {metrics['cv']['cv_r2_mean']:.4f} ± {metrics['cv']['cv_r2_std']:.4f}")
            
            report.append(f"\n  Residual Analysis:")
            report.append(f"    Mean: {metrics['test']['residuals_mean']:.4f}")
            report.append(f"    Std: {metrics['test']['residuals_std']:.4f}")
            report.append(f"    Skewness: {metrics['test']['residuals_skewness']:.4f}")
            report.append(f"    Kurtosis: {metrics['test']['residuals_kurtosis']:.4f}")
        
        # Recommendations
        report.append("\n" + "="*80)
        report.append("RECOMMENDATIONS")
        report.append("="*80)
        
        best_overall = ranked[0][0]
        report.append(f"\n✓ Best Overall Model: {best_overall}")
        report.append(f"  Recommended for: General predictions")
        
        best_simple = min(
            [(name, met) for name, met in self.metrics.items() 
             if 'Tree' in name],
            key=lambda x: x[1]['test']['rmse'],
            default=None
        )
        
        if best_simple:
            report.append(f"\n✓ Best Interpretable Model: {best_simple[0]}")
            report.append(f"  Recommended for: When interpretability is crucial")
        
        report_text = "\n".join(report)

        # Save report
        output_path = os.path.join(OUTPUT_DIR, 'model_comparison_report.txt')
        with open(output_path, 'w') as f:
            f.write(report_text)

        print("\n" + report_text)
        print(f"\n✓ Saved to: {output_path}")
        
        return report_text
    
    def run_full_comparison(self):
        """Run all comparison analyses"""
        print("\n" + "="*80)
        print("RUNNING COMPREHENSIVE MODEL COMPARISON")
        print("="*80)
        
        # Calculate metrics
        self.calculate_comprehensive_metrics()
        
        # Create tables
        self.create_metrics_table()
        
        # Statistical comparison
        self.statistical_comparison()
        
        # Segment analysis
        self.analyze_by_segments()
        
        # Generate visualizations
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        self.plot_comprehensive_comparison()
        self.plot_error_analysis()
        self.plot_segment_performance()
        
        # Generate summary report
        self.generate_summary_report()
        
        print("\n" + "="*80)
        print("COMPARISON COMPLETE!")
        print("="*80)
        print("\nGenerated files:")
        print(f"  ✓ {os.path.join(OUTPUT_DIR, 'comprehensive_metrics.csv')}")
        print(f"  ✓ {os.path.join(OUTPUT_DIR, 'statistical_comparison.csv')}")
        print(f"  ✓ {os.path.join(OUTPUT_DIR, 'segment_performance.csv')}")
        print(f"  ✓ {os.path.join(FIGURES_DIR, 'comprehensive_comparison.png')}")
        print(f"  ✓ {os.path.join(FIGURES_DIR, 'error_analysis.png')}")
        print(f"  ✓ {os.path.join(FIGURES_DIR, 'segment_performance.png')}")
        print(f"  ✓ {os.path.join(OUTPUT_DIR, 'model_comparison_report.txt')}")


def main():
    """Main execution"""
    comparator = ModelComparator()
    comparator.run_full_comparison()


if __name__ == "__main__":
    main()
