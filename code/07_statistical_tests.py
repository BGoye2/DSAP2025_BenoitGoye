"""
Statistical Significance Tests for Feature Importance
Performs permutation-based tests and bootstrap confidence intervals for feature importance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureImportanceTester:
    """Statistical tests for feature importance significance and stability"""

    def __init__(self, data_path: str = 'output/processed_data.csv'):
        """
        Initialize tester

        Parameters:
        -----------
        data_path : str
            Path to processed data
        """
        self.data_path = data_path
        self.data = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_split_data(self) -> None:
        """Load data and create train/test split"""
        print("Loading data...")
        self.data = pd.read_csv(self.data_path)

        # Load feature names
        try:
            feature_names_df = pd.read_csv('output/feature_names.csv')
            self.feature_names = feature_names_df['feature'].tolist()
        except FileNotFoundError:
            self.feature_names = [col for col in self.data.columns if col != 'SI.POV.GINI']

        # Prepare data
        X = self.data[self.feature_names].values
        y = self.data['SI.POV.GINI'].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Loaded {len(X)} observations with {len(self.feature_names)} features")
        print(f"Train: {len(self.X_train)}, Test: {len(self.X_test)}")

    def bootstrap_feature_importance(self, model_name: str = 'XGBoost',
                                     n_iterations: int = 100) -> Dict:
        """
        Calculate bootstrap confidence intervals for feature importance

        Parameters:
        -----------
        model_name : str
            Model to use ('RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM')
        n_iterations : int
            Number of bootstrap iterations

        Returns:
        --------
        dict
            Feature importance statistics with confidence intervals
        """
        print(f"\n{'='*60}")
        print(f"Bootstrap Feature Importance Analysis ({model_name})")
        print(f"{'='*60}")
        print(f"Running {n_iterations} bootstrap iterations...")

        # Initialize model
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1, verbose=-1)
        }

        if model_name not in models:
            raise ValueError(f"Model {model_name} not supported")

        # Store importance from each iteration
        importance_samples = []

        for i in range(n_iterations):
            if (i + 1) % 20 == 0:
                print(f"  Iteration {i+1}/{n_iterations}...")

            # Bootstrap sample
            indices = np.random.choice(len(self.X_train), size=len(self.X_train), replace=True)
            X_boot = self.X_train[indices]
            y_boot = self.y_train[indices]

            # Train model
            model = models[model_name]
            if hasattr(model, 'random_state'):
                model.random_state = i  # Different seed for each iteration

            model.fit(X_boot, y_boot)

            # Get feature importance
            importance = model.feature_importances_
            importance_samples.append(importance)

        importance_samples = np.array(importance_samples)

        # Calculate statistics
        mean_importance = np.mean(importance_samples, axis=0)
        std_importance = np.std(importance_samples, axis=0)
        ci_lower = np.percentile(importance_samples, 2.5, axis=0)
        ci_upper = np.percentile(importance_samples, 97.5, axis=0)

        # Organize results
        results = []
        for i, feature in enumerate(self.feature_names):
            results.append({
                'feature': feature,
                'mean_importance': mean_importance[i],
                'std_importance': std_importance[i],
                'ci_lower': ci_lower[i],
                'ci_upper': ci_upper[i],
                'ci_width': ci_upper[i] - ci_lower[i],
                'cv': std_importance[i] / mean_importance[i] if mean_importance[i] > 0 else np.inf
            })

        # Sort by mean importance
        results = sorted(results, key=lambda x: x['mean_importance'], reverse=True)

        print(f"\nTop 10 Features with 95% Confidence Intervals:")
        print(f"{'Feature':<40} {'Mean':<10} {'95% CI':<20} {'Significant':<12}")
        print("-" * 85)

        for r in results[:10]:
            ci_str = f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
            # Feature is "significant" if CI doesn't include 0
            significant = "Yes" if r['ci_lower'] > 0 else "No"
            print(f"{r['feature'][:38]:<40} {r['mean_importance']:<10.4f} {ci_str:<20} {significant:<12}")

        return {'model': model_name, 'results': results, 'raw_samples': importance_samples}

    def permutation_importance_test(self, model_name: str = 'XGBoost',
                                   n_permutations: int = 50,
                                   top_k: int = 20) -> Dict:
        """
        Perform permutation test for feature importance significance

        Parameters:
        -----------
        model_name : str
            Model to use
        n_permutations : int
            Number of permutations per feature
        top_k : int
            Test only top K features (for computational efficiency)

        Returns:
        --------
        dict
            Permutation test results with p-values
        """
        print(f"\n{'='*60}")
        print(f"Permutation Importance Test ({model_name})")
        print(f"{'='*60}")

        # Train baseline model
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1, verbose=-1)
        }

        print("Training baseline model...")
        model = models[model_name]
        model.fit(self.X_train, self.y_train)

        # Baseline performance
        baseline_score = r2_score(self.y_test, model.predict(self.X_test))
        print(f"Baseline R² on test set: {baseline_score:.4f}")

        # Get top K features
        baseline_importance = model.feature_importances_
        top_indices = np.argsort(baseline_importance)[-top_k:][::-1]

        print(f"\nTesting top {top_k} features with {n_permutations} permutations each...")

        results = []

        for idx in top_indices:
            feature = self.feature_names[idx]

            # Permute this feature and measure performance drop
            score_drops = []

            for perm in range(n_permutations):
                # Create permuted test set
                X_test_perm = self.X_test.copy()
                X_test_perm[:, idx] = np.random.permutation(X_test_perm[:, idx])

                # Evaluate on permuted data
                perm_score = r2_score(self.y_test, model.predict(X_test_perm))
                score_drop = baseline_score - perm_score
                score_drops.append(score_drop)

            score_drops = np.array(score_drops)

            # Calculate statistics
            mean_drop = np.mean(score_drops)
            std_drop = np.std(score_drops)

            # One-sided t-test: is mean drop significantly > 0?
            t_stat, p_value = stats.ttest_1samp(score_drops, 0, alternative='greater')

            results.append({
                'feature': feature,
                'baseline_importance': baseline_importance[idx],
                'mean_score_drop': mean_drop,
                'std_score_drop': std_drop,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })

        # Sort by mean score drop
        results = sorted(results, key=lambda x: x['mean_score_drop'], reverse=True)

        print(f"\nPermutation Test Results (α=0.05):")
        print(f"{'Feature':<40} {'Score Drop':<15} {'p-value':<12} {'Significant':<12}")
        print("-" * 80)

        for r in results[:15]:
            sig_str = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else "ns"
            print(f"{r['feature'][:38]:<40} {r['mean_score_drop']:<15.4f} {r['p_value']:<12.4f} {sig_str:<12}")

        return {'model': model_name, 'results': results}

    def cross_model_consistency_test(self) -> Dict:
        """
        Test consistency of feature importance rankings across models using Spearman correlation

        Returns:
        --------
        dict
            Correlation matrix and consistency scores
        """
        print(f"\n{'='*60}")
        print("Cross-Model Feature Importance Consistency Test")
        print(f"{'='*60}")

        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1, verbose=-1)
        }

        # Train all models and collect importance
        importance_dict = {}

        for model_name, model in models.items():
            print(f"Training {model_name}...")
            model.fit(self.X_train, self.y_train)
            importance_dict[model_name] = model.feature_importances_

        # Calculate pairwise Spearman correlations
        model_names = list(importance_dict.keys())
        n_models = len(model_names)
        correlation_matrix = np.zeros((n_models, n_models))
        pvalue_matrix = np.zeros((n_models, n_models))

        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                    pvalue_matrix[i, j] = 0.0
                else:
                    corr, pval = spearmanr(importance_dict[model1], importance_dict[model2])
                    correlation_matrix[i, j] = corr
                    pvalue_matrix[i, j] = pval

        print("\nSpearman Rank Correlation Between Models:")
        print(f"{'Model':<20}", end='')
        for name in model_names:
            print(f"{name[:15]:<18}", end='')
        print()
        print("-" * (20 + 18 * n_models))

        for i, name1 in enumerate(model_names):
            print(f"{name1[:18]:<20}", end='')
            for j in range(n_models):
                sig = "***" if pvalue_matrix[i, j] < 0.001 else "**" if pvalue_matrix[i, j] < 0.01 else "*" if pvalue_matrix[i, j] < 0.05 else ""
                print(f"{correlation_matrix[i, j]:.3f}{sig:<13}", end='')
            print()

        # Overall consistency score (mean off-diagonal correlation)
        off_diag_mask = ~np.eye(n_models, dtype=bool)
        mean_correlation = np.mean(correlation_matrix[off_diag_mask])

        print(f"\nOverall consistency score: {mean_correlation:.3f}")
        print("Interpretation: Higher values indicate more consistent feature rankings across models")

        return {
            'correlation_matrix': correlation_matrix,
            'pvalue_matrix': pvalue_matrix,
            'model_names': model_names,
            'mean_correlation': mean_correlation,
            'importance_dict': importance_dict
        }

    def create_bootstrap_plot(self, bootstrap_results: Dict, output_file: str,
                             top_k: int = 15) -> None:
        """
        Visualize bootstrap confidence intervals for top features

        Parameters:
        -----------
        bootstrap_results : dict
            Results from bootstrap_feature_importance
        output_file : str
            Path to save plot
        top_k : int
            Number of top features to plot
        """
        results = bootstrap_results['results'][:top_k]

        features = [r['feature'][:30] + '...' if len(r['feature']) > 30 else r['feature'] for r in results]
        means = [r['mean_importance'] for r in results]
        ci_lowers = [r['ci_lower'] for r in results]
        ci_uppers = [r['ci_upper'] for r in results]

        # Calculate error bars
        yerr_lower = [means[i] - ci_lowers[i] for i in range(len(means))]
        yerr_upper = [ci_uppers[i] - means[i] for i in range(len(means))]

        fig, ax = plt.subplots(figsize=(10, 8))

        y_pos = np.arange(len(features))

        ax.barh(y_pos, means, xerr=[yerr_lower, yerr_upper], capsize=5,
               color='steelblue', alpha=0.7, ecolor='darkred', linewidth=1.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance', fontsize=11)
        ax.set_title(f'Top {top_k} Features with 95% Bootstrap Confidence Intervals\n({bootstrap_results["model"]})',
                    fontsize=12, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved bootstrap plot to {output_file}")
        plt.close()

    def create_correlation_heatmap(self, consistency_results: Dict, output_file: str) -> None:
        """
        Create heatmap of cross-model consistency correlations

        Parameters:
        -----------
        consistency_results : dict
            Results from cross_model_consistency_test
        output_file : str
            Path to save plot
        """
        fig, ax = plt.subplots(figsize=(8, 7))

        sns.heatmap(consistency_results['correlation_matrix'],
                   xticklabels=consistency_results['model_names'],
                   yticklabels=consistency_results['model_names'],
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlGn',
                   vmin=0,
                   vmax=1,
                   cbar_kws={'label': 'Spearman Correlation'},
                   ax=ax,
                   square=True)

        ax.set_title('Feature Importance Ranking Consistency Across Models\n(Spearman Rank Correlation)',
                    fontsize=12, fontweight='bold', pad=15)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved correlation heatmap to {output_file}")
        plt.close()

    def save_results_to_csv(self, bootstrap_results: Dict, permutation_results: Dict,
                           consistency_results: Dict) -> None:
        """Save all statistical test results to CSV files"""

        # Bootstrap results
        df_bootstrap = pd.DataFrame(bootstrap_results['results'])
        df_bootstrap.to_csv('output/statistical_tests_bootstrap.csv', index=False)
        print("Saved bootstrap results to output/statistical_tests_bootstrap.csv")

        # Permutation test results
        df_permutation = pd.DataFrame(permutation_results['results'])
        df_permutation.to_csv('output/statistical_tests_permutation.csv', index=False)
        print("Saved permutation test results to output/statistical_tests_permutation.csv")

        # Consistency results
        df_consistency = pd.DataFrame(
            consistency_results['correlation_matrix'],
            columns=consistency_results['model_names'],
            index=consistency_results['model_names']
        )
        df_consistency.to_csv('output/statistical_tests_consistency.csv')
        print("Saved consistency matrix to output/statistical_tests_consistency.csv")

    def create_summary_report(self, bootstrap_results: Dict, permutation_results: Dict,
                             consistency_results: Dict) -> None:
        """Create comprehensive text summary report"""

        with open('output/statistical_tests_summary.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write("STATISTICAL SIGNIFICANCE TESTS FOR FEATURE IMPORTANCE\n")
            f.write("="*70 + "\n\n")

            # Bootstrap Analysis
            f.write("1. BOOTSTRAP CONFIDENCE INTERVALS\n")
            f.write("-"*70 + "\n")
            f.write(f"Model: {bootstrap_results['model']}\n")
            f.write(f"Number of bootstrap iterations: {len(bootstrap_results['raw_samples'])}\n\n")
            f.write("Top 10 features with 95% confidence intervals:\n\n")
            f.write(f"{'Rank':<6} {'Feature':<40} {'Mean Imp.':<12} {'95% CI':<25}\n")
            f.write("-"*85 + "\n")

            for i, r in enumerate(bootstrap_results['results'][:10], 1):
                ci_str = f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
                f.write(f"{i:<6} {r['feature'][:38]:<40} {r['mean_importance']:<12.4f} {ci_str:<25}\n")

            f.write("\nInterpretation: Features whose 95% CI does not include zero are\n")
            f.write("statistically significantly different from zero importance.\n\n")

            # Permutation Test
            f.write("\n2. PERMUTATION IMPORTANCE TEST\n")
            f.write("-"*70 + "\n")
            f.write(f"Model: {permutation_results['model']}\n\n")
            f.write("Features ranked by performance drop when permuted:\n\n")
            f.write(f"{'Rank':<6} {'Feature':<40} {'Score Drop':<15} {'p-value':<10}\n")
            f.write("-"*75 + "\n")

            for i, r in enumerate(permutation_results['results'][:10], 1):
                sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else "ns"
                f.write(f"{i:<6} {r['feature'][:38]:<40} {r['mean_score_drop']:<15.4f} {r['p_value']:<10.4f} {sig}\n")

            f.write("\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05, ns not significant\n")
            f.write("Interpretation: Larger score drops indicate more important features.\n\n")

            # Cross-model consistency
            f.write("\n3. CROSS-MODEL CONSISTENCY\n")
            f.write("-"*70 + "\n")
            f.write(f"Overall consistency score: {consistency_results['mean_correlation']:.3f}\n\n")
            f.write("Spearman rank correlation matrix:\n\n")

            model_names = consistency_results['model_names']
            corr_matrix = consistency_results['correlation_matrix']

            f.write(f"{'Model':<20}", )
            for name in model_names:
                f.write(f"{name[:15]:<18}")
            f.write("\n" + "-"*90 + "\n")

            for i, name1 in enumerate(model_names):
                f.write(f"{name1[:18]:<20}")
                for j in range(len(model_names)):
                    f.write(f"{corr_matrix[i, j]:<18.3f}")
                f.write("\n")

            f.write("\nInterpretation: High correlations (>0.7) indicate consistent feature\n")
            f.write("importance rankings across different modeling approaches.\n\n")

            # Key Findings
            f.write("\n4. KEY FINDINGS\n")
            f.write("-"*70 + "\n\n")

            # Count significant features
            n_significant_bootstrap = sum(1 for r in bootstrap_results['results'] if r['ci_lower'] > 0)
            n_significant_permutation = sum(1 for r in permutation_results['results'] if r['significant'])

            f.write(f"- {n_significant_bootstrap} features have 95% CIs excluding zero\n")
            f.write(f"- {n_significant_permutation} features show significant permutation importance (p<0.05)\n")
            f.write(f"- Mean cross-model correlation: {consistency_results['mean_correlation']:.3f}\n\n")

            # Top consensus features
            f.write("Top 5 consensus features (appear in top 10 for all tests):\n")

            # Find features in top 10 for both tests
            top10_bootstrap = set(r['feature'] for r in bootstrap_results['results'][:10])
            top10_permutation = set(r['feature'] for r in permutation_results['results'][:10])
            consensus = top10_bootstrap & top10_permutation

            for i, feat in enumerate(list(consensus)[:5], 1):
                f.write(f"  {i}. {feat}\n")

            f.write("\n")

        print("Saved statistical tests summary to output/statistical_tests_summary.txt")

    def run_full_analysis(self) -> None:
        """Run all statistical tests"""
        print("="*60)
        print("STATISTICAL SIGNIFICANCE TESTS FOR FEATURE IMPORTANCE")
        print("="*60)

        # Load data
        self.load_and_split_data()

        # Bootstrap analysis
        bootstrap_results = self.bootstrap_feature_importance(model_name='XGBoost', n_iterations=100)
        self.create_bootstrap_plot(bootstrap_results, 'output/statistical_tests_bootstrap.png')

        # Permutation test
        permutation_results = self.permutation_importance_test(model_name='XGBoost', n_permutations=50)

        # Cross-model consistency
        consistency_results = self.cross_model_consistency_test()
        self.create_correlation_heatmap(consistency_results, 'output/statistical_tests_consistency.png')

        # Save results
        self.save_results_to_csv(bootstrap_results, permutation_results, consistency_results)
        self.create_summary_report(bootstrap_results, permutation_results, consistency_results)

        print("\n" + "="*60)
        print("STATISTICAL TESTS COMPLETE")
        print("="*60)
        print("\nGenerated files:")
        print("  - output/statistical_tests_bootstrap.csv")
        print("  - output/statistical_tests_bootstrap.png")
        print("  - output/statistical_tests_permutation.csv")
        print("  - output/statistical_tests_consistency.csv")
        print("  - output/statistical_tests_consistency.png")
        print("  - output/statistical_tests_summary.txt")


def main():
    """Main execution"""
    tester = FeatureImportanceTester()
    tester.run_full_analysis()


if __name__ == "__main__":
    main()
