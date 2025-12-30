"""
Regional and Income-Level Segmentation Analysis

This script investigates whether model performance and feature importance vary across
different country contexts, providing insights for context-specific policy interventions.

Segmentation Approaches:

1. Income-Level Segmentation:
   - Divides countries into quartiles based on GDP per capita
   - Categories: Low Income, Lower-Middle Income, Upper-Middle Income, High Income
   - Reveals whether inequality drivers differ by development stage

2. Regional Segmentation:
   - Groups countries by World Bank regional classification
   - Regions: East Asia & Pacific, Europe & Central Asia, Latin America & Caribbean,
             Middle East & North Africa, North America, South Asia, Sub-Saharan Africa
   - Identifies geographic patterns in inequality predictors

Analysis for Each Segment:
- Train separate models (Random Forest, XGBoost, LightGBM)
- Compare performance metrics (RMSE, MAE, R²)
- Identify top 10 most important features
- Generate visualizations showing performance and feature importance variations

Key Insights:
- Do models perform better in developed vs developing economies?
- Are inequality drivers universal or context-specific?
- Which features matter most in different regions/income levels?

Input: output/processed_data.csv, output/feature_names.csv
Output: segmentation_income_results.csv, output/figures/segmentation_income_performance.png,
        output/figures/segmentation_income_features.png, segmentation_regional_results.csv,
        output/figures/segmentation_regional_performance.png,
        output/figures/segmentation_regional_features.png, segmentation_summary_report.txt
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

from config.constants import TARGET_VARIABLE, COUNTRY_REGIONS_PATH


# Load World Bank Regional Classification from JSON
# Source: https://datahelpdesk.worldbank.org/knowledgebase/articles/906519-world-bank-country-and-lending-groups
def load_country_regions() -> Dict[str, str]:
    """
    Load country-to-region mapping from JSON configuration file.

    Returns:
    --------
    Dict[str, str]
        Mapping of country codes to region names
    """
    try:
        with open(COUNTRY_REGIONS_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {COUNTRY_REGIONS_PATH} not found. Using empty regional mapping.")
        return {}

COUNTRY_REGIONS = load_country_regions()


class SegmentationAnalyzer:
    """Analyze model performance and feature importance across country segments"""

    def __init__(self, data_path: str = 'output/processed_data.csv'):
        """
        Initialize analyzer

        Parameters:
        -----------
        data_path : str
            Path to processed data
        """
        self.data_path = data_path
        self.data = None
        self.feature_names = None
        self.models = {}
        self.segment_results = {}

    def load_data(self):
        """Load and prepare data"""
        print("Loading data...")
        self.data = pd.read_csv(self.data_path)

        # Load feature names
        try:
            feature_names_df = pd.read_csv('output/feature_names.csv')
            self.feature_names = feature_names_df['feature'].tolist()
        except FileNotFoundError:
            self.feature_names = [col for col in self.data.columns if col != TARGET_VARIABLE]

        print(f"Loaded {len(self.data)} observations with {len(self.feature_names)} features")

    def create_income_segments(self):
        """
        Create income-level segments based on GDP per capita

        Returns:
        --------
        pd.Series
            Income segment labels for each observation
        """
        # Find GDP per capita column
        gdp_cols = [col for col in self.feature_names if 'NY.GDP.PCAP' in col or 'gdp' in col.lower()]

        if not gdp_cols:
            print("Warning: GDP per capita column not found. Using quartiles of first numeric column.")
            gdp_col = self.feature_names[0]
        else:
            gdp_col = gdp_cols[0]

        print(f"Using {gdp_col} for income segmentation")

        # Create quartile-based segments
        gdp_values = self.data[gdp_col]
        quartiles = gdp_values.quantile([0.25, 0.5, 0.75])

        segments = pd.cut(gdp_values,
                         bins=[-np.inf, quartiles[0.25], quartiles[0.5], quartiles[0.75], np.inf],
                         labels=['Low Income', 'Lower-Middle Income', 'Upper-Middle Income', 'High Income'])

        print("\nIncome segment distribution:")
        print(segments.value_counts().sort_index())

        return segments

    def create_regional_segments(self):
        """
        Create regional segments based on World Bank regional classification

        Returns:
        --------
        pd.Series
            Regional segment labels
        """
        # Map country codes to regions
        segments = self.data['country_code'].map(COUNTRY_REGIONS)

        # Handle unmapped countries (assign to "Other")
        unmapped_count = segments.isna().sum()
        if unmapped_count > 0:
            print(f"\nWarning: {unmapped_count} observations with unmapped country codes")
            unique_unmapped = self.data.loc[segments.isna(), 'country_code'].unique()
            print(f"Unmapped country codes: {', '.join(unique_unmapped[:10])}")
            if len(unique_unmapped) > 10:
                print(f"... and {len(unique_unmapped) - 10} more")
            segments = segments.fillna('Other')

        print("\nRegional segment distribution:")
        print(segments.value_counts().sort_index())

        return segments

    def train_models_by_segment(self, segments: pd.Series, segment_name: str):
        """
        Train models for each segment and evaluate performance

        Parameters:
        -----------
        segments : pd.Series
            Segment labels for each observation
        segment_name : str
            Name of segmentation (e.g., 'Income Level', 'Region')

        Returns:
        --------
        dict
            Results for each segment
        """
        print(f"\n{'='*60}")
        print(f"Training models by {segment_name}")
        print(f"{'='*60}")

        X = self.data[self.feature_names].values
        y = self.data[TARGET_VARIABLE].values

        results = {}

        for segment_value in segments.dropna().unique():
            print(f"\n--- Analyzing {segment_value} ---")

            # Filter data for this segment
            segment_mask = segments == segment_value
            X_segment = X[segment_mask]
            y_segment = y[segment_mask]

            print(f"Segment size: {len(y_segment)} observations")

            if len(y_segment) < 30:
                print(f"Warning: Segment too small ({len(y_segment)} < 30). Skipping.")
                continue

            # Split data
            test_size = min(0.2, max(0.1, 10 / len(y_segment)))  # Adaptive test size
            X_train, X_test, y_train, y_test = train_test_split(
                X_segment, y_segment, test_size=test_size, random_state=42
            )

            segment_results = {
                'n_observations': len(y_segment),
                'n_train': len(y_train),
                'n_test': len(y_test),
                'mean_gini': np.mean(y_segment),
                'std_gini': np.std(y_segment),
                'models': {}
            }

            # Train models on segment-specific data
            # NOTE: We intentionally train NEW models here (not reusing pre-trained ones)
            # because the purpose of segmentation analysis is to see how models perform
            # when trained on different subsets of data (income levels/regions)
            models_to_train = {
                'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
                'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1),
                'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1, verbose=-1)
            }

            for model_name, model in models_to_train.items():
                print(f"  Training {model_name}...")

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Get feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    top_10_indices = np.argsort(importance)[-10:][::-1]
                    top_10_features = [(self.feature_names[i], importance[i]) for i in top_10_indices]
                else:
                    top_10_features = []

                segment_results['models'][model_name] = {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'feature_importance': top_10_features
                }

                print(f"    RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")

            results[segment_value] = segment_results

        return results

    def create_performance_comparison_plot(self, results: Dict, segment_name: str,
                                          output_file: str):
        """
        Create visualization comparing model performance across segments

        Parameters:
        -----------
        results : dict
            Results from train_models_by_segment
        segment_name : str
            Name of segmentation
        output_file : str
            Path to save plot
        """
        print(f"\nCreating performance comparison plot...")

        # Prepare data for plotting
        segments = []
        models = []
        rmse_values = []
        r2_values = []

        for segment, data in results.items():
            for model_name, metrics in data['models'].items():
                segments.append(str(segment))
                models.append(model_name)
                rmse_values.append(metrics['rmse'])
                r2_values.append(metrics['r2'])

        df_plot = pd.DataFrame({
            'Segment': segments,
            'Model': models,
            'RMSE': rmse_values,
            'R²': r2_values
        })

        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # RMSE comparison
        sns.barplot(data=df_plot, x='Segment', y='RMSE', hue='Model', ax=axes[0])
        axes[0].set_title(f'RMSE by {segment_name}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(segment_name, fontsize=10)
        axes[0].set_ylabel('RMSE (lower is better)', fontsize=10)
        axes[0].legend(title='Model', fontsize=8)
        axes[0].tick_params(axis='x', rotation=45)

        # R² comparison
        sns.barplot(data=df_plot, x='Segment', y='R²', hue='Model', ax=axes[1])
        axes[1].set_title(f'R² by {segment_name}', fontsize=12, fontweight='bold')
        axes[1].set_xlabel(segment_name, fontsize=10)
        axes[1].set_ylabel('R² (higher is better)', fontsize=10)
        axes[1].legend(title='Model', fontsize=8)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
        plt.close()

    def create_feature_importance_heatmap(self, results: Dict, segment_name: str,
                                         output_file: str, model_name: str = 'XGBoost'):
        """
        Create heatmap showing how feature importance varies across segments

        Parameters:
        -----------
        results : dict
            Results from train_models_by_segment
        segment_name : str
            Name of segmentation
        output_file : str
            Path to save plot
        model_name : str
            Which model to visualize
        """
        print(f"\nCreating feature importance heatmap for {model_name}...")

        # Collect all features that appear in top 10 for any segment
        all_features = set()
        for segment_data in results.values():
            if model_name in segment_data['models']:
                for feat, _ in segment_data['models'][model_name]['feature_importance']:
                    all_features.add(feat)

        all_features = sorted(list(all_features))

        # Create importance matrix
        importance_matrix = []
        segment_labels = []

        for segment, segment_data in results.items():
            if model_name not in segment_data['models']:
                continue

            segment_labels.append(str(segment))
            importance_dict = dict(segment_data['models'][model_name]['feature_importance'])

            row = [importance_dict.get(feat, 0) for feat in all_features]
            importance_matrix.append(row)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(6, len(segment_labels) * 0.5)))

        importance_matrix = np.array(importance_matrix)

        sns.heatmap(importance_matrix,
                   xticklabels=[feat[:30] + '...' if len(feat) > 30 else feat for feat in all_features],
                   yticklabels=segment_labels,
                   cmap='YlOrRd',
                   annot=False,
                   fmt='.3f',
                   cbar_kws={'label': 'Feature Importance'},
                   ax=ax)

        ax.set_title(f'Feature Importance by {segment_name} ({model_name})',
                    fontsize=12, fontweight='bold', pad=20)
        ax.set_xlabel('Features', fontsize=10)
        ax.set_ylabel(segment_name, fontsize=10)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=9)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {output_file}")
        plt.close()

    def save_results_to_csv(self, results: Dict, segment_name: str, output_file: str):
        """
        Save segmentation results to CSV

        Parameters:
        -----------
        results : dict
            Results from train_models_by_segment
        segment_name : str
            Name of segmentation
        output_file : str
            Path to save CSV
        """
        rows = []

        for segment, data in results.items():
            for model_name, metrics in data['models'].items():
                row = {
                    'Segment_Type': segment_name,
                    'Segment': str(segment),
                    'N_Observations': data['n_observations'],
                    'Mean_GINI': data['mean_gini'],
                    'Std_GINI': data['std_gini'],
                    'Model': model_name,
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae'],
                    'R2': metrics['r2']
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"\nSaved results to {output_file}")

    def run_full_analysis(self):
        """Run complete segmentation analysis"""
        print("="*60)
        print("REGIONAL AND INCOME-LEVEL SEGMENTATION ANALYSIS")
        print("="*60)

        # Load data
        self.load_data()

        # Income-level analysis
        print("\n" + "="*60)
        print("INCOME-LEVEL SEGMENTATION")
        print("="*60)
        income_segments = self.create_income_segments()
        income_results = self.train_models_by_segment(income_segments, 'Income Level')

        self.create_performance_comparison_plot(
            income_results,
            'Income Level',
            'output/figures/segmentation_income_performance.png'
        )

        self.create_feature_importance_heatmap(
            income_results,
            'Income Level',
            'output/figures/segmentation_income_features.png',
            model_name='XGBoost'
        )

        self.save_results_to_csv(
            income_results,
            'Income Level',
            'output/segmentation_income_results.csv'
        )

        # Regional analysis
        print("\n" + "="*60)
        print("REGIONAL SEGMENTATION")
        print("="*60)
        regional_segments = self.create_regional_segments()
        regional_results = self.train_models_by_segment(regional_segments, 'Region')

        self.create_performance_comparison_plot(
            regional_results,
            'Region',
            'output/figures/segmentation_regional_performance.png'
        )

        self.create_feature_importance_heatmap(
            regional_results,
            'Region',
            'output/figures/segmentation_regional_features.png',
            model_name='XGBoost'
        )

        self.save_results_to_csv(
            regional_results,
            'Region',
            'output/segmentation_regional_results.csv'
        )

        # Summary report
        self.create_summary_report(income_results, regional_results)

        print("\n" + "="*60)
        print("SEGMENTATION ANALYSIS COMPLETE")
        print("="*60)
        print("\nGenerated files:")
        print("  - output/figures/segmentation_income_performance.png")
        print("  - output/figures/segmentation_income_features.png")
        print("  - output/segmentation_income_results.csv")
        print("  - output/figures/segmentation_regional_performance.png")
        print("  - output/figures/segmentation_regional_features.png")
        print("  - output/segmentation_regional_results.csv")
        print("  - output/segmentation_summary_report.txt")

    def create_summary_report(self, income_results: Dict, regional_results: Dict):
        """
        Create text summary report

        Parameters:
        -----------
        income_results : dict
            Income segmentation results
        regional_results : dict
            Regional segmentation results
        """
        with open('output/segmentation_summary_report.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write("SEGMENTATION ANALYSIS SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")

            # Income-level summary
            f.write("INCOME-LEVEL SEGMENTATION\n")
            f.write("-"*70 + "\n\n")

            for segment, data in income_results.items():
                f.write(f"{segment}:\n")
                f.write(f"  Sample size: {data['n_observations']} observations\n")
                f.write(f"  Mean GINI: {data['mean_gini']:.2f} (±{data['std_gini']:.2f})\n")
                f.write(f"  Model Performance:\n")

                for model_name, metrics in data['models'].items():
                    f.write(f"    {model_name}: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}\n")

                # Top 5 features for XGBoost
                if 'XGBoost' in data['models']:
                    f.write(f"  Top 5 Features (XGBoost):\n")
                    for i, (feat, imp) in enumerate(data['models']['XGBoost']['feature_importance'][:5], 1):
                        f.write(f"    {i}. {feat[:50]}: {imp:.4f}\n")

                f.write("\n")

            # Regional summary
            f.write("\n" + "="*70 + "\n")
            f.write("REGIONAL SEGMENTATION\n")
            f.write("-"*70 + "\n\n")

            for segment, data in regional_results.items():
                f.write(f"{segment}:\n")
                f.write(f"  Sample size: {data['n_observations']} observations\n")
                f.write(f"  Mean GINI: {data['mean_gini']:.2f} (±{data['std_gini']:.2f})\n")
                f.write(f"  Model Performance:\n")

                for model_name, metrics in data['models'].items():
                    f.write(f"    {model_name}: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}\n")

                # Top 5 features for XGBoost
                if 'XGBoost' in data['models']:
                    f.write(f"  Top 5 Features (XGBoost):\n")
                    for i, (feat, imp) in enumerate(data['models']['XGBoost']['feature_importance'][:5], 1):
                        f.write(f"    {i}. {feat[:50]}: {imp:.4f}\n")

                f.write("\n")

        print("Saved summary report to output/segmentation_summary_report.txt")


def main():
    """Main execution"""
    analyzer = SegmentationAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
