"""
LaTeX Table Population for Research Paper

This script automatically generates LaTeX tables from analysis results,
eliminating manual copy-paste errors and ensuring reproducibility.

Generated Tables:

1. Table 1 - Descriptive Statistics:
   - Summary statistics for key variables (GINI, GDP, education, health, etc.)
   - Includes N, mean, std dev, min, max, and missing data percentage
   - Provides overview of dataset characteristics

2. Table 2 - Model Performance Comparison:
   - Training and test RMSE for all models
   - Test R² scores
   - Cross-validation RMSE
   - Enables comparison of model accuracy

3. Table 3 - Top Features by Importance:
   - Feature importance rankings from XGBoost model
   - Shows top 10 most important predictors
   - Extracted directly from trained model

4. Table 4 - Model Performance by Income Level:
   - Segmented performance across income quartiles
   - Shows RMSE and R² for each income level
   - Reveals whether models work better for certain economic contexts
   - Uses multirow formatting for readability

4. Summary Text:
   - Dynamically generated summary statistics text
   - Updates automatically when data changes

Process:
1. Load results from CSV files (model_comparison, segmentation_income, etc.)
2. Generate LaTeX table code with proper formatting
3. Save to output/tables/*.tex files
4. Tables are imported by paper.tex using \\input{} commands

Input: output/processed_data.csv, output/model_comparison.csv,
       output/segmentation_income_results.csv, output/trained_models.pkl
Output: output/tables/table1_descriptive.tex, output/tables/table2_performance.tex,
        output/tables/table3_features.tex, output/tables/table4_income.tex,
        output/tables/summary_text.tex
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd

from config.constants import FEATURE_NAMES_PATH, MODELS_PATH, OUTPUT_DIR, PROCESSED_DATA_PATH, TABLES_DIR, get_display_name, TARGET_VARIABLE
from config.feature_categories import filter_features_by_category

warnings.filterwarnings('ignore')


class PaperTablePopulator:
    """Generates LaTeX table content from actual model results and updates paper"""

    def __init__(self):
        """Initialize table populator"""
        self.processed_data = None
        self.model_comparison = None
        self.income_results = None
        self.feature_names = None
        self.trained_models = None

    def load_data(self):
        """Load all necessary data files"""
        print("Loading data files...")

        try:
            self.processed_data = pd.read_csv(PROCESSED_DATA_PATH)
            print(f"✓ Loaded processed_data.csv: {len(self.processed_data)} rows")
        except FileNotFoundError:
            print("✗ processed_data.csv not found - run preprocessing first")
            return False

        try:
            model_comparison_path = os.path.join(OUTPUT_DIR, 'model_comparison.csv')
            self.model_comparison = pd.read_csv(model_comparison_path)
            print(f"✓ Loaded model_comparison.csv: {len(self.model_comparison)} rows")
        except FileNotFoundError:
            print("✗ model_comparison.csv not found - run model training first")
            return False

        try:
            income_results_path = os.path.join(OUTPUT_DIR, 'segmentation_income_results.csv')
            self.income_results = pd.read_csv(income_results_path)
            print(f"✓ Loaded segmentation_income_results.csv: {len(self.income_results)} rows")
        except FileNotFoundError:
            print("⚠ segmentation_income_results.csv not found - Table 4 will be skipped")
            self.income_results = None

        try:
            self.feature_names = pd.read_csv(FEATURE_NAMES_PATH)
            print(f"✓ Loaded feature_names.csv: {len(self.feature_names)} features")
        except FileNotFoundError:
            print("✗ feature_names.csv not found")
            return False

        try:
            self.trained_models = joblib.load(MODELS_PATH)
            print(f"✓ Loaded trained_models.pkl: {len(self.trained_models['models'])} models")
        except FileNotFoundError:
            print("⚠ trained_models.pkl not found - Table 3 will be skipped")
            self.trained_models = None

        return True

    def generate_descriptive_table(self):
        """Generate Table 1: Descriptive Statistics"""
        print("\nGenerating Table 1: Descriptive Statistics...")

        # Get key variables using feature categories
        gini = self.processed_data[TARGET_VARIABLE]

        # Find GDP per capita column
        gdp_cols = filter_features_by_category(self.processed_data.columns.tolist(), 'gdp')
        gdp_cols = [col for col in gdp_cols if 'NY.GDP.PCAP' in col and col in self.processed_data.columns]
        gdp = self.processed_data[gdp_cols[0]] if gdp_cols else None

        # Find education expenditure column
        edu_cols = filter_features_by_category(self.processed_data.columns.tolist(), 'education')
        edu_cols = [col for col in edu_cols if 'SE.XPD' in col and col in self.processed_data.columns]
        edu = self.processed_data[edu_cols[0]] if edu_cols else None

        # Find health expenditure column
        health_cols = filter_features_by_category(self.processed_data.columns.tolist(), 'health')
        health_cols = [col for col in health_cols if 'SH.XPD' in col and 'PC' in col and col in self.processed_data.columns]
        health = self.processed_data[health_cols[0]] if health_cols else None

        # Find unemployment column
        unemp_cols = filter_features_by_category(self.processed_data.columns.tolist(), 'labor')
        unemp_cols = [col for col in unemp_cols if 'SL.UEM.TOTL' in col and col in self.processed_data.columns]
        unemp = self.processed_data[unemp_cols[0]] if unemp_cols else None

        # Find urban population column
        urban_cols = filter_features_by_category(self.processed_data.columns.tolist(), 'demographics')
        urban_cols = [col for col in urban_cols if ('SP.URB.TOTL' in col or 'urbanization' in col.lower()) and col in self.processed_data.columns]
        urban = self.processed_data[urban_cols[0]] if urban_cols else None

        n_total = len(self.processed_data)

        latex = "\\begin{tabular}{lcccccc}\n"
        latex += "\\toprule\n"
        latex += "Variable & N & Mean & Std Dev & Min & Max & Missing \\% \\\\\n"
        latex += "\\midrule\n"

        # GINI row
        latex += f"GINI Coefficient & {n_total} & {gini.mean():.1f} & {gini.std():.1f} & {gini.min():.1f} & {gini.max():.1f} & 0.0\\% \\\\\n"

        # GDP per capita row
        if gdp is not None:
            missing_pct = (gdp.isna().sum() / n_total) * 100
            latex += f"GDP per capita (PPP) & {n_total} & {gdp.mean():.0f} & {gdp.std():.0f} & {gdp.min():.0f} & {gdp.max():.0f} & {missing_pct:.1f}\\% \\\\\n"

        # Education expenditure row
        if edu is not None:
            missing_pct = (edu.isna().sum() / n_total) * 100
            latex += f"Education expenditure (\\% GDP) & {n_total} & {edu.mean():.1f} & {edu.std():.1f} & {edu.min():.1f} & {edu.max():.1f} & {missing_pct:.1f}\\% \\\\\n"

        # Health expenditure row
        if health is not None:
            missing_pct = (health.isna().sum() / n_total) * 100
            latex += f"Health expenditure (per capita) & {n_total} & {health.mean():.0f} & {health.std():.0f} & {health.min():.0f} & {health.max():.0f} & {missing_pct:.1f}\\% \\\\\n"

        # Unemployment row
        if unemp is not None:
            missing_pct = (unemp.isna().sum() / n_total) * 100
            latex += f"Unemployment rate (\\%) & {n_total} & {unemp.mean():.1f} & {unemp.std():.1f} & {unemp.min():.1f} & {unemp.max():.1f} & {missing_pct:.1f}\\% \\\\\n"

        # Urban population row
        if urban is not None:
            missing_pct = (urban.isna().sum() / n_total) * 100
            latex += f"Urban population (\\%) & {n_total} & {urban.mean():.1f} & {urban.std():.1f} & {urban.min():.1f} & {urban.max():.1f} & {missing_pct:.1f}\\% \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}"

        print("✓ Table 1 generated")
        return latex

    def generate_performance_table(self):
        """Generate Table 2: Model Performance Comparison"""
        print("\nGenerating Table 2: Model Performance Comparison...")

        latex = "\\begin{tabularx}{\\columnwidth}{@{}X@{\\hspace{4pt}}c@{\\hspace{4pt}}c@{\\hspace{4pt}}c@{\\hspace{4pt}}c@{}}\n"
        latex += "\\toprule\n"
        latex += "Model & Train & Test & Test & CV \\\\\n"
        latex += " & RMSE & RMSE & R² & RMSE \\\\\n"
        latex += "\\midrule\n"

        for _, row in self.model_comparison.iterrows():
            model_name = self._clean_model_name(row['Model'])
            train_rmse = row['Train RMSE']
            test_rmse = row['Test RMSE']
            test_r2 = row['Test R²']
            cv_rmse = row['CV RMSE']

            latex += f"{model_name} & {train_rmse:.2f} & {test_rmse:.2f} & {test_r2:.3f} & {cv_rmse:.2f} \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabularx}"

        print("✓ Table 2 generated")
        return latex

    def generate_feature_importance_table(self, model_name: str = 'XGBoost', top_n: int = 10):
        """Generate Table 3: Top Features by Importance"""
        print(f"\nGenerating Table 3: Top {top_n} Features by Importance ({model_name})...")

        if self.trained_models is None:
            print("✗ No trained models available")
            return None

        # Get the model
        models = self.trained_models['models']
        if model_name not in models:
            print(f"✗ Model '{model_name}' not found in trained models")
            return None

        model = models[model_name]

        # Check if model has feature_importances_
        if not hasattr(model, 'feature_importances_'):
            print(f"✗ Model '{model_name}' does not have feature_importances_")
            return None

        # Get feature importances
        importances = model.feature_importances_
        feature_names = self.trained_models['feature_names']

        # Create dataframe and sort by importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        # Generate LaTeX table
        latex = "\\begin{tabularx}{\\columnwidth}{@{}r@{\\hspace{6pt}}X@{\\hspace{6pt}}c@{}}\n"
        latex += "\\toprule\n"
        latex += "\\textbf{Rank} & \\textbf{Feature} & \\textbf{Imp.} \\\\\n"
        latex += "\\midrule\n"

        for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
            feature = row['feature']
            importance = row['importance']

            # Clean up feature name for display
            display_name = self._clean_feature_name(feature)

            latex += f"{rank} & {display_name} & {importance:.3f} \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabularx}"

        print("✓ Table 3 generated")
        return latex

    def _clean_model_name(self, model_name: str) -> str:
        """Clean model name for display in LaTeX table"""
        # Map to proper capitalization
        name_mapping = {
            'decision tree': 'Decision Tree',
            'random forest': 'Random Forest',
            'gradient boosting': 'Gradient Boosting',
            'xgboost': 'XGBoost',
            'lightgbm': 'LightGBM',
        }

        # Try case-insensitive match
        model_lower = model_name.lower()
        if model_lower in name_mapping:
            return name_mapping[model_lower]

        # Fallback: return as-is
        return model_name

    def _clean_feature_name(self, feature: str) -> str:
        """Clean feature name for display in LaTeX table using centralized mapping"""
        # Use centralized feature name mapping
        display_name = get_display_name(feature)

        # Escape special LaTeX characters
        display_name = display_name.replace('%', '\\%')

        return display_name

    def generate_income_performance_table(self):
        """Generate Table 4: Model Performance by Income Level"""
        print("\nGenerating Table 4: Model Performance by Income Level...")

        if self.income_results is None:
            print("✗ No income segmentation results available")
            return None

        latex = "\\begin{tabular}{llccc}\n"
        latex += "\\toprule\n"
        latex += "Income Level & Model & N & RMSE & R² \\\\\n"
        latex += "\\midrule\n"

        # Group by segment
        segments = self.income_results['Segment'].unique()

        # Focus on top 3 models
        top_models = ['Random Forest', 'XGBoost', 'LightGBM']

        for i, segment in enumerate(segments):
            segment_data = self.income_results[self.income_results['Segment'] == segment]

            for j, model in enumerate(top_models):
                model_data = segment_data[segment_data['Model'] == model]

                if len(model_data) > 0:
                    row = model_data.iloc[0]
                    n = int(row['N_Observations'])
                    rmse = row['RMSE']
                    r2 = row['R2']

                    if j == 0:
                        # First model in segment - show segment name
                        latex += f"\\multirow{{3}}{{*}}{{{segment}}} & {model} & {n} & {rmse:.2f} & {r2:.3f} \\\\\n"
                    else:
                        # Subsequent models - no segment name
                        latex += f" & {model} & {n} & {rmse:.2f} & {r2:.3f} \\\\\n"

            # Add midrule between segments (except after last)
            if i < len(segments) - 1:
                latex += "\\midrule\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}"

        print("✓ Table 4 generated")
        return latex

    def generate_summary_text(self):
        """Generate summary text for the note in descriptive statistics section"""
        gini = self.processed_data[TARGET_VARIABLE]

        gdp_cols = [col for col in self.processed_data.columns if 'NY.GDP.PCAP' in col]
        gdp = self.processed_data[gdp_cols[0]] if gdp_cols else None

        summary = f"The mean GINI coefficient is {gini.mean():.1f} with standard deviation {gini.std():.1f}, "
        summary += f"ranging from {gini.min():.1f} to {gini.max():.1f}. "

        if gdp is not None:
            summary += f"GDP per capita shows substantial variation (mean: {gdp.mean():.0f}, "
            summary += f"std: {gdp.std():.0f}), reflecting the inclusion of both developed and developing economies."

        return summary

    def save_table_files(self):
        """Generate and save tables as .tex files"""
        print("\n" + "="*60)
        print("GENERATING LATEX TABLE FILES")
        print("="*60)

        # Generate Table 1
        table1 = self.generate_descriptive_table()
        table1_path = os.path.join(TABLES_DIR, 'table1_descriptive.tex')
        with open(table1_path, 'w') as f:
            f.write(table1)
        print(f"✓ Saved to: {table1_path}")

        # Generate Table 2
        table2 = self.generate_performance_table()
        table2_path = os.path.join(TABLES_DIR, 'table2_performance.tex')
        with open(table2_path, 'w') as f:
            f.write(table2)
        print(f"✓ Saved to: {table2_path}")

        # Generate Table 3 (Feature Importance) if models available
        if self.trained_models is not None:
            table3 = self.generate_feature_importance_table(model_name='xgboost', top_n=10)
            if table3:
                table3_path = os.path.join(TABLES_DIR, 'table3_features.tex')
                with open(table3_path, 'w') as f:
                    f.write(table3)
                print(f"✓ Saved to: {table3_path}")

        # Generate summary text
        summary = self.generate_summary_text()
        summary_path = os.path.join(TABLES_DIR, 'summary_text.tex')
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(f"✓ Saved to: {summary_path}")

        # Generate Table 4 if data available
        if self.income_results is not None:
            table4 = self.generate_income_performance_table()
            if table4:
                table4_path = os.path.join(TABLES_DIR, 'table4_income.tex')
                with open(table4_path, 'w') as f:
                    f.write(table4)
                print(f"✓ Saved to: {table4_path}")

    def run(self):
        """Execute the full table population process"""
        if not self.load_data():
            print("\n⚠ Some required data files are missing. Run the full pipeline first:")
            print("  python main.py")
            return False

        # Create tables directory if it doesn't exist
        os.makedirs(TABLES_DIR, exist_ok=True)

        # Generate and save table files
        self.save_table_files()

        print("\n" + "="*60)
        print("TABLE POPULATION COMPLETE")
        print("="*60)
        print(f"\nTables saved to {TABLES_DIR}/")
        print("The LaTeX paper imports these tables automatically.")
        print("\nYou can now compile the report with:")
        print("  cd report")
        print("  pdflatex paper.tex")
        print("  bibtex paper")
        print("  pdflatex paper.tex")
        print("  pdflatex paper.tex")

        return True


def main():
    """Main execution function"""
    populator = PaperTablePopulator()
    populator.run()


if __name__ == "__main__":
    main()
