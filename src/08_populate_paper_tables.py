"""
LaTeX Table Population for Research Paper

This script automatically generates LaTeX tables from analysis results and inserts them
into the research paper, eliminating manual copy-paste errors and ensuring reproducibility.

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

3. Table 4 - Model Performance by Income Level:
   - Segmented performance across income quartiles
   - Shows RMSE and R² for each income level
   - Reveals whether models work better for certain economic contexts
   - Uses multirow formatting for readability

4. Summary Text:
   - Dynamically generated summary statistics text
   - Inserted into descriptive statistics section
   - Updates automatically when data changes

Process:
1. Load results from CSV files (model_comparison, segmentation_income, etc.)
2. Generate LaTeX table code with proper formatting
3. Save to report/tables/*.tex files
4. Update research_paper.tex to \\input{} these tables
5. Ensures paper always reflects latest results

Input: output/processed_data.csv, output/model_comparison.csv,
       output/segmentation_income_results.csv
Output: report/tables/table1_descriptive.tex, report/tables/table2_performance.tex,
        report/tables/table4_income.tex, report/tables/summary_text.tex,
        updated report/research_paper.tex
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')


class PaperTablePopulator:
    """Generates LaTeX table content from actual model results and updates paper"""

    def __init__(self):
        """Initialize table populator"""
        self.processed_data = None
        self.model_comparison = None
        self.income_results = None
        self.feature_names = None

    def load_data(self):
        """Load all necessary data files"""
        print("Loading data files...")

        try:
            self.processed_data = pd.read_csv('output/processed_data.csv')
            print(f"✓ Loaded processed_data.csv: {len(self.processed_data)} rows")
        except FileNotFoundError:
            print("✗ processed_data.csv not found - run preprocessing first")
            return False

        try:
            self.model_comparison = pd.read_csv('output/model_comparison.csv')
            print(f"✓ Loaded model_comparison.csv: {len(self.model_comparison)} rows")
        except FileNotFoundError:
            print("✗ model_comparison.csv not found - run model training first")
            return False

        try:
            self.income_results = pd.read_csv('output/segmentation_income_results.csv')
            print(f"✓ Loaded segmentation_income_results.csv: {len(self.income_results)} rows")
        except FileNotFoundError:
            print("⚠ segmentation_income_results.csv not found - Table 4 will be skipped")
            self.income_results = None

        try:
            self.feature_names = pd.read_csv('output/feature_names.csv')
            print(f"✓ Loaded feature_names.csv: {len(self.feature_names)} features")
        except FileNotFoundError:
            print("✗ feature_names.csv not found")
            return False

        return True

    def generate_descriptive_table(self) -> str:
        """Generate Table 1: Descriptive Statistics"""
        print("\nGenerating Table 1: Descriptive Statistics...")

        # Get key variables
        gini = self.processed_data['SI.POV.GINI']

        # Find GDP per capita column
        gdp_cols = [col for col in self.processed_data.columns if 'NY.GDP.PCAP' in col]
        gdp = self.processed_data[gdp_cols[0]] if gdp_cols else None

        # Find education expenditure column
        edu_cols = [col for col in self.processed_data.columns if 'SE.XPD' in col]
        edu = self.processed_data[edu_cols[0]] if edu_cols else None

        # Find health expenditure column
        health_cols = [col for col in self.processed_data.columns if 'SH.XPD' in col and 'PC' in col]
        health = self.processed_data[health_cols[0]] if health_cols else None

        # Find unemployment column
        unemp_cols = [col for col in self.processed_data.columns if 'SL.UEM.TOTL' in col]
        unemp = self.processed_data[unemp_cols[0]] if unemp_cols else None

        # Find urban population column
        urban_cols = [col for col in self.processed_data.columns if 'SP.URB.TOTL' in col or 'urbanization' in col.lower()]
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

    def generate_performance_table(self) -> str:
        """Generate Table 2: Model Performance Comparison"""
        print("\nGenerating Table 2: Model Performance Comparison...")

        latex = "\\begin{tabular}{lcccc}\n"
        latex += "\\toprule\n"
        latex += "Model & Train RMSE & Test RMSE & Test R² & CV RMSE \\\\\n"
        latex += "\\midrule\n"

        for _, row in self.model_comparison.iterrows():
            model_name = row['Model']
            train_rmse = row['Train RMSE']
            test_rmse = row['Test RMSE']
            test_r2 = row['Test R²']
            cv_rmse = row['CV RMSE']

            latex += f"{model_name} & {train_rmse:.2f} & {test_rmse:.2f} & {test_r2:.3f} & {cv_rmse:.2f} \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}"

        print("✓ Table 2 generated")
        return latex

    def generate_income_performance_table(self) -> str:
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

    def generate_summary_text(self) -> str:
        """Generate summary text for the note in descriptive statistics section"""
        gini = self.processed_data['SI.POV.GINI']

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
        with open('report/tables/table1_descriptive.tex', 'w') as f:
            f.write(table1)
        print("✓ Saved to: report/tables/table1_descriptive.tex")

        # Generate Table 2
        table2 = self.generate_performance_table()
        with open('report/tables/table2_performance.tex', 'w') as f:
            f.write(table2)
        print("✓ Saved to: report/tables/table2_performance.tex")

        # Generate summary text
        summary = self.generate_summary_text()
        with open('report/tables/summary_text.tex', 'w') as f:
            f.write(summary)
        print("✓ Saved to: report/tables/summary_text.tex")

        # Generate Table 4 if data available
        if self.income_results is not None:
            table4 = self.generate_income_performance_table()
            if table4:
                with open('report/tables/table4_income.tex', 'w') as f:
                    f.write(table4)
                print("✓ Saved to: report/tables/table4_income.tex")

    def update_latex_paper(self):
        """Update research_paper.tex to include the generated tables"""
        print("\n" + "="*60)
        print("UPDATING RESEARCH PAPER")
        print("="*60)

        paper_path = 'report/research_paper.tex'

        # Read the current paper
        with open(paper_path, 'r') as f:
            content = f.read()

        # Update summary text (line ~236)
        pattern1 = r'(Table \\ref\{tab:descriptive\} presents summary statistics.*?)\. The mean GINI coefficient is approximately XX with standard deviation YY, ranging from ZZ to WW\. GDP per capita shows substantial variation, reflecting the inclusion of both developed and developing economies\.'
        replacement1 = r'\1. \\input{tables/summary_text.tex}'
        content = re.sub(pattern1, replacement1, content, flags=re.DOTALL)

        # Update Table 1 (Descriptive Statistics) - Replace the entire tabular environment
        pattern_table1 = r'(\\begin\{table\}\[H\]\s*\\centering\s*\\caption\{Descriptive Statistics\}\s*\\label\{tab:descriptive\}\s*)\\begin\{tabular\}\{lcccccc\}.*?\\end\{tabular\}'
        replacement_table1 = r'\1\\input{tables/table1_descriptive.tex}'
        content = re.sub(pattern_table1, replacement_table1, content, flags=re.DOTALL)

        # Update Table 2 (Model Performance) - Replace the entire tabular environment
        pattern_table2 = r'(\\begin\{table\}\[H\]\s*\\centering\s*\\caption\{Model Performance Comparison\}\s*\\label\{tab:performance\}\s*)\\begin\{tabular\}\{lcccc\}.*?\\end\{tabular\}'
        replacement_table2 = r'\1\\input{tables/table2_performance.tex}'
        content = re.sub(pattern_table2, replacement_table2, content, flags=re.DOTALL)

        # Update Table 4 (Income Performance) - Replace the entire tabular environment
        if self.income_results is not None:
            pattern_table4 = r'(\\begin\{table\}\[H\]\s*\\centering\s*\\caption\{Model Performance by Income Level Segment\}\s*\\label\{tab:income_performance\}\s*)\\begin\{tabular\}\{llccc\}.*?\\end\{tabular\}'
            replacement_table4 = r'\1\\input{tables/table4_income.tex}'
            content = re.sub(pattern_table4, replacement_table4, content, flags=re.DOTALL)

        # Write the updated paper
        with open(paper_path, 'w') as f:
            f.write(content)

        print("✓ Updated research_paper.tex with \\input{} commands")
        print("\nThe paper now automatically includes:")
        print("  - tables/summary_text.tex (descriptive statistics summary)")
        print("  - tables/table1_descriptive.tex (Table 1)")
        print("  - tables/table2_performance.tex (Table 2)")
        if self.income_results is not None:
            print("  - tables/table4_income.tex (Table 4)")

    def run(self):
        """Execute the full table population process"""
        if not self.load_data():
            print("\n⚠ Some required data files are missing. Run the full pipeline first:")
            print("  python main.py")
            return False

        # Create tables directory if it doesn't exist
        import os
        os.makedirs('report/tables', exist_ok=True)

        # Generate and save table files
        self.save_table_files()

        # Update the LaTeX paper
        self.update_latex_paper()

        print("\n" + "="*60)
        print("TABLE POPULATION COMPLETE")
        print("="*60)
        print("\nYou can now compile the report with:")
        print("  cd report")
        print("  pdflatex research_paper.tex")
        print("  pdflatex research_paper.tex")

        return True


def main():
    """Main execution function"""
    populator = PaperTablePopulator()
    populator.run()


if __name__ == "__main__":
    main()
