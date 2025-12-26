# GINI Coefficient Prediction using World Bank Data

This project implements machine learning models to predict the GINI coefficient (income inequality measure) using socioeconomic indicators from the World Bank database.

## Project Overview

The project uses five tree-based regression models:
1. **Decision Tree Regressor** - Simple interpretable model
2. **Random Forest Regressor** - Ensemble of decision trees
3. **Gradient Boosting Regressor** - Sequential ensemble method
4. **XGBoost** - Extreme Gradient Boosting with advanced regularization
5. **LightGBM** - Fast gradient boosting framework optimized for efficiency

## Project Structure

```
.
├── code/                           # Python scripts
│   ├── 01_data_collection.py      # Fetch data from World Bank API
│   ├── 02_data_preprocessing.py   # Clean and prepare data
│   ├── 03_model_training.py       # Train and evaluate models
│   ├── 04_predict.py              # Make predictions (on-demand training)
│   ├── 05_comprehensive_comparison.py  # Detailed model comparison
│   ├── 06_segmentation_analysis.py     # Income/regional segmentation analysis
│   └── 07_statistical_tests.py         # Statistical significance tests
├── output/                         # Generated outputs
│   ├── *.csv                      # Data and metrics
│   ├── *.png                      # Visualizations
│   └── *.txt                      # Reports
├── paper/                          # LaTeX paper and PDF
│   ├── research_paper.tex         # Main research paper
│   └── gini_prediction_paper.*    # Additional paper files
├── main.py                         # Master pipeline script (run from root)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

**Note:** This project does NOT use `.pkl` files. Models are trained on-demand when needed for predictions.

## Installation

1. **Install Python 3.8 or higher**

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Run the Complete Pipeline

The easiest way to run everything:

```bash
python main.py
```

This will:
1. Collect data from World Bank API (with country identifiers)
2. Preprocess the data
3. Train all models
4. Generate comprehensive comparison
5. Run segmentation analysis (income-level and geographic regional)
6. Perform statistical significance tests
7. Populate LaTeX tables for research paper
8. Save all outputs to `output/` folder

**What runs automatically:**
- Data collection from World Bank API (including ISO3 country codes)
- Data preprocessing and feature engineering
- Training of all 5 models (Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM)
- Comprehensive model comparison and evaluation
- **Segmentation analysis** (income-level and 7 geographic regions using World Bank classification)
- **Statistical significance tests** (bootstrap, permutation, consistency)
- **LaTeX table population** (generates table content for research paper)

**Pipeline Options:**

```bash
# Quick run (default) - includes all analyses
python main.py

# Fast run (recent data only, 2015-2023)
python main.py --mode fast

# Optimized run (with hyperparameter tuning - slower)
python main.py --mode optimized

# Custom run with specific steps
python main.py --mode custom --skip-collection --skip-preprocessing

# Skip comprehensive comparison (also skips segmentation & statistical tests)
python main.py --skip-comparison

# See all options
python main.py --help
```

**Expected Runtime (complete pipeline):**
- Quick mode (default): ~15-20 minutes
- Fast mode (2015-2023 data): ~10-15 minutes
- Optimized mode (with tuning): ~1-2 hours

The additional time includes:
- Segmentation analysis: ~2-3 minutes
- Statistical significance tests: ~3-4 minutes

### Option 2: Run Individual Steps

#### Step 1: Data Collection

Collect data from the World Bank API:

```bash
cd code
python 01_data_collection.py
```

This will:
- Fetch ~50 socioeconomic indicators from 2000-2023
- Save raw data to `output/world_bank_data.csv`
- Display data summary and missing value statistics

**Time:** ~5-10 minutes (due to API rate limiting)

#### Step 2: Data Preprocessing

Clean and prepare the data:

```bash
cd code
python 02_data_preprocessing.py
```

This will:
- Filter for countries/years with GINI data
- Handle missing values using median imputation
- Create engineered features (urbanization rate, economic diversity, etc.)
- Save processed data to `output/processed_data.csv`
- Save feature names to `output/feature_names.csv`

**Options you can modify in the script:**
- `strategy`: Imputation method ('median', 'mean', 'knn', 'drop')
- `threshold`: Drop features with >X% missing data (default: 0.6)
- `remove_outliers()`: Uncomment to remove statistical outliers

#### Step 3: Model Training

Train all five models:

```bash
cd code
python 03_model_training.py
```

This will:
- Split data into train (80%) and test (20%) sets
- Train Decision Tree, Random Forest, Gradient Boosting, XGBoost, and LightGBM models
- Perform cross-validation
- Generate evaluation metrics and visualizations
- Models are NOT saved (trained on-demand when needed)

**Outputs in `output/` folder:**
- `model_comparison.csv` - Performance comparison table
- `feature_importance.png` - Top features for each model
- `predictions_plot.png` - Actual vs predicted scatter plots
- `residuals_plot.png` - Residual analysis

**Hyperparameter Tuning:**
- Set `tune_hyperparameters=True` for better results (much slower)
- Set `tune_hyperparameters=False` for faster training with decent defaults

**Time:**
- Without tuning: ~1-2 minutes
- With tuning: ~30-60 minutes

#### Step 4: Comprehensive Model Comparison (Optional)

Perform in-depth analysis and comparison of all models:

```bash
cd code
python 05_comprehensive_comparison.py
```

This will:
- Train all models on-demand
- Calculate 15+ performance metrics for each model
- Perform statistical tests (paired t-test, Wilcoxon test)
- Analyze performance across different GINI segments
- Generate detailed visualizations
- Create a comprehensive comparison report

**Outputs in `output/` folder:**
- `comprehensive_metrics.csv` - Detailed metrics table
- `statistical_comparison.csv` - Statistical test results
- `segment_performance.csv` - Performance by GINI range
- `comprehensive_comparison.png` - Multi-panel comparison visualization
- `error_analysis.png` - Detailed error analysis plots
- `segment_performance.png` - Performance across segments
- `model_comparison_report.txt` - Comprehensive text report

**Metrics Included:**
- RMSE, MAE, MAPE, R², Explained Variance
- Max Error, Median Absolute Error
- Residual statistics (mean, std, skewness, kurtosis)
- Cross-validation scores with confidence intervals
- Overfitting gap analysis

**Time:** ~1-2 minutes

#### Step 5: Segmentation Analysis

**Note:** This step runs automatically when using `python main.py`. You can also run it standalone:

```bash
cd code
python 06_segmentation_analysis.py
```

This will:
- Segment countries by income level (GDP per capita quartiles)
- Create regional segments using World Bank geographic classifications:
  - East Asia & Pacific
  - Europe & Central Asia
  - Latin America & Caribbean
  - Middle East & North Africa
  - North America
  - South Asia
  - Sub-Saharan Africa
- Train models separately for each segment
- Compare performance across segments
- Identify context-specific feature importance patterns
- Generate heatmaps showing how features matter differently across contexts

**Outputs in `output/` folder:**
- `segmentation_income_performance.png` - Performance comparison by income level
- `segmentation_income_features.png` - Feature importance heatmap by income level
- `segmentation_income_results.csv` - Detailed results by income level
- `segmentation_regional_performance.png` - Performance comparison by region
- `segmentation_regional_features.png` - Feature importance heatmap by region
- `segmentation_regional_results.csv` - Detailed results by region
- `segmentation_summary_report.txt` - Comprehensive text report

**Key Insights:**
- Identifies whether inequality drivers differ across development stages (income segmentation)
- Reveals geographic patterns in inequality mechanisms (regional segmentation)
- Tests Kuznets curve hypothesis empirically
- Reveals context-specific policy priorities for different regions and development levels

**Time:** ~2-3 minutes

#### Step 6: Statistical Significance Tests

**Note:** This step runs automatically when using `python main.py`. You can also run it standalone:

```bash
cd code
python 07_statistical_tests.py
```

This will:
- Calculate bootstrap 95% confidence intervals for feature importance (100 iterations)
- Perform permutation importance tests with statistical significance (p-values)
- Compute cross-model consistency using Spearman rank correlations
- Identify which features are robustly important vs. potentially spurious

**Outputs in `output/` folder:**
- `statistical_tests_bootstrap.csv` - Bootstrap confidence intervals
- `statistical_tests_bootstrap.png` - Visualization with error bars
- `statistical_tests_permutation.csv` - Permutation test results with p-values
- `statistical_tests_consistency.csv` - Cross-model correlation matrix
- `statistical_tests_consistency.png` - Heatmap of model agreements
- `statistical_tests_summary.txt` - Comprehensive statistical report

**Metrics Included:**
- 95% Bootstrap confidence intervals for all features
- Permutation importance with one-sided t-tests
- Spearman rank correlations between all model pairs
- Statistical significance codes (***p<0.001, **p<0.01, *p<0.05)

**Key Insights:**
- Confirms which features are statistically significantly important
- Validates that top features are not due to chance
- Shows consistency (or divergence) across modeling approaches

**Time:** ~3-4 minutes

#### Step 7: Populating LaTeX Tables

**Note:** This step runs automatically when using `python main.py`. You can also run it standalone:

```bash
cd code
python 08_populate_paper_tables.py
```

This will:
- Load processed data and model results
- Generate LaTeX table files (.tex) with actual values
- **Automatically update research_paper.tex** to include the tables using `\input{}` commands
- Replace placeholder "XX" values with real statistics

**Outputs in `paper/tables/` folder:**
- `summary_text.tex` - Summary text for descriptive statistics section
- `table1_descriptive.tex` - LaTeX table for descriptive statistics
- `table2_performance.tex` - LaTeX table for model performance comparison
- `table4_income.tex` - LaTeX table for income-level segmentation results

**How it works:**
The script automatically modifies [paper/research_paper.tex](paper/research_paper.tex) to use `\input{tables/table1_descriptive.tex}` instead of hardcoded placeholder tables. When you compile the PDF, LaTeX will automatically include the generated tables with real data.

**Time:** <1 minute

#### Step 8: Making Predictions

Use trained models to predict GINI for new data:

```bash
cd code
python 04_predict.py
```

**Example usage in Python:**

```python
import sys
sys.path.append('code')
from code.predict_04 import GINIPredictor

# Initialize predictor (models will be trained automatically)
predictor = GINIPredictor()

# Predict from CSV file
predictor.predict_from_csv(
    input_file='new_data.csv',
    output_file='predictions.csv',
    model_name='random_forest'
)

# Or predict single observation
features = {
    'NY.GDP.PCAP.PP.CD': 15000,
    'SL.UEM.TOTL.ZS': 8.5,
    # ... all other features
}
gini_prediction = predictor.predict_single(features, 'random_forest')

# Get ensemble prediction from all models
all_predictions = predictor.predict_ensemble(features)
```

## Features Used

The models use approximately 50+ features including:

### Economic Indicators
- GDP, GDP growth, GDP per capita
- Sector composition (agriculture, industry, services)
- Trade openness, foreign investment
- Exchange rates, interest rates

### Demographics
- Population (total, urban, rural)
- Urban growth rate
- Age dependency ratio
- Fertility rate

### Human Development
- Health expenditure (total, per capita, public/private)
- Education expenditure
- School enrollment (primary, secondary)

### Labor Market
- Labor force participation (total, male, female)
- Unemployment (total, by gender, youth)
- Employment by sector

### Infrastructure & Environment
- Internet access
- Electricity access
- Renewable energy
- Air pollution
- Agricultural land, forest area

### Governance
- Women in parliament
- Gender parity in education

### Engineered Features
- Urbanization rate
- Trade openness
- Economic diversity index
- Health to education spending ratio
- Gender labor gap

## Model Performance

Expected performance metrics (will vary based on data):

| Model | Test RMSE | Test R² | Cross-Val RMSE |
|-------|-----------|---------|----------------|
| Decision Tree | ~5-7 | ~0.60-0.70 | ~6-8 |
| Random Forest | ~4-5 | ~0.75-0.85 | ~5-6 |
| Gradient Boosting | ~4-5 | ~0.75-0.85 | ~5-6 |
| XGBoost | ~3.5-4.5 | ~0.80-0.90 | ~4-5 |
| LightGBM | ~3.5-4.5 | ~0.80-0.90 | ~4-5 |

**Note:** XGBoost and LightGBM typically perform best, followed by Random Forest and Gradient Boosting

## Interpreting Results

### Feature Importance
- Check `output/feature_importance.png` to see which factors most influence GINI
- Common important features: GDP per capita, education, health expenditure

### Model Comparison
- **R² score**: Proportion of variance explained (higher is better)
- **RMSE**: Average prediction error in GINI points (lower is better)
- **Overfit gap**: Difference between train and test RMSE (should be small)

### Predictions
- GINI ranges from 0 (perfect equality) to 100 (perfect inequality)
- Most countries have GINI between 25-50
- Values >50 indicate high inequality

## Customization

### Modify data collection period:
Edit `code/01_data_collection.py`:
```python
collector.collect_all_data(start_year=2010, end_year=2023)
```

### Change train/test split:
Edit `code/03_model_training.py`:
```python
predictor.load_and_split_data(test_size=0.3)  # 30% test set
```

### Add/remove features:
Edit the `indicators` list in `code/01_data_collection.py`

### Try different imputation strategies:
Edit `code/02_data_preprocessing.py`:
```python
preprocessor.handle_missing_values(strategy='knn')  # or 'mean', 'drop'
```

### Enable outlier removal:
Uncomment this line in `code/02_data_preprocessing.py`:
```python
preprocessor.remove_outliers(method='iqr', threshold=3.0)
```

## Output Files

All generated files are saved in the `output/` folder:

### Data Files
- `world_bank_data.csv` - Raw data from World Bank API
- `processed_data.csv` - Cleaned and preprocessed data
- `feature_names.csv` - List of feature names used in models

### Metrics and Reports
- `model_comparison.csv` - Basic performance metrics
- `comprehensive_metrics.csv` - Detailed performance metrics
- `statistical_comparison.csv` - Statistical test results
- `segment_performance.csv` - Performance by GINI range
- `model_comparison_report.txt` - Comprehensive text report
- `segmentation_income_results.csv` - Income-level segmentation results
- `segmentation_regional_results.csv` - Regional segmentation results
- `segmentation_summary_report.txt` - Segmentation analysis summary
- `statistical_tests_bootstrap.csv` - Bootstrap confidence intervals
- `statistical_tests_permutation.csv` - Permutation test p-values
- `statistical_tests_consistency.csv` - Cross-model correlations
- `statistical_tests_summary.txt` - Statistical tests summary

### LaTeX Tables (in `paper/tables/` folder)
- `summary_text.tex` - Summary text for paper descriptive statistics
- `table1_descriptive.tex` - LaTeX table for descriptive statistics
- `table2_performance.tex` - LaTeX table for model performance
- `table4_income.tex` - LaTeX table for income segmentation

### Visualizations
- `feature_importance.png` - Feature importance charts
- `predictions_plot.png` - Actual vs predicted plots
- `residuals_plot.png` - Residual analysis
- `comprehensive_comparison.png` - Multi-panel comparison
- `error_analysis.png` - Detailed error analysis
- `segment_performance.png` - Performance across segments
- `segmentation_income_performance.png` - Performance by income level
- `segmentation_income_features.png` - Feature importance by income level
- `segmentation_regional_performance.png` - Performance by region
- `segmentation_regional_features.png` - Feature importance by region
- `statistical_tests_bootstrap.png` - Bootstrap confidence intervals plot
- `statistical_tests_consistency.png` - Cross-model consistency heatmap

## Troubleshooting

### API Rate Limiting
If you get rate limit errors, increase the sleep time in `code/01_data_collection.py`:
```python
time.sleep(1.0)  # Instead of 0.5
```

### Missing Data Issues
If too many features are dropped, lower the threshold in `code/02_data_preprocessing.py`:
```python
preprocessor.handle_missing_values(threshold=0.8)  # Allow 80% missing
```

### Memory Issues
Reduce the date range or number of features if running out of memory

### Poor Model Performance
- Enable hyperparameter tuning (slower but better)
- Remove outliers
- Try different imputation strategies
- Use feature selection to remove noisy features

## Important Notes

### No .pkl Files
This project uses **on-demand model training** instead of saving models as `.pkl` files:
- Models are trained fresh when needed for predictions
- No security concerns with pickle files
- Simpler project structure
- Always uses latest training data
- Trade-off: Takes a few seconds to train when making predictions

### Research Paper
The `paper/` folder contains a comprehensive LaTeX research paper:

**`research_paper.tex`** - Academic paper exploring the research question:
*"What socioeconomic factors are the strongest predictors of income inequality across countries, and how does their relative importance vary across different machine learning models?"*

The paper includes:
- Full literature review on inequality and ML methods
- Detailed methodology for all 5 models
- Complete results including 26 key findings
- Statistical significance tests (bootstrap, permutation, consistency)
- Income-level and regional segmentation analysis
- Policy implications and future research directions
- 19 academic references

**To compile the PDF:**
```bash
cd paper
pdflatex research_paper.tex
pdflatex research_paper.tex  # Run twice for references
```

The paper integrates findings from all analysis scripts and provides academic-level documentation suitable for publication or presentation.

## Data Sources

All data comes from the World Bank Open Data API:
- Website: https://data.worldbank.org/
- API Documentation: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392

## Limitations

1. **Missing Data**: Many indicators have significant missing values, especially for developing countries
2. **Temporal Coverage**: GINI data is not available for all countries/years
3. **Causality**: Models show correlations, not causal relationships
4. **Data Quality**: Varies by country and indicator
5. **Lag Effects**: Current model doesn't account for time lags in economic effects

## Future Improvements

Potential enhancements:
- Implement time-series features (lagged values, trends)
- Use more sophisticated imputation (MICE, deep learning)
- Add country-level fixed effects
- Build a web interface for predictions
- Add confidence intervals for predictions
- Implement model caching for faster repeated predictions

## License

This project uses data from the World Bank under their terms of use.

## Contact

For questions or issues, please refer to the World Bank API documentation or scikit-learn documentation.

---

**Note:** This is a research/educational project. Predictions should not be used for policy decisions without thorough validation and expert review.
