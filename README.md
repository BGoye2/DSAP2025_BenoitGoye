# GINI Coefficient Prediction using World Bank Data

Machine learning pipeline to predict the GINI coefficient (income inequality measure) using socioeconomic indicators from the World Bank database.

## Project Overview

This project uses five tree-based regression models to predict income inequality:
1. **Decision Tree Regressor** - Simple interpretable baseline
2. **Random Forest Regressor** - Ensemble of decision trees
3. **Gradient Boosting Regressor** - Sequential ensemble method
4. **XGBoost** - Extreme Gradient Boosting with advanced regularization
5. **LightGBM** - Fast gradient boosting framework optimized for efficiency

## Project Structure

```
.
├── src/                            # Source code
│   ├── 01_data_collection.py      # Fetch data from World Bank API
│   ├── 02_data_preprocessing.py   # Clean and prepare data
│   ├── 03_model_training.py       # Train ML models
│   ├── 04_model_evaluation.py     # Evaluate models (metrics & visualizations)
│   ├── 05_comprehensive_comparison.py  # Detailed model comparison
│   ├── 06_segmentation_analysis.py     # Income/regional segmentation
│   ├── 07_statistical_tests.py         # Statistical significance tests
│   ├── 08_populate_paper_tables.py     # Generate LaTeX tables
│   ├── utils.py                    # Utility functions (Spinner, print helpers)
│   └── config/                     # Configuration files
│       ├── constants.py            # Project constants and hyperparameters
│       ├── indicators.py           # World Bank indicator definitions
│       ├── feature_names.py        # Display name mappings
│       ├── feature_engineering.py  # Engineered feature definitions
│       ├── feature_categories.py   # Feature categorization
│       └── country_regions.json    # World Bank regional classifications
├── output/                         # Generated outputs
│   ├── *.csv                      # Data and metrics
│   ├── trained_models.pkl         # Saved models with metadata
│   ├── figures/                   # Generated visualizations (PNG files)
│   ├── tables/                    # Generated LaTeX table files
│   └── .cache/                    # Bootstrap cache for faster reruns
├── documentation/                  # Project documentation
│   ├── paper/                     # LaTeX research report
│   │   ├── ProjectReport.tex      # Main research report
│   │   └── references.bib         # Bibliography
│   └── presentation/              # Project presentations
├── main.py                         # Master pipeline script
├── requirements.txt                # Python dependencies (pip)
├── environment.yml                 # Conda environment specification
└── README.md                       # This file
```

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate dsap-project
```

### Option 2: Using pip

```bash
# Install Python 3.8+ (3.12+ recommended)
pip install -r requirements.txt

# For macOS users: Install OpenMP for XGBoost
brew install libomp
```

## Quick Start

Run the complete pipeline with one command:

```bash
python main.py
```

This executes all steps:
1. Data collection from World Bank API
2. Data preprocessing and feature engineering
3. Model training (5 models)
4. Model evaluation and visualizations
5. Comprehensive model comparison
6. Segmentation analysis (income-level and regional)
7. Statistical significance tests
8. LaTeX table generation for research report

**Pipeline Options:**

```bash
python main.py                    # Quick run (default)
python main.py --mode fast        # Recent data only 2015-2023
python main.py --mode optimized   # With hyperparameter tuning (slower)
python main.py --help             # See all options
```

**Performance Features:**
- Parallel API calls (6x faster data collection)
- Parallel bootstrap iterations (6x faster statistical tests)
- Vectorized outlier removal (25x faster preprocessing)
- Bootstrap caching (60-100x faster on reruns)
- Model versioning (train once, reuse everywhere)

## Individual Pipeline Steps

You can run each step separately if needed:

```bash
cd src
python 01_data_collection.py      # Fetch data from World Bank API
python 02_data_preprocessing.py   # Clean and prepare data
python 03_model_training.py       # Train all 5 models
python 04_model_evaluation.py     # Evaluate models with visualizations
python 05_comprehensive_comparison.py  # Detailed model comparison
python 06_segmentation_analysis.py     # Income/regional analysis
python 07_statistical_tests.py         # Statistical significance tests
python 08_populate_paper_tables.py     # Generate LaTeX tables
```

**Key Scripts:**
- **01_data_collection.py**: Fetches 53 indicators from 2000-2023
- **03_model_training.py**: Trains all models, saves to `output/trained_models.pkl`
- **06_segmentation_analysis.py**: Analyzes 7 regions and income quartiles
- **07_statistical_tests.py**: Bootstrap + permutation tests with caching (significantly faster on reruns)

## Features

The models use socioeconomic indicators from the World Bank organized into categories:

- **Economic**: GDP, sector composition, trade, investment
- **Demographics**: Population, urbanization, dependency ratios
- **Human Development**: Health/education expenditure, school enrollment
- **Labor Market**: Employment, unemployment by gender and sector
- **Infrastructure**: Internet, electricity, renewable energy
- **Governance**: Gender parity, political representation
- **Engineered**: Urbanization rate, economic diversity, labor gaps

## Expected Performance

| Model | Test RMSE | Test R² |
|-------|-----------|---------|
| Decision Tree | ~5-7 | ~0.60-0.70 |
| Random Forest | ~4-5 | ~0.75-0.85 |
| Gradient Boosting | ~4-5 | ~0.75-0.85 |
| XGBoost | ~3.5-4.5 | ~0.80-0.90 |
| LightGBM | ~3.5-4.5 | ~0.80-0.90 |

**Note:** XGBoost and LightGBM typically perform best. Results vary based on data quality and completeness.

## Output Files

All outputs are saved in the `output/` folder:

**Data Files:**
- `world_bank_data.csv` - Raw API data
- `processed_data.csv` - Cleaned data
- `trained_models.pkl` - Saved models with metadata

**Visualizations** (in `output/figures/`):
- `feature_importance.png` - Feature importance charts
- `predictions_plot.png` - Actual vs predicted
- `residuals_plot.png` - Residual analysis
- `error_analysis.png` - Error distribution analysis
- `comprehensive_comparison.png` - Multi-panel comparison
- `segment_performance.png` - Overall segment performance
- `segmentation_income_performance.png` - Performance by income level
- `segmentation_income_features.png` - Income-level feature importance
- `segmentation_regional_performance.png` - Performance by region
- `segmentation_regional_features.png` - Regional feature importance
- `statistical_tests_bootstrap.png` - Bootstrap confidence intervals
- `statistical_tests_consistency.png` - Model consistency analysis

**Reports and Metrics:**
- `model_comparison.csv` - Performance metrics
- `comprehensive_metrics.csv` - Detailed metrics
- `segment_performance.csv` - Segment performance summary
- `segmentation_income_results.csv` - Income-level analysis
- `segmentation_regional_results.csv` - Regional analysis
- `statistical_tests_bootstrap.csv` - Bootstrap test results
- `statistical_tests_permutation.csv` - Permutation test results
- `statistical_tests_consistency.csv` - Consistency test results
- `statistical_comparison.csv` - Statistical comparison summary
- `*.txt` - Text reports

**LaTeX Tables** (in `output/tables/`):
- `table1_descriptive.tex` - Descriptive statistics
- `table2_performance.tex` - Model performance
- `table3_features.tex` - Feature importance
- `table4_income.tex` - Income segmentation
- `summary_text.tex` - Summary text snippets

## Key Features

**Reproducibility:**
- All random operations use `random_state=42` for consistent results
- Models saved with version metadata and data hash validation
- Results are fully reproducible across runs

**Model Persistence:**
- Models saved to `output/trained_models.pkl` with metadata
- SHA256 hash validation prevents using models on incompatible data
- Train once, reuse across all analysis scripts

**Caching:**
- Bootstrap results cached in `output/.cache/`
- Automatic cache invalidation on data changes
- 60-100x speedup on repeated runs

**Research Report:**
- Comprehensive LaTeX report in `documentation/paper/ProjectReport.tex`
- Includes literature review, methodology, and key findings
- Compile with: `cd documentation/paper && pdflatex ProjectReport.tex && bibtex ProjectReport && pdflatex ProjectReport.tex && pdflatex ProjectReport.tex`

## Data Source

All data from [World Bank Open Data API](https://data.worldbank.org/)
- 53 socioeconomic indicators from 2000-2023 (including GINI target variable)
- API Docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392

## Limitations

- **Missing Data**: Many indicators incomplete, especially for developing countries
- **Temporal Coverage**: GINI data not available for all countries/years
- **Causality**: Models show correlations, not causal relationships
- **Lag Effects**: Current model doesn't account for time lags

## Configuration System

The project uses a modular configuration system with all settings centralized in `src/config/`.

**Key configuration modules:**

### `constants.py` - Core Constants
Central location for all project constants including the target variable, paths, and model parameters.

```python
from config.constants import TARGET_VARIABLE, DEFAULT_RANDOM_SEED

# Use target variable
y = data[TARGET_VARIABLE].values  # 'SI.POV.GINI'
```

### `feature_engineering.py` - Engineered Features
Declarative configuration for all engineered features with automatic dependency checking.

```python
from config.feature_engineering import create_all_engineered_features

# Apply all engineered features (urbanization_rate, log_gdp_per_capita, etc.)
data = create_all_engineered_features(data)
```

**Supported operations**: ratio_scaled, log1p, sum, ratio, difference, diversity

### `feature_categories.py` - Feature Classification
Categorizes features by domain (gdp, education, health, labor, demographics, etc.) for robust feature selection.

```python
from config.feature_categories import filter_features_by_category

# Get all GDP-related features
gdp_features = filter_features_by_category(features, 'gdp')
```

**11 categories**: gdp, education, health, labor, demographics, trade, government, infrastructure, economic_structure, income_inequality, other

### `feature_names.py` - Display Names
Maps World Bank indicator codes to human-readable names for tables and plots.

```python
from config.constants import get_display_name

# In visualizations
plt.xlabel(get_display_name('NY.GDP.PCAP.CD'))  # "GDP per capita (current $)"
```

### `indicators.py` - World Bank Indicators
Defines all 53 World Bank indicator codes to fetch from the API.

```python
from config.indicators import WORLD_BANK_INDICATORS

# All indicators are defined in one place
for indicator in WORLD_BANK_INDICATORS:
    fetch_data(indicator)
```

**Adding new features to the project:**
1. Add the World Bank indicator code to `indicators.py`
2. Add a human-readable display name to `feature_names.py`
3. Assign the feature to the appropriate category in `feature_categories.py`
4. (Optional) Define any engineered features based on it in `feature_engineering.py`

## Project Notes

This is a research/educational project implementing ML best practices:
- Modular configuration system (all settings in `src/config/`)
- Comprehensive type hints and documentation
- DRY principle applied throughout
- Parallel processing for performance
- Model versioning and caching

**Note:** Predictions should not be used for policy decisions without thorough validation and expert review.
