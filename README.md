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
│   ├── 04_predict.py              # Make predictions with trained models
│   ├── 05_model_evaluation.py     # Evaluate models (metrics & visualizations)
│   ├── 06_comprehensive_comparison.py  # Detailed model comparison
│   ├── 07_segmentation_analysis.py     # Income/regional segmentation
│   ├── 08_statistical_tests.py         # Statistical significance tests
│   ├── 09_populate_paper_tables.py     # Generate LaTeX tables
│   ├── utils.py                    # Utility functions (Spinner, print helpers)
│   └── config/                     # Configuration files
│       ├── constants.py            # Project constants and hyperparameters
│       ├── country_regions.json   # World Bank regional classifications
│       └── indicators.py           # World Bank indicator definitions
├── output/                         # Generated outputs
│   ├── *.csv                      # Data and metrics
│   ├── *.png                      # Visualizations
│   ├── trained_models.pkl         # Saved models with metadata
│   └── .cache/                    # Bootstrap cache for faster reruns
├── report/                         # LaTeX research report
│   ├── research_paper.tex         # Main research report
│   └── tables/                    # Generated LaTeX table files
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

# Verify installation
python -c "import sklearn, xgboost, lightgbm; print('✓ All dependencies installed')"
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
python main.py                    # Quick run (default, ~5-7 minutes)
python main.py --mode fast        # Recent data only 2015-2023 (~3-5 minutes)
python main.py --mode optimized   # With hyperparameter tuning (~30-60 minutes)
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
python 04_predict.py              # Make predictions (optional)
python 05_model_evaluation.py     # Evaluate models with visualizations
python 06_comprehensive_comparison.py  # Detailed model comparison
python 07_segmentation_analysis.py     # Income/regional analysis
python 08_statistical_tests.py         # Statistical significance tests
python 09_populate_paper_tables.py     # Generate LaTeX tables
```

**Key Scripts:**
- **01_data_collection.py**: Fetches ~50 indicators from 2000-2023 (~1-2 min)
- **03_model_training.py**: Trains all models, saves to `output/trained_models.pkl` (~1-2 min)
- **07_segmentation_analysis.py**: Analyzes 7 regions and income quartiles (~40 sec)
- **08_statistical_tests.py**: Bootstrap + permutation tests with caching (~30-40 sec first run, ~5 sec cached)

## Features

The models use 50+ socioeconomic indicators organized into categories:

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

**Visualizations:**
- `feature_importance.png` - Feature importance charts
- `predictions_plot.png` - Actual vs predicted
- `residuals_plot.png` - Residual analysis
- `comprehensive_comparison.png` - Multi-panel comparison
- `segmentation_*_performance.png` - Performance by income/region
- `segmentation_*_features.png` - Feature importance heatmaps
- `statistical_tests_*.png` - Bootstrap and consistency plots

**Reports and Metrics:**
- `model_comparison.csv` - Performance metrics
- `comprehensive_metrics.csv` - Detailed metrics
- `segmentation_*_results.csv` - Segmentation analysis
- `statistical_tests_*.csv` - Statistical test results
- `*.txt` - Text reports

**LaTeX Tables** (in `report/tables/`):
- `table1_descriptive.tex` - Descriptive statistics
- `table2_performance.tex` - Model performance
- `table4_income.tex` - Income segmentation

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
- Comprehensive LaTeX report in `report/paper.tex`
- Includes literature review, methodology, and key findings
- Compile with: `cd report && pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex`

## Data Source

All data from [World Bank Open Data API](https://data.worldbank.org/)
- ~50 socioeconomic indicators from 2000-2023
- API Docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392

## Limitations

- **Missing Data**: Many indicators incomplete, especially for developing countries
- **Temporal Coverage**: GINI data not available for all countries/years
- **Causality**: Models show correlations, not causal relationships
- **Lag Effects**: Current model doesn't account for time lags

## Project Notes

This is a research/educational project implementing ML best practices:
- Centralized configuration and hyperparameters
- Comprehensive type hints and documentation
- DRY principle applied throughout
- Parallel processing for performance
- Model versioning and caching

Predictions should not be used for policy decisions without thorough validation and expert review.
