"""
Constants Module for GINI Coefficient Prediction Pipeline
===========================================================
Centralized constants used across the pipeline for consistency and maintainability.

This module defines:
- Random seeds for reproducibility
- File paths and directory locations
- Model configuration and hyperparameters
- Plot dimensions and styling
- Validation thresholds
- Model name mappings
- Feature name mappings
"""

import os

from .feature_names import get_display_name, FEATURE_NAME_MAPPING

# =============================================================================
# TARGET VARIABLE
# =============================================================================

TARGET_VARIABLE = 'SI.POV.GINI'  # GINI coefficient - primary target for prediction
TARGET_DISPLAY_NAME = 'GINI Coefficient'  # Human-readable name for display

# =============================================================================
# RANDOM SEEDS
# =============================================================================

DEFAULT_RANDOM_SEED = 42  # Base seed for all random operations (ensures reproducibility)


# =============================================================================
# FILE PATHS
# =============================================================================

# Directory paths
OUTPUT_DIR = 'output'
CACHE_DIR = os.path.join(OUTPUT_DIR, '.cache')
CONFIG_DIR = 'src/config'

# Output subdirectories
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')

# Model and data paths
MODELS_PATH = os.path.join(OUTPUT_DIR, 'trained_models.pkl')
PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, 'processed_data.csv')
WORLD_BANK_DATA_PATH = os.path.join(OUTPUT_DIR, 'world_bank_data.csv')
FEATURE_NAMES_PATH = os.path.join(OUTPUT_DIR, 'feature_names.csv')

# Configuration file paths
COUNTRY_REGIONS_PATH = os.path.join(CONFIG_DIR, 'country_regions.json')


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Model name mapping (internal keys -> display names)
MODEL_NAME_MAPPING = {
    'decision_tree': 'Decision Tree',
    'random_forest': 'Random Forest',
    'gradient_boosting': 'Gradient Boosting',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM'
}

# Reverse mapping (display names -> internal keys)
DISPLAY_TO_INTERNAL_MAPPING = {v: k for k, v in MODEL_NAME_MAPPING.items()}

# Default model hyperparameters
# These values are chosen based on typical ranges for tree-based models
# and balance between model complexity and training time
DEFAULT_HYPERPARAMETERS = {
    'decision_tree': {
        'max_depth': 10,         # Prevents overfitting while allowing reasonable complexity
        'min_samples_split': 10, # Requires 10 samples to split (reduces noise sensitivity)
        'min_samples_leaf': 4,   # Each leaf must have at least 4 samples
        'random_state': DEFAULT_RANDOM_SEED
    },
    'random_forest': {
        'n_estimators': 200,      # 200 trees for stable ensemble predictions
        'max_depth': 20,          # Deeper trees allowed due to ensemble averaging
        'min_samples_split': 5,   # Standard split threshold
        'min_samples_leaf': 2,    # Minimum leaf size
        'max_features': 'sqrt',   # Square root of features for diversity
        'random_state': DEFAULT_RANDOM_SEED,
        'n_jobs': -1              # Use all CPU cores
    },
    'gradient_boosting': {
        'n_estimators': 200,      # 200 boosting iterations
        'learning_rate': 0.05,    # Conservative learning rate for stability
        'max_depth': 5,           # Shallow trees typical for gradient boosting
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'subsample': 0.9,         # Use 90% of samples per tree (prevents overfitting)
        'random_state': DEFAULT_RANDOM_SEED
    },
    'xgboost': {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_child_weight': 3,    # Minimum sum of instance weight in child
        'subsample': 0.9,
        'colsample_bytree': 0.9,  # Use 90% of features per tree
        'gamma': 0.1,             # Minimum loss reduction for split
        'random_state': DEFAULT_RANDOM_SEED,
        'n_jobs': -1,
        'tree_method': 'hist'     # Histogram-based method for speed
    },
    'lightgbm': {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 5,
        'num_leaves': 50,         # Max number of leaves per tree
        'min_child_samples': 20,  # Minimum data in leaf
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.1,         # L1 regularization
        'reg_lambda': 0.1,        # L2 regularization
        'random_state': DEFAULT_RANDOM_SEED,
        'n_jobs': -1,
        'verbose': -1             # Suppress output
    }
}

# Hyperparameter grids for grid search tuning
# REDUCED GRIDS: Optimized for faster tuning (~90% reduction in combinations)
# Original: 36,906 combinations → Reduced: ~432 combinations
HYPERPARAMETER_GRIDS = {
    'decision_tree': {
        'max_depth': [5, 10, None],              # 3 values (was 5)
        'min_samples_split': [5, 10],            # 2 values (was 4)
        'min_samples_leaf': [2, 4],              # 2 values (was 4)
        'max_features': ['sqrt', None]           # 2 values (was 3)
    },  # Total: 3 × 2 × 2 × 2 = 24 combinations (was 240)
    'random_forest': {
        'n_estimators': [100, 200],              # 2 values (was 3)
        'max_depth': [15, 20, None],             # 3 values (was 4)
        'min_samples_split': [2, 5],             # 2 values (was 3)
        'min_samples_leaf': [1, 2],              # 2 values (was 3)
        'max_features': ['sqrt', 'log2']         # 2 values (same)
    },  # Total: 2 × 3 × 2 × 2 × 2 = 48 combinations (was 216)
    'gradient_boosting': {
        'n_estimators': [100, 200],              # 2 values (was 3)
        'learning_rate': [0.05, 0.1],            # 2 values (was 3)
        'max_depth': [3, 5],                     # 2 values (was 3)
        'min_samples_split': [2, 5],             # 2 values (was 3)
        'min_samples_leaf': [1, 2],              # 2 values (was 3)
        'subsample': [0.8, 0.9]                  # 2 values (was 3)
    },  # Total: 2 × 2 × 2 × 2 × 2 × 2 = 64 combinations (was 729)
    'xgboost': {
        'n_estimators': [100, 200],              # 2 values (was 3)
        'learning_rate': [0.05, 0.1],            # 2 values (was 3)
        'max_depth': [3, 5, 7],                  # 3 values (was 4)
        'min_child_weight': [1, 3],              # 2 values (was 3)
        'subsample': [0.8, 0.9],                 # 2 values (was 3)
        'colsample_bytree': [0.8, 0.9],          # 2 values (was 3)
        'gamma': [0, 0.1]                        # 2 values (was 3)
    },  # Total: 2 × 2 × 3 × 2 × 2 × 2 × 2 = 192 combinations (was 2,916)
    'lightgbm': {
        'n_estimators': [100, 200],              # 2 values (was 3)
        'learning_rate': [0.05, 0.1],            # 2 values (was 3)
        'max_depth': [3, 5, -1],                 # 3 values (was 5)
        'num_leaves': [31, 50],                  # 2 values (was 4)
        'min_child_samples': [20, 30],           # 2 values (was 3)
        'subsample': [0.8, 0.9],                 # 2 values (was 3)
        'colsample_bytree': [0.8, 0.9],          # 2 values (was 3)
        'reg_alpha': [0, 0.1],                   # 2 values (was 3)
        'reg_lambda': [0, 0.1]                   # 2 values (was 3)
    }  # Total: 2 × 2 × 3 × 2 × 2 × 2 × 2 × 2 × 2 = 384 combinations (was 32,805)
}


# =============================================================================
# PLOTTING CONFIGURATION
# =============================================================================

# Plot dimensions
PLOT_WIDTH_PER_MODEL = 7      # Width multiplier per model in comparison plots
PLOT_HEIGHT_STANDARD = 6      # Standard height for plots
MAX_PLOT_COLUMNS = 3          # Maximum columns in subplot grids

# Plot colors (using matplotlib default color cycle)
PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# DPI for saving figures
PLOT_DPI = 300


# =============================================================================
# ANALYSIS THRESHOLDS
# =============================================================================

# Segmentation analysis
MIN_SEGMENT_SIZE = 30          # Minimum observations required for segment analysis
MAX_FEATURE_NAME_LENGTH = 30   # Maximum length for feature names in plots

# GINI segment boundaries for performance analysis
GINI_SEGMENT_BOUNDARIES = [0, 30, 40, 100]  # Low (<30), Moderate (30-40), High (>40)
GINI_SEGMENT_LABELS = ['Low Inequality', 'Moderate Inequality', 'High Inequality']


# =============================================================================
# PREDICTION CONFIGURATION
# =============================================================================

# Ensemble prediction weights
# Based on typical performance: RF and XGBoost strong, GB moderate, LightGBM good, DT weak
DEFAULT_ENSEMBLE_WEIGHTS = {
    'decision_tree': 0.05,      # 5% - Simple baseline
    'random_forest': 0.20,      # 20% - Strong performer
    'gradient_boosting': 0.15,  # 15% - Moderate performer
    'xgboost': 0.30,            # 30% - Best performer typically
    'lightgbm': 0.30            # 30% - Best performer typically
}

# Weight validation tolerance
WEIGHT_TOLERANCE = 0.01  # Allow ±1% deviation from sum=1.0

# Predictions output suffix
PREDICTIONS_SUFFIX = '_predictions'


# =============================================================================
# UI/FORMATTING
# =============================================================================

# Text separators
SEPARATOR_WIDTH = 70
SECTION_SEPARATOR = "=" * SEPARATOR_WIDTH
SUBSECTION_SEPARATOR = "-" * SEPARATOR_WIDTH

# Pipeline step counting
SEGMENTATION_STEPS = 3  # Number of additional steps when compare_models=True


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

CV_FOLDS = 5  # Number of cross-validation folds
CV_SCORING = 'neg_mean_squared_error'  # Scoring metric for CV


# =============================================================================
# BOOTSTRAP CONFIGURATION
# =============================================================================

BOOTSTRAP_ITERATIONS = 100  # Number of bootstrap samples for feature importance
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95  # 95% confidence intervals


# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

CACHE_ENABLED = True  # Enable caching by default
