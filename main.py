"""
GINI Coefficient Prediction Pipeline
=====================================
Main script to run the complete machine learning pipeline for predicting
GINI coefficients using World Bank data.

Usage:
    python main.py                    # Quick run (recommended)
    python main.py --mode fast        # Fast run with recent data only
    python main.py --mode optimized   # With hyperparameter tuning (slower)
    python main.py --mode custom --skip-collection --start-year 2010

For more information, run: python main.py --help
"""

import argparse
import os
import sys
from datetime import datetime

from src.config.constants import PROCESSED_DATA_PATH, WORLD_BANK_DATA_PATH
from src.utils import (
    display_completion_summary,
    display_error_summary,
    ensure_directories_exist,
    print_header,
    print_step,
    run_script,
    validate_environment,
    validate_file_exists,
)


def run_pipeline(
    collect_data=True,
    preprocess_data=True,
    train_models=True,
    evaluate_models=True,
    compare_models=True,
    tune_hyperparameters=False,
    start_year=2000,
    end_year=2023
):
    """
    Execute the complete GINI prediction pipeline.

    The pipeline consists of the following stages:
    1. Data Collection - Fetch data from World Bank API
    2. Data Preprocessing - Clean and prepare data for modeling
    3. Model Training - Train multiple ML models
    4. Model Evaluation - Basic evaluation and visualizations
    5. Comprehensive Comparison - Detailed model analysis
    6. Segmentation Analysis - Analyze performance by income/region
    7. Statistical Tests - Validate results with statistical tests
    8. LaTeX Tables - Generate publication-ready tables

    Parameters:
    -----------
    collect_data : bool, default=True
        Whether to collect data from World Bank API
    preprocess_data : bool, default=True
        Whether to preprocess the data
    train_models : bool, default=True
        Whether to train the models
    evaluate_models : bool, default=True
        Whether to evaluate models (basic metrics and visualizations)
    compare_models : bool, default=True
        Whether to run comprehensive model comparison and analysis
    tune_hyperparameters : bool, default=False
        Whether to perform hyperparameter tuning (slower but better results)
    start_year : int, default=2000
        Start year for data collection
    end_year : int, default=2023
        End year for data collection
    """
    # Validate environment dependencies before starting pipeline
    validate_environment(verbose=True, exit_on_failure=True)

    # Ensure output directories exist before starting pipeline
    ensure_directories_exist()

    start_time = datetime.now()

    # Calculate total steps for progress tracking
    total_steps = sum([collect_data, preprocess_data, train_models, evaluate_models, compare_models])
    if compare_models:
        total_steps += 3  # Segmentation, statistical tests, LaTeX tables
    current_step = 0

    # Display configuration
    print_header("GINI COEFFICIENT PREDICTION PIPELINE")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  • Collect data: {collect_data}")
    print(f"  • Preprocess data: {preprocess_data}")
    print(f"  • Train models: {train_models}")
    print(f"  • Evaluate models: {evaluate_models}")
    print(f"  • Compare models: {compare_models}")
    if compare_models:
        print(f"  • Segmentation analysis: Enabled")
        print(f"  • Statistical tests: Enabled")
        print(f"  • LaTeX table generation: Enabled")
    print(f"  • Hyperparameter tuning: {tune_hyperparameters}")
    print(f"  • Year range: {start_year}-{end_year}")

    try:
        # STEP 1: Data Collection
        if collect_data:
            current_step += 1
            print_step(current_step, total_steps, "Data Collection from World Bank API")

            run_script(
                'src/01_data_collection.py',
                "Fetching data from World Bank API",
                "Data collection completed successfully"
            )

            validate_file_exists(
                WORLD_BANK_DATA_PATH,
                "Data collection failed - world_bank_data.csv not found"
            )

        # STEP 2: Data Preprocessing
        if preprocess_data:
            current_step += 1
            print_step(current_step, total_steps, "Data Preprocessing")

            validate_file_exists(
                WORLD_BANK_DATA_PATH,
                "world_bank_data.csv not found. Run data collection first."
            )

            run_script(
                'src/02_data_preprocessing.py',
                "Cleaning and preprocessing data",
                "Data preprocessing completed successfully"
            )

            validate_file_exists(
                PROCESSED_DATA_PATH,
                "Preprocessing failed - processed_data.csv not found"
            )

        # STEP 3: Model Training
        if train_models:
            current_step += 1
            print_step(current_step, total_steps, "Model Training")

            validate_file_exists(
                PROCESSED_DATA_PATH,
                "processed_data.csv not found. Run preprocessing first."
            )

            # Pass --tune flag if hyperparameter tuning is enabled
            training_args = ['--tune'] if tune_hyperparameters else []

            run_script(
                'src/03_model_training.py',
                "Training machine learning models" + (" with hyperparameter tuning" if tune_hyperparameters else ""),
                "Model training completed successfully",
                args=training_args
            )

        # STEP 4: Model Evaluation
        if evaluate_models:
            current_step += 1
            print_step(current_step, total_steps, "Model Evaluation")

            if not os.path.exists(PROCESSED_DATA_PATH):
                print("⚠ Warning: processed_data.csv not found. Skipping evaluation step...")
            else:
                run_script(
                    'src/04_model_evaluation.py',
                    "Evaluating models and generating visualizations",
                    "Model evaluation completed successfully"
                )

        # STEP 5: Comprehensive Model Comparison
        if compare_models:
            current_step += 1
            print_step(current_step, total_steps, "Comprehensive Model Comparison")

            if not os.path.exists(PROCESSED_DATA_PATH):
                print("⚠ Warning: processed_data.csv not found. Skipping comparison step...")
            else:
                run_script(
                    'src/05_comprehensive_comparison.py',
                    "Running comprehensive model analysis",
                    "Model comparison completed successfully"
                )

        # STEP 6: Segmentation Analysis
        if compare_models:
            current_step += 1
            print_step(current_step, total_steps, "Segmentation Analysis")

            if not os.path.exists(PROCESSED_DATA_PATH):
                print("⚠ Warning: processed_data.csv not found. Skipping segmentation analysis...")
            else:
                try:
                    run_script(
                        'src/06_segmentation_analysis.py',
                        "Analyzing performance across income levels and regions",
                        "Segmentation analysis completed successfully"
                    )
                except RuntimeError as e:
                    print(f"⚠ Warning: Segmentation analysis failed but continuing pipeline...")
                    print(f"  Error: {e}")

        # STEP 7: Statistical Significance Tests
        if compare_models:
            current_step += 1
            print_step(current_step, total_steps, "Statistical Significance Tests")

            if not os.path.exists(PROCESSED_DATA_PATH):
                print("⚠ Warning: processed_data.csv not found. Skipping statistical tests...")
            else:
                try:
                    run_script(
                        'src/07_statistical_tests.py',
                        "Running bootstrap, permutation, and consistency tests",
                        "Statistical significance tests completed successfully"
                    )
                except RuntimeError as e:
                    print(f"⚠ Warning: Statistical tests failed but continuing pipeline...")
                    print(f"  Error: {e}")

        # STEP 8: Populate LaTeX Tables
        if compare_models:
            current_step += 1
            print_step(current_step, total_steps, "LaTeX Table Generation")

            try:
                run_script(
                    'src/08_populate_paper_tables.py',
                    "Generating LaTeX table content from results",
                    "LaTeX tables generated and inserted into paper.tex"
                )
            except RuntimeError as e:
                print(f"⚠ Warning: Table population failed but continuing pipeline...")
                print(f"  Error: {e}")

        # Display final summary
        display_completion_summary(start_time)

    except Exception as e:
        display_error_summary(e, current_step, total_steps)
        sys.exit(1)


# ============================================================================
# PRESET PIPELINE CONFIGURATIONS
# ============================================================================

def quick_run():
    """
    Quick run with default settings (RECOMMENDED).

    • Full pipeline from 2000-2023
    • No hyperparameter tuning
    • Good balance of speed and accuracy
    """
    run_pipeline(
        collect_data=True,
        preprocess_data=True,
        train_models=True,
        evaluate_models=True,
        compare_models=True,
        tune_hyperparameters=False,
        start_year=2000,
        end_year=2023
    )


def fast_run():
    """
    Fast run with recent data only (2015-2023).

    • Smaller dataset for faster execution
    • Good for quick testing and debugging
    """
    run_pipeline(
        collect_data=True,
        preprocess_data=True,
        train_models=True,
        evaluate_models=True,
        compare_models=True,
        tune_hyperparameters=False,
        start_year=2015,
        end_year=2023
    )


def optimized_run():
    """
    Optimized run with hyperparameter tuning (SLOW).

    • Full pipeline with grid search optimization
    • Best possible model accuracy
    • Use for final production models
    """
    run_pipeline(
        collect_data=True,
        preprocess_data=True,
        train_models=True,
        evaluate_models=True,
        compare_models=True,
        tune_hyperparameters=True,
        start_year=2000,
        end_year=2023
    )


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='GINI Coefficient Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Quick run (recommended)
  python main.py --mode fast                  # Fast run with recent data
  python main.py --mode optimized             # With hyperparameter tuning
  python main.py --mode custom --skip-collection --start-year 2010

For more information, see the README.md file.
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='quick',
        choices=['quick', 'fast', 'optimized', 'custom'],
        help='Pipeline mode (default: quick)'
    )

    parser.add_argument(
        '--skip-collection',
        action='store_true',
        help='Skip data collection step'
    )

    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip preprocessing step'
    )

    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training step'
    )

    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip model evaluation step'
    )

    parser.add_argument(
        '--skip-comparison',
        action='store_true',
        help='Skip comprehensive model comparison and analysis'
    )

    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning (slower but better)'
    )

    parser.add_argument(
        '--start-year',
        type=int,
        default=2000,
        help='Start year for data collection (default: 2000)'
    )

    parser.add_argument(
        '--end-year',
        type=int,
        default=2023,
        help='End year for data collection (default: 2023)'
    )

    args = parser.parse_args()

    # Execute selected pipeline mode
    if args.mode == 'quick':
        quick_run()
    elif args.mode == 'fast':
        fast_run()
    elif args.mode == 'optimized':
        optimized_run()
    else:  # custom mode
        run_pipeline(
            collect_data=not args.skip_collection,
            preprocess_data=not args.skip_preprocessing,
            train_models=not args.skip_training,
            evaluate_models=not args.skip_evaluation,
            compare_models=not args.skip_comparison,
            tune_hyperparameters=args.tune,
            start_year=args.start_year,
            end_year=args.end_year
        )
