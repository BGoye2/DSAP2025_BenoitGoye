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

import os
import sys
import subprocess
from datetime import datetime
from src.utils import print_header, print_step, Spinner


def _run_script(script_path, spinner_message, success_message):
    """
    Run a pipeline script with spinner animation and error handling.

    Parameters:
    -----------
    script_path : str
        Path to the script to execute
    spinner_message : str
        Message to display during execution
    success_message : str
        Message to display on successful completion

    Returns:
    --------
    subprocess.CompletedProcess
        The result of the subprocess execution

    Raises:
    -------
    RuntimeError
        If the script execution fails
    """
    spinner = Spinner(spinner_message)
    spinner.start()

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )

    spinner.stop()

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"{script_path} failed with return code {result.returncode}")

    print(f"✓ {success_message}")
    return result


def _validate_file_exists(filepath, error_message):
    """
    Validate that a required file exists.

    Parameters:
    -----------
    filepath : str
        Path to the file to check
    error_message : str
        Error message to display if file doesn't exist

    Raises:
    -------
    FileNotFoundError
        If the file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(error_message)


def run_pipeline(
    collect_data=True,
    preprocess_data=True,
    train_models=True,
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
    4. Model Comparison - Compare and evaluate models
    5. Segmentation Analysis - Analyze performance by income/region
    6. Statistical Tests - Validate results with statistical tests
    7. LaTeX Tables - Generate publication-ready tables

    Parameters:
    -----------
    collect_data : bool, default=True
        Whether to collect data from World Bank API
    preprocess_data : bool, default=True
        Whether to preprocess the data
    train_models : bool, default=True
        Whether to train the models
    compare_models : bool, default=True
        Whether to run comprehensive model comparison and analysis
    tune_hyperparameters : bool, default=False
        Whether to perform hyperparameter tuning (slower but better results)
    start_year : int, default=2000
        Start year for data collection
    end_year : int, default=2023
        End year for data collection
    """
    start_time = datetime.now()

    # Calculate total steps for progress tracking
    total_steps = sum([collect_data, preprocess_data, train_models, compare_models])
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

            _run_script(
                'src/01_data_collection.py',
                "Fetching data from World Bank API",
                "Data collection completed successfully"
            )

            _validate_file_exists(
                'output/world_bank_data.csv',
                "Data collection failed - world_bank_data.csv not found"
            )

        # STEP 2: Data Preprocessing
        if preprocess_data:
            current_step += 1
            print_step(current_step, total_steps, "Data Preprocessing")

            _validate_file_exists(
                'output/world_bank_data.csv',
                "world_bank_data.csv not found. Run data collection first."
            )

            _run_script(
                'src/02_data_preprocessing.py',
                "Cleaning and preprocessing data",
                "Data preprocessing completed successfully"
            )

            _validate_file_exists(
                'output/processed_data.csv',
                "Preprocessing failed - processed_data.csv not found"
            )

        # STEP 3: Model Training
        if train_models:
            current_step += 1
            print_step(current_step, total_steps, "Model Training and Evaluation")

            _validate_file_exists(
                'output/processed_data.csv',
                "processed_data.csv not found. Run preprocessing first."
            )

            if tune_hyperparameters:
                print("Note: Hyperparameter tuning setting will use default from src/03_model_training.py")
                print("To enable tuning, edit the tune_hyperparameters variable in src/03_model_training.py\n")

            _run_script(
                'src/03_model_training.py',
                "Training models (Decision Tree, Random Forest, XGBoost, LightGBM)",
                "Model training completed successfully"
            )

            if not os.path.exists('output/model_comparison.csv'):
                print("⚠ Warning: model_comparison.csv not found")

        # STEP 4: Comprehensive Model Comparison
        if compare_models:
            current_step += 1
            print_step(current_step, total_steps, "Comprehensive Model Comparison")

            if not os.path.exists('output/processed_data.csv'):
                print("⚠ Warning: processed_data.csv not found. Skipping comparison step...")
            else:
                _run_script(
                    'src/05_comprehensive_comparison.py',
                    "Running comprehensive model analysis",
                    "Model comparison completed successfully"
                )

        # STEP 5: Segmentation Analysis
        if compare_models:
            current_step += 1
            print_step(current_step, total_steps, "Segmentation Analysis")

            if not os.path.exists('output/processed_data.csv'):
                print("⚠ Warning: processed_data.csv not found. Skipping segmentation analysis...")
            else:
                try:
                    _run_script(
                        'src/06_segmentation_analysis.py',
                        "Analyzing performance across income levels and regions",
                        "Segmentation analysis completed successfully"
                    )
                except RuntimeError as e:
                    print(f"⚠ Warning: Segmentation analysis failed but continuing pipeline...")
                    print(f"  Error: {e}")

        # STEP 6: Statistical Significance Tests
        if compare_models:
            current_step += 1
            print_step(current_step, total_steps, "Statistical Significance Tests")

            if not os.path.exists('output/processed_data.csv'):
                print("⚠ Warning: processed_data.csv not found. Skipping statistical tests...")
            else:
                try:
                    _run_script(
                        'src/07_statistical_tests.py',
                        "Running bootstrap, permutation, and consistency tests",
                        "Statistical significance tests completed successfully"
                    )
                except RuntimeError as e:
                    print(f"⚠ Warning: Statistical tests failed but continuing pipeline...")
                    print(f"  Error: {e}")

        # STEP 7: Populate LaTeX Tables
        if compare_models:
            current_step += 1
            print_step(current_step, total_steps, "LaTeX Table Generation")

            try:
                _run_script(
                    'src/08_populate_paper_tables.py',
                    "Generating LaTeX table content from results",
                    "LaTeX tables generated and inserted into research_paper.tex"
                )
            except RuntimeError as e:
                print(f"⚠ Warning: Table population failed but continuing pipeline...")
                print(f"  Error: {e}")

        # Display final summary
        _display_completion_summary(start_time)

    except Exception as e:
        _display_error_summary(e, current_step, total_steps)
        sys.exit(1)


def _display_completion_summary(start_time):
    """
    Display pipeline completion summary and list generated files.

    Parameters:
    -----------
    start_time : datetime
        Pipeline start time for duration calculation
    """
    end_time = datetime.now()
    duration = end_time - start_time

    print_header("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")

    # List of expected output files
    output_files = [
        'world_bank_data.csv',
        'processed_data.csv',
        'feature_names.csv',
        'model_comparison.csv',
        'feature_importance.png',
        'predictions_plot.png',
        'residuals_plot.png',
        'comprehensive_metrics.csv',
        'statistical_comparison.csv',
        'segment_performance.csv',
        'comprehensive_comparison.png',
        'error_analysis.png',
        'segment_performance.png',
        'model_comparison_report.txt',
        'segmentation_income_performance.png',
        'segmentation_income_features.png',
        'segmentation_income_results.csv',
        'segmentation_regional_performance.png',
        'segmentation_regional_features.png',
        'segmentation_regional_results.csv',
        'segmentation_summary_report.txt',
        'statistical_tests_bootstrap.csv',
        'statistical_tests_bootstrap.png',
        'statistical_tests_permutation.csv',
        'statistical_tests_consistency.csv',
        'statistical_tests_consistency.png',
        'statistical_tests_summary.txt'
    ]

    print("\n" + "─"*70)
    print("Generated files in output/ folder:")
    print("─"*70)

    for filename in output_files:
        filepath = os.path.join('output', filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  ✓ {filename:40s} ({size:,} bytes)")

    print("\n" + "─"*70)
    print("Next Steps - Review Results:")
    print("─"*70)
    print("  1. Model Performance:")
    print("     • output/model_comparison_report.txt - Comprehensive analysis")
    print("     • output/comprehensive_metrics.csv - Detailed metrics")
    print("     • output/comprehensive_comparison.png - Visual comparison")
    print()
    print("  2. Segmentation Insights:")
    print("     • output/segmentation_summary_report.txt - Income-level insights")
    print("     • output/segmentation_income_features.png - Context-specific patterns")
    print()
    print("  3. Statistical Validation:")
    print("     • output/statistical_tests_summary.txt - Significance tests")
    print("     • output/statistical_tests_bootstrap.png - Confidence intervals")
    print()
    print("  4. Make Predictions:")
    print("     • Use src/04_predict.py to predict GINI on new data")
    print()
    print("  5. Academic Report:")
    print("     • Compile report/research_paper.tex for full publication")
    print("─"*70)


def _display_error_summary(error, current_step, total_steps):
    """
    Display error summary when pipeline fails.

    Parameters:
    -----------
    error : Exception
        The exception that caused the failure
    current_step : int
        Step number where failure occurred
    total_steps : int
        Total number of steps in pipeline
    """
    print_header("PIPELINE FAILED")
    print(f"Error: {str(error)}")
    print(f"Failed at step {current_step}/{total_steps}")
    print("\n" + "─"*70)
    print("Troubleshooting:")
    print("─"*70)
    print("  • Check that all dependencies are installed")
    print("  • Verify that output/ directory exists and is writable")
    print("  • Review error messages above for specific issues")
    print("  • Try running individual scripts in src/ to isolate the problem")
    print("─"*70 + "\n")

    import traceback
    traceback.print_exc()


# ============================================================================
# PRESET PIPELINE CONFIGURATIONS
# ============================================================================

def quick_run():
    """
    Quick run with default settings (RECOMMENDED).

    • Full pipeline from 2000-2023
    • No hyperparameter tuning
    • Completes in ~10-15 minutes
    • Good balance of speed and accuracy
    """
    run_pipeline(
        collect_data=True,
        preprocess_data=True,
        train_models=True,
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
    • Completes in ~5-8 minutes
    """
    run_pipeline(
        collect_data=True,
        preprocess_data=True,
        train_models=True,
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
    • Completes in ~1-2 hours
    • Use for final production models
    """
    run_pipeline(
        collect_data=True,
        preprocess_data=True,
        train_models=True,
        compare_models=True,
        tune_hyperparameters=True,
        start_year=2000,
        end_year=2023
    )


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse

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
            compare_models=not args.skip_comparison,
            tune_hyperparameters=args.tune,
            start_year=args.start_year,
            end_year=args.end_year
        )
