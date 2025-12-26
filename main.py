"""
Main Pipeline Script
Runs the entire GINI prediction workflow from data collection to model training
"""

import os
import sys
import time
import threading
from datetime import datetime


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def print_step(step_num, total_steps, description):
    """Print step information"""
    print(f"\n{'─'*70}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'─'*70}\n")


class Spinner:
    """Simple loading spinner for long-running tasks"""

    def __init__(self, message="Processing"):
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.message = message
        self.running = False
        self.thread = None

    def _spin(self):
        """Internal method to animate the spinner"""
        i = 0
        while self.running:
            sys.stdout.write(f'\r{self.spinner_chars[i % len(self.spinner_chars)]} {self.message}...')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def start(self):
        """Start the spinner animation"""
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()

    def stop(self, final_message=None):
        """Stop the spinner animation"""
        self.running = False
        if self.thread:
            self.thread.join()
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')  # Clear the line
        if final_message:
            print(final_message)
        sys.stdout.flush()


def run_pipeline(collect_data=True, preprocess_data=True, train_models=True,
                compare_models=True, tune_hyperparameters=False, 
                start_year=2000, end_year=2023):
    """
    Run the complete pipeline
    
    Parameters:
    -----------
    collect_data : bool
        Whether to collect data from World Bank API
    preprocess_data : bool
        Whether to preprocess the data
    train_models : bool
        Whether to train the models
    compare_models : bool
        Whether to run comprehensive model comparison
    tune_hyperparameters : bool
        Whether to perform hyperparameter tuning (slower but better)
    start_year : int
        Start year for data collection
    end_year : int
        End year for data collection
    """
    
    start_time = datetime.now()
    # Total steps includes segmentation, statistical tests, and table population if compare_models is True
    total_steps = sum([collect_data, preprocess_data, train_models, compare_models])
    if compare_models:
        total_steps += 3  # Add segmentation analysis, statistical tests, and LaTeX table population
    current_step = 0

    print_header("GINI COEFFICIENT PREDICTION PIPELINE")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  - Collect data: {collect_data}")
    print(f"  - Preprocess data: {preprocess_data}")
    print(f"  - Train models: {train_models}")
    print(f"  - Compare models: {compare_models}")
    if compare_models:
        print(f"  - Segmentation analysis: Enabled")
        print(f"  - Statistical tests: Enabled")
    print(f"  - Hyperparameter tuning: {tune_hyperparameters}")
    print(f"  - Year range: {start_year}-{end_year}")
    
    try:
        # Step 1: Data Collection
        if collect_data:
            current_step += 1
            print_step(current_step, total_steps, "Data Collection from World Bank API")

            # Run as subprocess to avoid namespace issues
            import subprocess

            spinner = Spinner("Fetching data from World Bank API")
            spinner.start()

            result = subprocess.run([sys.executable, 'code/01_data_collection.py'],
                                  capture_output=True, text=True)

            spinner.stop()

            if result.returncode != 0:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                raise RuntimeError(f"Data collection failed with return code {result.returncode}")

            if not os.path.exists('output/world_bank_data.csv'):
                raise FileNotFoundError("Data collection failed - world_bank_data.csv not found")

            print("✓ Data collection completed successfully")
        
        # Step 2: Data Preprocessing
        if preprocess_data:
            current_step += 1
            print_step(current_step, total_steps, "Data Preprocessing")

            if not os.path.exists('output/world_bank_data.csv'):
                raise FileNotFoundError("world_bank_data.csv not found. Run data collection first.")

            # Run as subprocess to avoid namespace issues
            import subprocess

            spinner = Spinner("Cleaning and preprocessing data")
            spinner.start()

            result = subprocess.run([sys.executable, 'code/02_data_preprocessing.py'],
                                  capture_output=True, text=True)

            spinner.stop()

            if result.returncode != 0:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                raise RuntimeError(f"Preprocessing failed with return code {result.returncode}")

            if not os.path.exists('output/processed_data.csv'):
                raise FileNotFoundError("Preprocessing failed - processed_data.csv not found")

            print("✓ Data preprocessing completed successfully")
        
        # Step 3: Model Training
        if train_models:
            current_step += 1
            print_step(current_step, total_steps, "Model Training and Evaluation")

            if not os.path.exists('output/processed_data.csv'):
                raise FileNotFoundError("processed_data.csv not found. Run preprocessing first.")

            # Run as subprocess to avoid namespace issues
            import subprocess

            # For now, run with default settings (tune_hyperparameters would need to be handled differently)
            if tune_hyperparameters:
                print("Note: Hyperparameter tuning setting will use default from code/03_model_training.py")
                print("To enable tuning, edit the tune_hyperparameters variable in code/03_model_training.py")

            spinner = Spinner("Training models")
            spinner.start()

            result = subprocess.run([sys.executable, 'code/03_model_training.py'],
                                  capture_output=True, text=True)

            spinner.stop()

            if result.returncode != 0:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                raise RuntimeError(f"Model training failed with return code {result.returncode}")

            # Check if comparison file was created
            if os.path.exists('output/model_comparison.csv'):
                print("✓ Model training completed successfully")
            else:
                print("Warning: model_comparison.csv not found")
        
        # Step 4: Comprehensive Model Comparison
        if compare_models:
            current_step += 1
            print_step(current_step, total_steps, "Model Comparison")

            if not os.path.exists('output/processed_data.csv'):
                print("Warning: processed_data.csv not found")
                print("Skipping comparison step...")
            else:
                # Run as subprocess to avoid namespace issues
                import subprocess

                spinner = Spinner("Running model analysis")
                spinner.start()

                result = subprocess.run([sys.executable, 'code/05_comprehensive_comparison.py'],
                                      capture_output=True, text=True)

                spinner.stop()

                if result.returncode != 0:
                    print("STDOUT:", result.stdout)
                    print("STDERR:", result.stderr)
                    raise RuntimeError(f"Comparison failed with return code {result.returncode}")

                print("✓ Model comparison completed successfully")

        # Step 5: Segmentation Analysis
        if compare_models:  # Run if we're doing comprehensive analysis
            current_step += 1
            print_step(current_step, total_steps, "Segmentation Analysis")

            if not os.path.exists('output/processed_data.csv'):
                print("Warning: processed_data.csv not found")
                print("Skipping segmentation analysis...")
            else:
                import subprocess

                spinner = Spinner("Analyzing performance across income levels and regions")
                spinner.start()

                result = subprocess.run([sys.executable, 'code/06_segmentation_analysis.py'],
                                      capture_output=True, text=True)

                spinner.stop()

                if result.returncode != 0:
                    print("STDOUT:", result.stdout)
                    print("STDERR:", result.stderr)
                    print("Warning: Segmentation analysis failed but continuing pipeline...")
                else:
                    print("✓ Segmentation analysis completed successfully")

        # Step 6: Statistical Significance Tests
        if compare_models:  # Run if we're doing comprehensive analysis
            current_step += 1
            print_step(current_step, total_steps, "Statistical Significance Tests")

            if not os.path.exists('output/processed_data.csv'):
                print("Warning: processed_data.csv not found")
                print("Skipping statistical tests...")
            else:
                import subprocess

                spinner = Spinner("Running bootstrap, permutation, and consistency tests")
                spinner.start()

                result = subprocess.run([sys.executable, 'code/07_statistical_tests.py'],
                                      capture_output=True, text=True)

                spinner.stop()

                if result.returncode != 0:
                    print("STDOUT:", result.stdout)
                    print("STDERR:", result.stderr)
                    print("Warning: Statistical tests failed but continuing pipeline...")
                else:
                    print("✓ Statistical significance tests completed successfully")

        # Step 7: Populate LaTeX Tables (if compare_models was run)
        if compare_models:
            current_step += 1
            print_step(current_step, total_steps, "Populating LaTeX Tables")

            import subprocess

            spinner = Spinner("Generating LaTeX table content from results")
            spinner.start()

            result = subprocess.run([sys.executable, 'code/08_populate_paper_tables.py'],
                                  capture_output=True, text=True)

            spinner.stop()

            if result.returncode != 0:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                print("Warning: Table population failed but continuing pipeline...")
            else:
                print("✓ LaTeX tables generated and inserted into research_paper.tex")

        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print_header("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration}")
        
        print("\nGenerated files in output/ folder:")
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

        for filename in output_files:
            filepath = os.path.join('output', filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"  ✓ {filename:35s} ({size:,} bytes)")

        print("\n" + "─"*70)
        print("Next steps:")
        print("  1. Review output/model_comparison_report.txt for comprehensive analysis")
        print("  2. Check output/comprehensive_metrics.csv for detailed performance metrics")
        print("  3. Examine output/comprehensive_comparison.png for visual comparison")
        print("  4. Review output/segmentation_summary_report.txt for income-level insights")
        print("  5. Check output/statistical_tests_summary.txt for significance tests")
        print("  6. View output/segmentation_income_features.png for context-specific patterns")
        print("  7. Examine output/statistical_tests_bootstrap.png for confidence intervals")
        print("  8. Use code/04_predict.py to make predictions on new data")
        print("  9. Compile paper/research_paper.tex to get the full academic paper")
        print("─"*70)
        
    except Exception as e:
        print_header("PIPELINE FAILED")
        print(f"Error: {str(e)}")
        print(f"\nFailed at step {current_step}/{total_steps}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def quick_run():
    """Quick run with default settings"""
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
    """Fast run with recent data only"""
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
    """Run with hyperparameter tuning for best results (slow)"""
    run_pipeline(
        collect_data=True,
        preprocess_data=True,
        train_models=True,
        compare_models=True,
        tune_hyperparameters=True,
        start_year=2000,
        end_year=2023
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run GINI prediction pipeline')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'fast', 'optimized', 'custom'],
                       help='Pipeline mode: quick (default), fast (recent data), optimized (with tuning), or custom')
    parser.add_argument('--skip-collection', action='store_true',
                       help='Skip data collection step')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing step')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training step')
    parser.add_argument('--skip-comparison', action='store_true',
                       help='Skip comprehensive model comparison step')
    parser.add_argument('--tune', action='store_true',
                       help='Enable hyperparameter tuning')
    parser.add_argument('--start-year', type=int, default=2000,
                       help='Start year for data collection')
    parser.add_argument('--end-year', type=int, default=2023,
                       help='End year for data collection')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        quick_run()
    elif args.mode == 'fast':
        fast_run()
    elif args.mode == 'optimized':
        optimized_run()
    else:  # custom
        run_pipeline(
            collect_data=not args.skip_collection,
            preprocess_data=not args.skip_preprocessing,
            train_models=not args.skip_training,
            compare_models=not args.skip_comparison,
            tune_hyperparameters=args.tune,
            start_year=args.start_year,
            end_year=args.end_year
        )
