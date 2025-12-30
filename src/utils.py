"""
Utility functions and classes for the GINI prediction pipeline
"""

import sys
import os
import time
import threading
import subprocess
from datetime import datetime


def print_header(text):
    """
    Print a formatted header with border lines for visual separation.

    Parameters:
    -----------
    text : str
        The header text to display
    """
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def print_step(step_num, total_steps, description):
    """
    Print progress information for a pipeline step.

    Parameters:
    -----------
    step_num : int
        Current step number
    total_steps : int
        Total number of steps in the pipeline
    description : str
        Description of the current step
    """
    print(f"\n{'─'*70}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'─'*70}\n")


class Spinner:
    """
    A simple loading spinner for long-running tasks.

    Displays an animated spinner in the terminal to indicate progress
    during operations that take significant time to complete.
    """

    def __init__(self, message="Processing"):
        """
        Initialize the spinner.

        Parameters:
        -----------
        message : str
            Message to display next to the spinner animation
        """
        # Unicode Braille patterns for smooth animation
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.message = message
        self.running = False
        self.thread = None

    def _spin(self):
        """
        Internal method to animate the spinner.

        Runs in a separate thread and cycles through spinner characters
        until stopped by the stop() method.
        """
        i = 0
        while self.running:
            # Display current spinner character with message
            sys.stdout.write(f'\r{self.spinner_chars[i % len(self.spinner_chars)]} {self.message}...')
            sys.stdout.flush()
            time.sleep(0.1)  # Animation speed
            i += 1

    def start(self):
        """
        Start the spinner animation in a background thread.

        Creates a daemon thread to avoid blocking the main program execution.
        """
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True  # Thread will terminate when main program exits
        self.thread.start()

    def stop(self, final_message=None):
        """
        Stop the spinner animation and optionally display a final message.

        Parameters:
        -----------
        final_message : str, optional
            Message to display after stopping the spinner
        """
        self.running = False
        if self.thread:
            self.thread.join()  # Wait for spinner thread to finish
        # Clear the spinner line
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        if final_message:
            print(final_message)
        sys.stdout.flush()


def run_script(script_path, spinner_message, success_message, args=None):
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
    args : list, optional
        Additional command-line arguments to pass to the script

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

    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)

    result = subprocess.run(
        cmd,
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


def validate_file_exists(filepath, error_message):
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


def display_completion_summary(start_time):
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
        'figures/feature_importance.png',
        'figures/predictions_plot.png',
        'figures/residuals_plot.png',
        'comprehensive_metrics.csv',
        'statistical_comparison.csv',
        'segment_performance.csv',
        'figures/comprehensive_comparison.png',
        'figures/error_analysis.png',
        'figures/segment_performance.png',
        'model_comparison_report.txt',
        'figures/segmentation_income_performance.png',
        'figures/segmentation_income_features.png',
        'segmentation_income_results.csv',
        'figures/segmentation_regional_performance.png',
        'figures/segmentation_regional_features.png',
        'segmentation_regional_results.csv',
        'segmentation_summary_report.txt',
        'statistical_tests_bootstrap.csv',
        'figures/statistical_tests_bootstrap.png',
        'statistical_tests_permutation.csv',
        'statistical_tests_consistency.csv',
        'figures/statistical_tests_consistency.png',
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

    print("─"*70)


def display_error_summary(error, current_step, total_steps):
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
    import traceback

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

    traceback.print_exc()
