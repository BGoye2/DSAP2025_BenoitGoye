"""
Utility functions and classes for the GINI prediction pipeline
"""

import os
import subprocess
import sys
import threading
import time
from datetime import datetime


def validate_environment(verbose=True, exit_on_failure=False):
    """
    Validate that all required dependencies are installed with correct versions.

    This function checks:
    - Python version (>= 3.8, recommends 3.12+)
    - All required packages from requirements.txt
    - Package versions meet minimum requirements

    Parameters:
    -----------
    verbose : bool, default=True
        If True, print detailed validation results
    exit_on_failure : bool, default=False
        If True, call sys.exit(1) on validation failure

    Returns:
    --------
    bool
        True if all requirements are met, False otherwise

    Examples:
    ---------
    >>> from src.utils import validate_environment
    >>> validate_environment()  # Check all dependencies
    True
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        from packaging.version import parse as parse_version
    except ImportError:
        if verbose:
            print("ERROR: 'packaging' library not found.")
            print("Install with: pip install packaging>=23.0")
        if exit_on_failure:
            sys.exit(1)
        return False

    # Define required packages with minimum versions (from requirements.txt)
    REQUIRED_PACKAGES = {
        'pandas': '2.2.0',
        'numpy': '1.26.0',
        'scipy': '1.13.0',
        'scikit-learn': '1.5.0',
        'joblib': '1.4.0',
        'xgboost': '3.1.0',
        'lightgbm': '4.6.0',
        'matplotlib': '3.9.0',
        'seaborn': '0.13.0',
        'requests': '2.32.0',
    }

    # Check Python version
    python_version = sys.version_info
    python_ok = python_version >= (3, 8)
    python_recommended = python_version >= (3, 12)

    if verbose:
        print("=" * 70)
        print("ENVIRONMENT VALIDATION")
        print("=" * 70)
        print(f"\n{'Python Version:':<30} {python_version.major}.{python_version.minor}.{python_version.micro}")

        if not python_ok:
            print(f"{'Status:':<30} ❌ FAILED (requires Python >= 3.8)")
        elif not python_recommended:
            print(f"{'Status:':<30} ⚠️  OK (Python >= 3.12 recommended)")
        else:
            print(f"{'Status:':<30} ✓ OK")
        print()

    # Check package versions
    missing_packages = []
    outdated_packages = []
    valid_packages = []

    if verbose:
        print(f"{'Package':<20} {'Required':<15} {'Installed':<15} {'Status'}")
        print("-" * 70)

    for package, min_version in REQUIRED_PACKAGES.items():
        try:
            installed_version = version(package)
            installed_parsed = parse_version(installed_version)
            required_parsed = parse_version(min_version)

            if installed_parsed >= required_parsed:
                valid_packages.append(package)
                status = "✓ OK"
            else:
                outdated_packages.append((package, min_version, installed_version))
                status = "⚠️  Outdated"

            if verbose:
                print(f"{package:<20} >={min_version:<14} {installed_version:<15} {status}")

        except PackageNotFoundError:
            missing_packages.append((package, min_version))
            if verbose:
                print(f"{package:<20} >={min_version:<14} {'NOT FOUND':<15} ❌ MISSING")

    # Summary
    all_valid = python_ok and not missing_packages and not outdated_packages

    if verbose:
        print("-" * 70)
        print(f"\nValidation Summary:")
        print(f"  ✓ Valid packages: {len(valid_packages)}/{len(REQUIRED_PACKAGES)}")

        if missing_packages:
            print(f"  ❌ Missing packages: {len(missing_packages)}")
        if outdated_packages:
            print(f"  ⚠️  Outdated packages: {len(outdated_packages)}")

        # Installation instructions if issues found
        if missing_packages or outdated_packages:
            print("\n" + "=" * 70)
            print("INSTALLATION REQUIRED")
            print("=" * 70)

            if missing_packages:
                print("\nMissing packages:")
                for pkg, ver in missing_packages:
                    print(f"  • {pkg}>={ver}")

            if outdated_packages:
                print("\nOutdated packages (upgrade recommended):")
                for pkg, required, installed in outdated_packages:
                    print(f"  • {pkg}: {installed} → >={required}")

            print("\nTo fix, run one of:")
            print("  pip install -r requirements.txt --upgrade")
            print("  conda env update -f environment.yml --prune")
            print("=" * 70)

        else:
            print(f"\n{'Status:':<30} ✓ All dependencies validated")
            print("=" * 70)

    if not all_valid and exit_on_failure:
        sys.exit(1)

    return all_valid


def ensure_directories_exist():
    """
    Ensure all required output directories exist.

    Creates the following directories if they don't exist:
    - output/
    - output/figures/
    - output/tables/
    - output/.cache/

    This function should be called once at the beginning of the pipeline
    (e.g., in main.py) to set up the directory structure.
    """
    # Import here to avoid circular dependency
    import sys
    from pathlib import Path

    # Determine if we're being called from main.py or from src/
    if 'src.utils' in sys.modules:
        # Called from main.py (imports as src.utils)
        from src.config.constants import CACHE_DIR, FIGURES_DIR, OUTPUT_DIR, TABLES_DIR
    else:
        # Called from within src/ directory
        from config.constants import CACHE_DIR, FIGURES_DIR, OUTPUT_DIR, TABLES_DIR

    directories = [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, CACHE_DIR]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


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
