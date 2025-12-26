"""
Utility functions and classes for the GINI prediction pipeline
"""

import sys
import time
import threading


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
