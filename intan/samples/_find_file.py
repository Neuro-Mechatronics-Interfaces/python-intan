"""
_find_file.py

Utility module for locating files within a specified directory tree. Primarily used to find example or
test files bundled with the Intan data processing package. If no custom path is provided, the search
defaults to the 'samples' directory adjacent to the module.

Functions:
- findFile: Search for a specific filename within a directory tree and return its absolute path.
"""
import os


def findFile(filename, search_path=None):
    """
    Search for a file in the given search path. If None, defaults to the intan/samples directory.

    Parameters:
    - filename (str): The name of the file to search for.
    - search_path (str): Optional path to search in. Defaults to intan/samples directory.

    Returns:
    - str or None: Full path to the file if found, otherwise None.
    """
    if search_path is None:
        # Use the path to the 'samples' directory inside this module
        search_path = os.path.join(os.path.dirname(__file__), "..", "samples")
        search_path = os.path.abspath(search_path)

    for root, _, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)

    return None

