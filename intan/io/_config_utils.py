"""
Configuration and interactive prompt utilities for EMG applications.

Provides functions for:
- Loading/saving simple key=value config files
- GUI prompts for directories, files, and text input
- Smart defaults with config file caching
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tkinter import Tk, filedialog, simpledialog, messagebox


def load_simple_config(config_path: Path | str) -> Dict[str, Any]:
    """
    Load configuration from a simple key=value file.
    
    Format:
        # Comments start with #
        key1=value1
        key2=value2
        bool_key=true
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dictionary of config key-value pairs. Boolean strings ('true'/'false') 
        are converted to Python booleans.
        
    Example:
        >>> config = load_simple_config('.my_config')
        >>> print(config['root_dir'])
        /path/to/data
    """
    config = {}
    config_path = Path(config_path)
    
    if not config_path.exists():
        return config
    
    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle boolean values
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                # Handle numeric values
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
                    
                config[key] = value
    
    return config


def save_simple_config(config: Dict[str, Any], config_path: Path | str, header: str = "Configuration File"):
    """
    Save configuration to a simple key=value file.
    
    Args:
        config: Dictionary of config key-value pairs
        config_path: Path to save config file
        header: Optional header comment for the file
        
    Example:
        >>> config = {'root_dir': '/path/to/data', 'verbose': True}
        >>> save_simple_config(config, '.my_config', 'My App Config')
    """
    config_path = Path(config_path)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(f"# {header}\n")
        f.write("# Automatically generated - edit as needed\n\n")
        
        for key, value in config.items():
            if isinstance(value, bool):
                value = str(value).lower()
            f.write(f"{key}={value}\n")


def prompt_directory(title: str = "Select Directory", initial_dir: Optional[str] = None, use_terminal: bool = False) -> Optional[str]:
    """
    Open a directory picker dialog (GUI or terminal).
    
    Args:
        title: Dialog window title (GUI) or prompt message (terminal)
        initial_dir: Initial directory to show (default: current working directory)
        use_terminal: If True, use terminal input instead of GUI
        
    Returns:
        Selected directory path, or None if cancelled
        
    Example:
        >>> root_dir = prompt_directory("Select Model Directory")
        >>> if root_dir:
        ...     print(f"Selected: {root_dir}")
    """
    if use_terminal:
        # Terminal-based input
        default = initial_dir or os.getcwd()
        result = input(f"{title} [{default}]: ").strip()
        path = result if result else default
        return path if os.path.isdir(path) else None
    
    # GUI-based input
    try:
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        directory = filedialog.askdirectory(
            title=title,
            initialdir=initial_dir or os.getcwd()
        )
        
        root.destroy()
        return directory if directory else None
    except Exception:
        # Fallback to terminal if GUI fails
        return prompt_directory(title, initial_dir, use_terminal=True)


def prompt_file(title: str = "Select File", initial_dir: Optional[str] = None, 
                filetypes: Optional[list] = None) -> Optional[str]:
    """
    Open a GUI file picker dialog.
    
    Args:
        title: Dialog window title
        initial_dir: Initial directory to show (default: current working directory)
        filetypes: List of (label, pattern) tuples, e.g., [("RHD files", "*.rhd"), ("All files", "*.*")]
        
    Returns:
        Selected file path, or None if cancelled
        
    Example:
        >>> file_path = prompt_file("Select RHD File", filetypes=[("RHD files", "*.rhd")])
        >>> if file_path:
        ...     print(f"Selected: {file_path}")
    """
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    filetypes = filetypes or [("All files", "*.*")]
    
    file_path = filedialog.askopenfilename(
        title=title,
        initialdir=initial_dir or os.getcwd(),
        filetypes=filetypes
    )
    
    root.destroy()
    return file_path if file_path else None


def prompt_text(title: str, prompt: str, initial_value: str = "", use_terminal: bool = False) -> Optional[str]:
    """
    Open a text input dialog (GUI or terminal).
    
    Args:
        title: Dialog window title (GUI) or prompt prefix (terminal)
        prompt: Prompt message to display
        initial_value: Initial value in text box
        use_terminal: If True, use terminal input instead of GUI
        
    Returns:
        Entered text, or None if cancelled
        
    Example:
        >>> label = prompt_text("Model Label", "Enter label (e.g., '128ch'):", "128ch")
        >>> if label:
        ...     print(f"Label: {label}")
    """
    if use_terminal:
        # Terminal-based input
        if initial_value:
            result = input(f"{prompt} [{initial_value}]: ").strip()
            return result if result else initial_value
        else:
            result = input(f"{prompt}: ").strip()
            return result if result else None
    
    # GUI-based input
    try:
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        result = simpledialog.askstring(
            title=title,
            prompt=prompt,
            initialvalue=initial_value
        )
        
        root.destroy()
        return result
    except Exception:
        # Fallback to terminal if GUI fails
        return prompt_text(title, prompt, initial_value, use_terminal=True)


def prompt_yes_no(title: str, message: str) -> bool:
    """
    Open a GUI yes/no confirmation dialog.
    
    Args:
        title: Dialog window title
        message: Message to display
        
    Returns:
        True if yes, False if no
        
    Example:
        >>> if prompt_yes_no("Save Config", "Save settings for future use?"):
        ...     print("User chose yes")
    """
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    result = messagebox.askyesno(title=title, message=message)
    
    root.destroy()
    return result


def get_or_prompt_value(
    arg_value: Any,
    config: Dict[str, Any],
    key: str,
    prompt_func,
    required: bool = True,
    **prompt_kwargs
) -> Tuple[Any, bool]:
    """
    Get value from argument, config file, or prompt user (in that priority order).
    
    Args:
        arg_value: Value from command-line argument (highest priority)
        config: Configuration dictionary
        key: Key to look up in config
        prompt_func: Function to call for prompting (e.g., prompt_directory, prompt_text)
        required: If True, exit if no value provided
        **prompt_kwargs: Additional arguments to pass to prompt_func
        
    Returns:
        Tuple of (value, was_prompted) where was_prompted is True if user was asked for input
        
    Example:
        >>> config = load_simple_config('.config')
        >>> root_dir, prompted = get_or_prompt_value(
        ...     args.root_dir, config, 'root_dir', prompt_directory,
        ...     title="Select Model Directory"
        ... )
    """
    # 1. Check command line args (highest priority)
    if arg_value:
        return arg_value, False
    
    # 2. Check config file
    if key in config:
        return config[key], False
    
    # 3. Prompt user
    value = prompt_func(**prompt_kwargs)
    
    if not value and required:
        raise ValueError(f"{key} is required but not provided")
    
    return value, True
