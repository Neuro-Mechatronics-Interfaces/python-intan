#!/usr/bin/env python3
"""
predict.py

Unified CLI for EMG gesture prediction across all modes:
- file:   Offline prediction from single RHD file
- batch:  Batch prediction across multiple RHD files with aggregated metrics
- record: Device recording mode (fixed duration)
- stream: Real-time streaming prediction with optional LSL

Examples:
  # Offline single file
  python predict.py --mode file --file_path data.rhd --root_dir model/ --verbose

  # Batch processing
  python predict.py --mode batch --rhd_glob "raw/**/*.rhd" --root_dir model/ --events_dir events/

  # Device recording (10 seconds)
  python predict.py --mode record --seconds 10 --root_dir model/ --label 128ch

  # Real-time streaming with LSL
  python predict.py --mode stream --root_dir model/ --infer_hz 20 --smooth_k 5 --use_lsl

Mode-specific help:
  python predict.py --mode file --help
  python predict.py --mode batch --help
  python predict.py --mode record --help
  python predict.py --mode stream --help

Configuration:
  Create a .gesture_config file in this directory to store default values.
  This config is shared across all gesture_classifier scripts (dataset building,
  training, and prediction). See .gesture_config.example for all available options.
  
  Example .gesture_config:
    root_dir=/path/to/project
    label=128ch
    verbose=true
"""

import sys
import argparse
from pathlib import Path

# Add gesture_classifier to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import config utilities from intan.io
from intan.io import (
    load_simple_config,
    save_simple_config,
    prompt_directory,
    prompt_file,
    prompt_text,
    prompt_yes_no,
    get_or_prompt_value,
)

CONFIG_FILE = Path(__file__).parent / ".gesture_config"


def create_parser():
    """Create the main argument parser with subparsers for each mode."""
    parser = argparse.ArgumentParser(
        description="Unified EMG gesture prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Prediction mode", required=True)
    
    # ============================================================================
    # FILE MODE (3a_predict_from_rhd.py)
    # ============================================================================
    file_parser = subparsers.add_parser(
        "file",
        help="Offline prediction from single RHD file",
        description="Predict gestures from a single .rhd file using trained model"
    )
    file_parser.add_argument("--config_file", type=str, help="Configuration file path")
    file_parser.add_argument("--root_dir", type=str,
                            help="Directory containing trained model/metadata (prompts if not provided)")
    file_parser.add_argument("--file_path", type=str,
                            help="Path to .rhd file to evaluate (prompts if not provided)")
    file_parser.add_argument("--events_file", type=str, default=None,
                            help="Optional path to events file for evaluation")
    file_parser.add_argument("--label", type=str, default="",
                            help="Model label/tag (e.g., '128ch', prompts if not provided)")
    file_parser.add_argument("--window_ms", type=int, default=None,
                            help="Override window size (else from metadata)")
    file_parser.add_argument("--step_ms", type=int, default=None,
                            help="Override step size (else from metadata)")
    file_parser.add_argument("--verbose", action="store_true",
                            help="Enable verbose logging")
    
    # ============================================================================
    # BATCH MODE (3b_batch_predict_from_rhd.py)
    # ============================================================================
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch prediction across multiple RHD files",
        description="Process multiple RHD files and generate aggregated metrics"
    )
    batch_parser.add_argument("--root_dir", type=str,
                             help="Directory containing trained model/metadata (prompts if not provided)")
    
    # Mutually exclusive: glob pattern OR file list
    batch_group = batch_parser.add_mutually_exclusive_group()
    batch_group.add_argument("--rhd_glob", type=str,
                            help='Glob pattern like "raw/**/*.rhd" (prompts if not provided)')
    batch_group.add_argument("--rhd_files", nargs="+",
                            help="Explicit list of .rhd files")
    
    batch_parser.add_argument("--events_dir", type=str, default=None,
                             help="Directory with .event files (optional)")
    batch_parser.add_argument("--label", type=str,
                             help="Model label/tag (e.g., '128ch', prompts if not provided)")
    batch_parser.add_argument("--window_ms", type=int, default=None,
                             help="Override window size (else from metadata)")
    batch_parser.add_argument("--step_ms", type=int, default=None,
                             help="Override step size (else from metadata)")
    batch_parser.add_argument("--zero_division", type=int, default=0,
                             help="Value for zero_division in classification report")
    batch_parser.add_argument("--save_eval", action="store_true",
                             help="Save aggregated evaluation to JSON")
    batch_parser.add_argument("--verbose", action="store_true",
                             help="Enable verbose logging")
    
    # ============================================================================
    # RECORD MODE (3c_predict_from_device_record.py)
    # ============================================================================
    record_parser = subparsers.add_parser(
        "record",
        help="Device recording mode (fixed duration)",
        description="Record EMG from device for fixed duration and predict"
    )
    record_parser.add_argument("--root_dir", type=str,
                              help="Directory containing trained model/metadata (prompts if not provided)")
    record_parser.add_argument("--label", type=str, default="",
                              help="Model label/tag (e.g., '128ch', prompts if not provided)")
    record_parser.add_argument("--seconds", type=float, default=10.0,
                              help="Recording duration in seconds (default: 10)")
    record_parser.add_argument("--event_file", type=str, default=None,
                              help="Optional events file for evaluation")
    record_parser.add_argument("--window_ms", type=int, default=None,
                              help="Override window size (else from metadata)")
    record_parser.add_argument("--step_ms", type=int, default=None,
                              help="Override step size (else from metadata)")
    record_parser.add_argument("--verbose", action="store_true",
                              help="Enable verbose logging")
    
    # ============================================================================
    # STREAM MODE (3d_predict_from_device_realtime.py)
    # ============================================================================
    stream_parser = subparsers.add_parser(
        "stream",
        help="Real-time streaming prediction",
        description="Stream EMG from device and predict in real-time"
    )
    stream_parser.add_argument("--root_dir", type=str,
                              help="Directory containing trained model/metadata (prompts if not provided)")
    stream_parser.add_argument("--label", type=str, default="",
                              help="Model label/tag (e.g., '128ch', prompts if not provided)")
    stream_parser.add_argument("--window_ms", type=int, default=None,
                              help="Override window size (else from metadata)")
    stream_parser.add_argument("--step_ms", type=int, default=None,
                              help="Override step size (else from metadata)")
    
    # Inference rate (mutually exclusive)
    rate_group = stream_parser.add_mutually_exclusive_group()
    rate_group.add_argument("--infer_hz", type=float, default=None,
                           help="Inference rate in Hz (e.g., 20)")
    rate_group.add_argument("--infer_ms", type=float, default=None,
                           help="Inference period in ms (e.g., 100)")
    
    stream_parser.add_argument("--smooth_k", type=int, default=1,
                              help="Majority vote window size (default: 1, no smoothing)")
    stream_parser.add_argument("--seconds_total", type=float, default=None,
                              help="Total streaming duration (None = infinite)")
    stream_parser.add_argument("--use_lsl", action="store_true",
                              help="Enable LSL publishing of predictions")
    stream_parser.add_argument("--verbose", action="store_true",
                              help="Enable verbose logging")
    
    return parser


def main():
    parser = create_parser()
    
    # Check if running in interactive mode (no mode specified)
    import sys
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help']):
        # Interactive mode - prompt for mode first
        print("\n=== EMG Gesture Prediction ===\n")
        print("Available modes:")
        print("  1. file   - Predict from single RHD file")
        print("  2. batch  - Batch prediction across multiple files")
        print("  3. record - Record from device (fixed duration)")
        print("  4. stream - Real-time streaming prediction")
        print()
        
        mode_map = {'1': 'file', '2': 'batch', '3': 'record', '4': 'stream'}
        mode_input = input("Select mode (1-4 or name): ").strip().lower()
        mode = mode_map.get(mode_input, mode_input)
        
        if mode not in ['file', 'batch', 'record', 'stream']:
            print(f"Invalid mode: {mode}")
            sys.exit(1)
        
        # Insert mode into argv for parser (as positional argument, not flag)
        sys.argv.append(mode)
    
    args = parser.parse_args()
    
    # Load config file
    config = load_simple_config(CONFIG_FILE)
    
    # Determine if we should use terminal prompts for text inputs (when running interactively)
    use_terminal = len(sys.argv) <= 3  # Only mode was specified (or interactive)
    
    # Track if we prompted for any values (to offer saving config)
    prompted_for_values = False
    
    # Helper to get prompt function with terminal preference for text inputs only
    def get_text_prompt_func():
        return lambda **kw: prompt_text(**{**kw, 'use_terminal': use_terminal})
    
    # Get root_dir (always use GUI directory picker)
    root_dir, was_prompted = get_or_prompt_value(
        arg_value=args.root_dir,
        config=config,
        key='root_dir',
        prompt_func=prompt_directory,
        title="Select Project Root Directory",
        initial_dir=config.get('root_dir') if 'root_dir' in config else None
    )
    # Save to config immediately
    if was_prompted:
        config['root_dir'] = root_dir
        save_simple_config(config, CONFIG_FILE, "EMG Gesture Prediction Configuration")
        print("[*] Saved root_dir to .gesture_config")
    else:
        # Save CLI-provided root_dir to config if not already there
        if 'root_dir' not in config or config.get('root_dir') != root_dir:
            config['root_dir'] = root_dir
            save_simple_config(config, CONFIG_FILE, "EMG Gesture Prediction Configuration")
            print("[*] Saved root_dir to .gesture_config")
    
    # Get label (prompt if needed) - only for modes that require it
    label = ""
    if args.mode in ('batch', 'record', 'stream'):
        label_required = args.mode == 'batch'  # batch requires label
        label_value = getattr(args, 'label', None)
        
        if not label_value and 'label' in config:
            label = config['label']
        elif not label_value and label_required:
            label, was_prompted = get_or_prompt_value(
                arg_value=None,
                config=config,
                key='label',
                prompt_func=get_text_prompt_func(),
                title="Model label",
                prompt="Model label (e.g., '128ch')",
                initial_value=config.get('label', '')
            )
            # Save to config immediately
            if was_prompted:
                config['label'] = label
                save_simple_config(config, CONFIG_FILE, "EMG Gesture Prediction Configuration")
                print("[*] Saved label to .gesture_config")
        else:
            label = label_value or config.get('label', '')
    else:
        # file mode uses label from args or config
        label = getattr(args, 'label', '') or config.get('label', '')
    
    # Import prediction mode functions from intan package
    from intan.ml import predict_file, predict_batch, predict_from_device, predict_realtime_stream
    
    # Dispatch to appropriate mode
    if args.mode == "file":
        # Get file_path (prompt if needed)
        file_path = args.file_path
        if not file_path:
            file_path = prompt_file(
                title="Select RHD File",
                initial_dir=root_dir,
                filetypes=[("RHD files", "*.rhd"), ("All files", "*.*")]
            )
            if not file_path:
                print("Error: file_path is required")
                sys.exit(1)
        
        # Run file prediction mode
        predict_file(
            root_dir=root_dir,
            file_path=file_path,
            label=label,
            window_ms=args.window_ms,
            step_ms=args.step_ms,
            events_file=args.events_file,
            save_predictions=True,
            verbose=args.verbose or config.get('verbose', False)
        )
        
    elif args.mode == "batch":
        # Get rhd_glob or rhd_files (prompt if needed)
        rhd_glob = args.rhd_glob
        rhd_files = args.rhd_files
        
        if not rhd_glob and not rhd_files:
            rhd_glob = get_text_prompt_func()(
                title="RHD file pattern",
                prompt="RHD file glob pattern (e.g., 'raw/**/*.rhd')",
                initial_value=config.get('rhd_glob', '**/*.rhd')
            )
            if not rhd_glob:
                print("Error: rhd_glob or rhd_files is required")
                sys.exit(1)
            # Save to config immediately
            config['rhd_glob'] = rhd_glob
            save_simple_config(config, CONFIG_FILE, "EMG Gesture Prediction Configuration")
            print("[*] Saved rhd_glob to .gesture_config")
        
        # Run batch prediction mode
        predict_batch(
            root_dir=root_dir,
            rhd_glob=rhd_glob,
            rhd_files=rhd_files,
            events_dir=args.events_dir,
            label=label,
            window_ms=args.window_ms,
            step_ms=args.step_ms,
            zero_division=args.zero_division,
            save_eval=args.save_eval,
            verbose=args.verbose or config.get('verbose', False)
        )
        
    elif args.mode == "record":
        # Run record prediction mode
        predict_from_device(
            root_dir=root_dir,
            label=label,
            seconds=args.seconds,
            event_file=args.event_file,
            window_ms=args.window_ms,
            step_ms=args.step_ms,
            verbose=args.verbose or config.get('verbose', False)
        )
        
    elif args.mode == "stream":
        # Convert infer_hz/infer_ms to period
        infer_period_s = None
        if args.infer_hz:
            infer_period_s = 1.0 / args.infer_hz
        elif args.infer_ms:
            infer_period_s = args.infer_ms / 1000.0
        
        # Run streaming prediction mode
        predict_realtime_stream(
            root_dir=root_dir,
            label=label,
            window_ms=args.window_ms,
            step_ms=args.step_ms,
            infer_period_s=infer_period_s,
            smooth_k=args.smooth_k,
            seconds_total=args.seconds_total,
            use_lsl=args.use_lsl,
            verbose=args.verbose or config.get('verbose', False)
        )
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
