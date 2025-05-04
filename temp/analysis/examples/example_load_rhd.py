import os
import sys

# Get the absolute path of the root project directory
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root_path)

import argparse
from intan_utilities import rhd_utilities as rhd_utils
import pprint

def main():

    filename_default = 'assets/index_flexion/index_flexion.rhd'
    parser = argparse.ArgumentParser(description='Load RHD data')
    parser.add_argument('--filename', type=str, default=filename_default, help='Path to the RHD file')
    parser.add_argument('--verbose', type=bool, default=False, help='Enable text debugging output')
    args = parser.parse_args()

    if args.filename == filename_default:
        args.filename = os.path.join(root_path, args.filename)  # Convert relative path to absolute path

    # Load the data
    result, data_present = rhd_utils.load_file(args.filename, verbose=args.verbose)

    if not data_present:
        print(f'No data present in {args.filename}')
        return

    # display the keys in the result dictionary
    pp = pprint.PrettyPrinter(indent=3, width=200)  # Create a pretty printer object
    for key, val in result.items():
        print(f"\n{key}:")
        pp.pprint(val)

if __name__ == "__main__":
    main()


