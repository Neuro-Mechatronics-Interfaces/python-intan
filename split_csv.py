# Open a csv and save it into 4 parts
#
# Usage: python split_csv.py <input_file> <output_file_prefix> <num_parts>
# Example: python split_csv.py test.csv test 4

import pandas as pd
import sys
import numpy as np

def split_csv(input_file, output_file_prefix, num_parts):

    # Load the csv
    df = pd.read_csv(input_file)

    # Split the data
    split_data = np.array_split(df, num_parts)

    # Save the data
    for i, data in enumerate(split_data):
        data.to_csv(f"{output_file_prefix}_{i}.csv", index=False)
        print(f"Saved {output_file_prefix}_{i}.csv")

    print(f"Split {input_file} into {num_parts} parts")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python split_csv.py <input_file> <output_file_prefix> <num_parts>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file_prefix = sys.argv[2]
    num_parts = int(sys.argv[3])

    split_csv(input_file, output_file_prefix, num_parts)

