import numpy as np
import argparse
import os

def inspect_npz(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"Inspecting file: {file_path}")
    try:
        data = np.load(file_path)
        print("File contains the following elements:")
        for key in data.files:
            print(f"  - {key}: shape={data[key].shape}, dtype={data[key].dtype}")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect the contents of a .npz file.')
    parser.add_argument('file_path', type=str, help='The path to the .npz file.')
    args = parser.parse_args()
    inspect_npz(args.file_path)
