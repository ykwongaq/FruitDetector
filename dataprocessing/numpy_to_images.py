import argparse
import os
import numpy as np
import cv2


def main(args):

    input_path = args.input
    assert os.path.exists(input_path), f"Input path not found at {input_path}"

    input_files = []
    if os.path.isfile(input_path):
        input_files.append(input_path)
    else:
        input_files = [
            os.path.join(input_path, file)
            for file in os.listdir(input_path)
            if file.endswith(".npy")
        ]

    output_folder = args.output
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Loop through each input file
    for input_file in input_files:
        # Load the numpy array from the input file
        image = np.load(input_file)

        # Get the filename without extension
        filename = os.path.splitext(os.path.basename(input_file))[0]

        # Save the image as a PNG file
        output_file = os.path.join(output_folder, f"{filename}.png")

        print(f"Convert {input_file} to {output_file}")

        cv2.imwrite(output_file, image)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert numpy to images")
    parser.add_argument(
        "--input",
        type=str,
        default="/mnt/hdd/davidwong/data/FruitDataset/dataset/input_numpy",
        help="Path to input numpy file or folder",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/mnt/hdd/davidwong/data/FruitDataset/dataset/input_images",
        help="Path to output images",
    )

    args = parser.parse_args()
    main(args)
