import os
import numpy as np
import cv2


def numpy_to_image(input_path: str, output_path: str) -> None:
    """
    Convert the given .npy files to .png images
    :param input_path: Path to input numpy file or folder
    :param output_path: Path to output images
    :return: None
    """

    assert os.path.exists(input_path), f"Input path not found at {input_path}"

    # Determine list of input .npy files
    input_files = []
    if os.path.isfile(input_path):
        input_files.append(input_path)
    else:
        input_files = [
            os.path.join(input_path, file)
            for file in os.listdir(input_path)
            if file.endswith(".npy")
        ]

    # Create the output folder if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Loop through each input file
    for input_file in input_files:
        # Load the numpy array from the input file
        image = np.load(input_file)

        # Get the filename without extension
        filename = os.path.splitext(os.path.basename(input_file))[0]

        # Save the image as a PNG file
        output_file = os.path.join(output_path, f"{filename}.png")

        print(f"Convert {input_file} to {output_file}")

        cv2.imwrite(output_file, image)
