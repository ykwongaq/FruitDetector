from fruit_locator.locator import Locator
import argparse
import os
import numpy as np
import cv2


def main(args):
    config_path = args.config
    assert os.path.exists(config_path), f"Config file not found at {config_path}"
    assert config_path.endswith(".yaml"), f"Config file must be a .yaml file"

    assert os.path.exists(args.image_path), f"Image file not found at {args.image_path}"
    assert os.path.exists(args.depth_path), f"Depth file not found at {args.depth_path}"

    # Read the image file
    image = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
    assert image is not None, f"Failed to read image file at {args.image_path}"

    # Load the depth map
    depth = np.load(args.depth_path)

    print(image.shape)
    locator = Locator(args.config)
    classes, xyz = locator.locate(
        image,
        depth,
        show_result=args.show_output,
        save_result=args.save_output,
        output_folder=args.output_folder,
    )
    print(classes)
    print(xyz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/locator.yaml")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--depth_path", type=str, required=True)
    parser.add_argument("--show_output", action="store_true")
    parser.add_argument("--save_output", action="store_true")
    parser.add_argument("--output_folder", type=str, default="./location_results")
    args = parser.parse_args()
    main(args)
