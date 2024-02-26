import argparse
import os

from ultralytics import YOLO


def main(args):
    model_path = args.model
    assert os.path.exists(model_path), f"Model path {model_path} does not exist"
    assert model_path.endswith(".pt"), f"Model path {model_path} is not a .pt file"

    data_folder = args.data
    assert os.path.exists(data_folder), f"Data folder {data_folder} does not exist"
    assert os.path.isdir(data_folder), f"Data folder {data_folder} is not a directory"

    model = YOLO(model_path)
    results = model.predict(data_folder, save=True, conf=args.conf)
    for result in results:
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv5 Training")
    parser.add_argument("--model", type=str, help="Path to the testing model")
    parser.add_argument("--data", type=str, help="Path to the data folder")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()
    main(args)
