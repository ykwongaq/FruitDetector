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

    imgsz = args.imgsz
    if "," in imgsz:
        imgsz = tuple(map(int, imgsz.split(",")))
    else:
        imgsz = int(imgsz)

    model = YOLO(model_path)
    results = model.predict(
        data_folder,
        save=True,
        conf=args.conf,
        imgsz=imgsz,
        save_txt=True,
        save_conf=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv5 Training")
    parser.add_argument("--model", type=str, help="Path to the testing model")
    parser.add_argument("--data", type=str, help="Path to the data folder")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument(
        "--imgsz",
        type=str,
        default="640",
        help="Defines the image size for inference. Can be a single integer 640 for square resizing or a (height, width) tuple. For example 640,640",
    )
    args = parser.parse_args()
    main(args)
