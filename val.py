import argparse
import os

from ultralytics import YOLO


def main(args):
    model_path = args.model
    assert os.path.exists(model_path), f"Model not found at {model_path}"
    assert model_path.endswith(".pt"), "Model should be a .pt file"

    data_config_path = args.data
    assert os.path.exists(
        data_config_path
    ), f"Data config not found at {data_config_path}"
    assert data_config_path.endswith(".yaml"), "Data config should be a .yaml file"

    model = YOLO(model_path)
    metrics = model.val(
        data=data_config_path, imgsz=args.imgsz, batch=args.batch, save_json=True
    )
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 validation")
    parser.add_argument("--model", type=str, help="Path to the model")
    parser.add_argument(
        "--data", type=str, default="./configs/coco128.yaml", help="Path to the data"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")

    args = parser.parse_args()
    main(args)
