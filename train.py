import argparse
import os
import yaml

from datetime import datetime

from ultralytics import YOLO


def main(args):
    dataset_config_path = args.dataset_config
    assert os.path.exists(
        dataset_config_path
    ), f"Dataset config file not found at {dataset_config_path}"
    assert dataset_config_path.endswith(
        ".yaml"
    ), "Dataset config file must be a .yaml file"

    pretrained_weights_path = args.pretrained_weights
    assert os.path.exists(
        pretrained_weights_path
    ), f"Pretrained weights not found at {pretrained_weights_path}"
    assert pretrained_weights_path.endswith(
        ".pt"
    ), "Pretrained weights must be a .pt file"

    devices = args.devices
    if devices != "cpu":
        devices = [
            int(device) for device in args.devices.split(",") if device.isdigit()
        ]

    model = YOLO(args.pretrained_weights)

    imgsz = args.imgsz
    if "," in imgsz:
        imgsz = tuple(map(int, imgsz.split(",")))
    else:
        imgsz = int(imgsz)

    # with open(dataset_config_path, "r") as file:
    #     dataset_config = yaml.safe_load(file)
    # augmentations = dataset_config.get("augmentations", None)

    results = model.train(
        model=pretrained_weights_path,
        data=dataset_config_path,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=imgsz,
        save_period=args.save_period,
        workers=args.workers,
        device=devices,
        verbose=args.verbose,
        project=args.output,
        name=args.name,
        resume=args.resume,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="./configs/fruit_dataset.yaml",
        help="Path to dataset config file",
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default="./weights/yolov8x.pt",
        help="Path to pretrained weights",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="0,1,2,3",
        help="Comma separated list of GPU device IDs to use for training",
    )
    parser.add_argument(
        "--epochs", type=int, default=400, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--imgsz",
        type=str,
        default="360,640",
        help="Image size for training.  Can be a single integer 640 for square resizing or a (height, width) tuple. For example 640,640",
    )
    parser.add_argument(
        "--save_period",
        type=int,
        default=50,
        help="Number of epochs between saving model weights",
    )
    parser.add_argument(
        "--workers", type=int, default=32, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs",
        help="Path to save model weights and logs",
    )
    parser.add_argument(
        "--exist_ok",
        action="store_true",
        help="If True, allows overwriting of an existing project/name directory. Useful for iterative experimentation "
        "without needing to manually clear previous outputs.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from the last checkpoint"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Print verbose logs during training"
    )
    # The default output name of the train is the time of training
    default_output_name = "yolov8_" + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    parser.add_argument(
        "--name",
        type=str,
        default=default_output_name,
        help="Name of the model to save",
    )

    args = parser.parse_args()
    main(args)
