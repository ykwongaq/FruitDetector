import argparse
import os
import yaml

from datetime import datetime

# from custom.trainer import CustomTrainer
from ultralytics import YOLO

# from ultralytics.data.augment import (
#     Compose,
#     LetterBox,
#     Mosaic,
#     CopyPaste,
#     RandomPerspective,
#     MixUp,
#     RandomHSV,
#     RandomFlip,
# )

# from ultralytics.data.dataset import YOLODataset
# from ultralytics.data.augment import (
#     Compose,
#     Format,
#     LetterBox,
# )
# from ultralytics.models.yolo.detect import DetectionTrainer
# from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first

# from ultralytics.data import build_dataloader
# import random

# from ultralytics.utils import LOGGER, colorstr
# from ultralytics.utils.checks import check_version
# from ultralytics.utils.torch_utils import de_parallel
# import albumentations as A
# import numpy as np

# import cv2


# class RegionalBlur(A.ImageOnlyTransform):

#     def __init__(
#         self,
#         always_apply=False,
#         p=0.5,
#         max_number_of_dots=15,
#         min_radius=0.01,
#         max_radius=0.1,
#         blur_value=(151, 211),
#     ):
#         super(RegionalBlur, self).__init__(always_apply, p)
#         self.max_number_of_dots = max_number_of_dots
#         self.max_radius = max_radius
#         self.min_radius = min_radius
#         self.blur_value = blur_value

#     def apply(self, img, **params):
#         blur = A.GaussianBlur(blur_limit=self.blur_value, p=1.0)
#         blurred_image = blur(image=img)["image"]
#         dimension = max(img.shape[:2])
#         mask = np.zeros_like(img, dtype=np.uint8)
#         for _ in range(self.max_number_of_dots):
#             x = np.random.randint(0, img.shape[1])
#             y = np.random.randint(0, img.shape[0])
#             r = np.random.randint(
#                 self.min_radius * dimension, self.max_radius * dimension
#             )
#             cv2.circle(mask, (x, y), r, [255, 255, 255], -1)
#         output_image = np.where(mask == [0, 0, 0], img, blurred_image)
#         return output_image


# class Albumentations:
#     """
#     Albumentations transformations.

#     Optional, uninstall package to disable. Applies Blur, Median Blur, convert to grayscale, Contrast Limited Adaptive
#     Histogram Equalization, random change of brightness and contrast, RandomGamma and lowering of image quality by
#     compression.
#     """

#     def __init__(self, hyp, p=1.0):
#         """Initialize the transform object for YOLO bbox formatted params."""
#         self.p = p
#         self.transform = None
#         prefix = colorstr("albumentations: ")
#         try:
#             import albumentations as A

#             check_version(A.__version__, "1.0.3", hard=True)  # version requirement

#             # Transforms
#             T = [
#                 A.AdvancedBlur(p=hyp.advanced_blur),
#                 A.GaussianBlur(p=hyp.gaussian_blur),
#                 A.ColorJitter(p=hyp.color_jitter),
#                 A.Downscale(p=hyp.downscale),
#                 A.ImageCompression(quality_lower=75, p=hyp.image_compression),
#                 A.ISONoise(p=hyp.iso_noise),
#                 A.RandomBrightnessContrast(p=hyp.brightness_contrast),
#                 A.RandomGamma(p=hyp.gamma),
#                 A.RandomShadow(p=hyp.random_shadow),
#                 A.Sharpen(p=hyp.sharpen),
#                 RegionalBlur(
#                     p=hyp.regional_blur,
#                     max_number_of_dots=hyp.max_number_of_dots,
#                     min_radius=hyp.min_radius,
#                     max_radius=hyp.max_radius,
#                     blur_value=hyp.blur_value,
#                 ),
#             ]
#             self.transform = A.Compose(
#                 T,
#                 bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
#             )

#             LOGGER.info(
#                 prefix
#                 + ", ".join(
#                     f"{x}".replace("always_apply=False, ", "") for x in T if x.p
#                 )
#             )
#         except ImportError:  # package not installed, skip
#             pass
#         except Exception as e:
#             LOGGER.info(f"{prefix}{e}")

#     def __call__(self, labels):
#         """Generates object detections and returns a dictionary with detection results."""
#         im = labels["img"]
#         cls = labels["cls"]
#         if len(cls):
#             labels["instances"].convert_bbox("xywh")
#             labels["instances"].normalize(*im.shape[:2][::-1])
#             bboxes = labels["instances"].bboxes
#             # TODO: add supports of segments and keypoints
#             if self.transform and random.random() < self.p:
#                 new = self.transform(
#                     image=im, bboxes=bboxes, class_labels=cls
#                 )  # transformed
#                 if len(new["class_labels"]) > 0:  # skip update if no bbox in new im
#                     labels["img"] = new["image"]
#                     labels["cls"] = np.array(new["class_labels"])
#                     bboxes = np.array(new["bboxes"], dtype=np.float32)
#             labels["instances"].update(bboxes=bboxes)
#         return labels


# def v8_transforms(dataset, imgsz, hyp, stretch=False):
#     """Convert images to a size suitable for YOLOv8 training."""
#     pre_transform = Compose(
#         [
#             Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
#             CopyPaste(p=hyp.copy_paste),
#             RandomPerspective(
#                 degrees=hyp.degrees,
#                 translate=hyp.translate,
#                 scale=hyp.scale,
#                 shear=hyp.shear,
#                 perspective=hyp.perspective,
#                 pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
#             ),
#         ]
#     )
#     flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
#     if dataset.use_keypoints:
#         kpt_shape = dataset.data.get("kpt_shape", None)
#         if len(flip_idx) == 0 and hyp.fliplr > 0.0:
#             hyp.fliplr = 0.0
#             LOGGER.warning(
#                 "WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'"
#             )
#         elif flip_idx and (len(flip_idx) != kpt_shape[0]):
#             raise ValueError(
#                 f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}"
#             )

#     return Compose(
#         [
#             pre_transform,
#             MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
#             Albumentations(p=1.0),
#             RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
#             RandomFlip(direction="vertical", p=hyp.flipud),
#             RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
#         ]
#     )  # transforms


# class CustomDataset(YOLODataset):

#     def __init__(self, *args, data=None, task="detect", **kwargs):
#         super().__init__(*args, data=data, task=task, **kwargs)

#     def __getitem__(self, index):
#         """Returns a single item from the dataset."""
#         return super().__getitem__(index)

#     def build_transforms(self, hyp=None):
#         """Builds and appends transforms to the list."""
#         if self.augment:
#             hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
#             hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
#             transforms = v8_transforms(self, self.imgsz, hyp)
#         else:
#             transforms = Compose(
#                 [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)]
#             )
#         transforms.append(
#             Format(
#                 bbox_format="xywh",
#                 normalize=True,
#                 return_mask=self.use_segments,
#                 return_keypoint=self.use_keypoints,
#                 return_obb=self.use_obb,
#                 batch_idx=True,
#                 mask_ratio=hyp.mask_ratio,
#                 mask_overlap=hyp.overlap_mask,
#             )
#         )
#         return transforms


# class CustomTrainer(DetectionTrainer):

#     def build_dataset(self, img_path, mode="train", batch=None):
#         gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
#         return CustomDataset(
#             img_path=img_path,
#             imgsz=self.args.imgsz,
#             batch_size=batch,
#             augment=mode == "train",
#             hyp=self.args,
#             rect=self.args.rect or mode == "val",
#             cache=self.args.cache or None,
#             single_cls=self.args.single_cls or False,
#             stride=int(gs),
#             pad=0.0 if mode == "train" else 0.5,
#             prefix=colorstr(f"{mode}: "),
#             task=self.args.task,
#             classes=self.args.classes,
#             data=self.data,
#             fraction=self.args.fraction if mode == "train" else 1.0,
#         )


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

    imgsz = args.imgsz
    if "," in imgsz:
        imgsz = tuple(map(int, imgsz.split(",")))
    else:
        imgsz = int(imgsz)

    model = YOLO()

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
        # default="cpu",
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
