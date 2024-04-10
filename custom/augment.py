from ultralytics.data.augment import (
    Compose,
    LetterBox,
    Mosaic,
    CopyPaste,
    RandomPerspective,
    MixUp,
    RandomHSV,
    RandomFlip,
)

from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_version

import albumentations as A
import numpy as np

import cv2


class RegionalBlur(A.ImageOnlyTransform):

    def __init__(
        self,
        always_apply=False,
        p=0.5,
        max_number_of_dots=15,
        min_radius=0.01,
        max_radius=0.1,
        blur_value=(151, 211),
    ):
        super(RegionalBlur, self).__init__(always_apply, p)
        self.max_number_of_dots = max_number_of_dots
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.blur_value = blur_value

    def apply(self, img, **params):
        blur = A.GaussianBlur(blur_limit=self.blur_value, p=1.0)
        blurred_image = blur(image=img)["image"]
        dimension = max(img.shape[:2])
        mask = np.zeros_like(img, dtype=np.uint8)
        for _ in range(self.max_number_of_dots):
            x = np.random.randint(0, img.shape[1])
            y = np.random.randint(0, img.shape[0])
            r = np.random.randint(
                self.min_radius * dimension, self.max_radius * dimension
            )
            cv2.circle(mask, (x, y), r, [255, 255, 255], -1)
        output_image = np.where(mask == [0, 0, 0], img, blurred_image)
        return output_image


class Albumentations:
    """
    Albumentations transformations.

    Optional, uninstall package to disable. Applies Blur, Median Blur, convert to grayscale, Contrast Limited Adaptive
    Histogram Equalization, random change of brightness and contrast, RandomGamma and lowering of image quality by
    compression.
    """

    def __init__(self, hyp, p=1.0):
        """Initialize the transform object for YOLO bbox formatted params."""
        self.p = p
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            # Transforms
            T = [
                A.AdvancedBlur(p=hyp.advanced_blur),
                A.GaussianBlur(p=hyp.gaussian_blur),
                A.ColorJitter(p=hyp.color_jitter),
                A.Downscale(p=hyp.downscale),
                A.ImageCompression(quality_lower=75, p=hyp.image_compression),
                A.ISONoise(p=hyp.iso_noise),
                A.RandomBrightnessContrast(p=hyp.brightness_contrast),
                A.RandomGamma(p=hyp.gamma),
                A.RandomShadow(p=hyp.random_shadow),
                A.Sharpen(p=hyp.sharpen),
                RegionalBlur(
                    p=hyp.regional_blur,
                    max_number_of_dots=hyp.max_number_of_dots,
                    min_radius=hyp.min_radius,
                    max_radius=hyp.max_radius,
                    blur_value=hyp.blur_value,
                ),
            ]
            self.transform = A.Compose(
                T,
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            )

            LOGGER.info(
                prefix
                + ", ".join(
                    f"{x}".replace("always_apply=False, ", "") for x in T if x.p
                )
            )
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, labels):
        """Generates object detections and returns a dictionary with detection results."""
        im = labels["img"]
        cls = labels["cls"]
        if len(cls):
            labels["instances"].convert_bbox("xywh")
            labels["instances"].normalize(*im.shape[:2][::-1])
            bboxes = labels["instances"].bboxes
            # TODO: add supports of segments and keypoints
            if self.transform and random.random() < self.p:
                new = self.transform(
                    image=im, bboxes=bboxes, class_labels=cls
                )  # transformed
                if len(new["class_labels"]) > 0:  # skip update if no bbox in new im
                    labels["img"] = new["image"]
                    labels["cls"] = np.array(new["class_labels"])
                    bboxes = np.array(new["bboxes"], dtype=np.float32)
            labels["instances"].update(bboxes=bboxes)
        return labels


def v8_transforms(dataset, imgsz, hyp, stretch=False):
    """Convert images to a size suitable for YOLOv8 training."""
    pre_transform = Compose(
        [
            Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
            CopyPaste(p=hyp.copy_paste),
            RandomPerspective(
                degrees=hyp.degrees,
                translate=hyp.translate,
                scale=hyp.scale,
                shear=hyp.shear,
                perspective=hyp.perspective,
                pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
            ),
        ]
    )
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning(
                "WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'"
            )
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(
                f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}"
            )

    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            Albumentations(p=1.0),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # transforms
