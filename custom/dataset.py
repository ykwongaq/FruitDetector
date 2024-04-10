from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import (
    Compose,
    Format,
    LetterBox,
)

from ultralytics.utils import LOGGER
from custom.augment import v8_transforms


class CustomDataset(YOLODataset):

    def __init__(self, *args, data=None, task="detect", **kwargs):
        super().__init__(*args, data=data, task=task, **kwargs)

    def __getitem__(self, index):
        """Returns a single item from the dataset."""
        return super().__getitem__(index)

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose(
                [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)]
            )
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms
