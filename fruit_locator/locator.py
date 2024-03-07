from ultralytics import YOLO
from .camera import Intrinsic
import yaml
import numpy as np
import os
import logging


class Locator:

    def __init__(self, config_path: str):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.color_intrinsic = Intrinsic(**self.config["color_camera"]).to_matrix()
        self.depth_intrinsic = Intrinsic(**self.config["depth_camera"]).to_matrix()

        self.yolo = YOLO(self.config["model"]["weight"])
        self.model_conf = self.config["model"]["conf"]
        self.image_height = self.config["model"]["height"]
        self.image_width = self.config["model"]["width"]

        # Set up the logging system
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)

        c_format = logging.Formatter("[INFO] %(name)s: %(message)s")
        c_handler.setFormatter(c_format)

        self.logger.addHandler(c_handler)

    def object_detection(self, color_image: np.ndarray):
        """
        Perform object detection on the given color image
        :param color_image: Color image HxWx3
        :return: List of YOLO results
        """
        assert type(color_image) == np.ndarray, "Color image must be a numpy array"
        results = self.yolo.predict(
            color_image,
            conf=self.model_conf,
            imgsz=(self.image_height, self.image_width),
        )
        return results

    def back_projection(self, center_points: list, depth_map: np.ndarray):
        """
        Back project the center points to xyz coordinates
        :param center_points: List of center points (x, y)
        :param depth_map: Depth map HxW
        :return: List of xyz coordinates
        """
        xyz = []
        for idx, center_point in enumerate(center_points):
            cx, cy = center_point
            x = np.array([[cx], [cy], [1]])
            X = np.linalg.inv(self.color_intrinsic) @ x

            z = depth_map[cy, cx]
            X = X * z
            X = tuple(X.flatten())
            xyz.append(X)

            self.logger.info(
                f"Center point {center_point}, depth {z}: {center_point} -> XYZ: {X}"
            )
        return xyz

    def locate(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        show_result: bool = False,
        save_result: bool = False,
        output_folder: str = "./location_results",
    ):
        """
        Locate the fruit in the given color and return the xyz coordinates
        :param color_image: Color image HxWx3
        :param depth_image: Depth image HxW
        :param show_result: Show the result if True
        :param save_result: Save the result to output_folder if True
        :param output_folder: Output folder
        :return: List of classes id and xyz coordinates
        """
        assert (
            color_image.shape[:2] == depth_image.shape
        ), "Color and depth image shape mismatch"
        assert type(color_image) == np.ndarray, "Color image must be a numpy array"
        assert type(depth_image) == np.ndarray, "Depth image must be a numpy array"

        # Perform object detection
        results = self.object_detection(color_image)

        # Visualization
        for idx, result in enumerate(results):
            if show_result:
                result.show()

            if save_result:
                # Create folder is not exist
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder, exist_ok=True)

                filename = f"result_{idx}.jpg"
                output_path = os.path.join(output_folder, filename)
                self.logger.info(f"Saving result to {output_path}")
                result.save(output_path)

        # Calculate center point
        classes = []
        center_points = []
        for result in results:
            for box in result.boxes:
                class_id = box.cls.cpu().numpy().item()
                classes.append(class_id)

                # box is a 2D array of shape (1, 4)
                xyxy = box.xyxy.cpu().numpy().astype(np.int32)[0]

                center_point = (xyxy[0] + xyxy[2]) // 2, (xyxy[1] + xyxy[3]) // 2
                center_points.append(center_point)

        # Back projection
        xyz = self.back_projection(center_points, depth_image)

        return classes, xyz
