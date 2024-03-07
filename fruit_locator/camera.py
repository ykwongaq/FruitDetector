import numpy as np


class Intrinsic:

    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def __str__(self):
        return f"Intrinsic(fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy})"

    def to_matrix(self):
        """
        Get numpy matrix of the intrinsic parameters
        :return: 3x3 numpy matrix
        """
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
