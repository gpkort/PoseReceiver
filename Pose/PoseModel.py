import cv2
import numpy as np
from enum import Enum
from math import pow


class ModelData:
    def __init__(self,
                 key_points: list,
                 pose_pairs: list,
                 map_index: list):
        self._key_points = list()
        self._pose_pairs = pose_pairs.copy()
        self._map_index = map_index.copy()

        if key_points is not None:
            self._key_points = key_points.copy()

        self._number_of_kp = len(self._key_points)

    @property
    def key_points(self) -> list:
        return self._key_points

    @property
    def pose_pairs(self) -> list:
        return self._pose_pairs

    @property
    def map_index(self) -> list:
        return self._map_index

    @property
    def num_of_kp(self) -> int:
        return self._number_of_kp


class ModelBackend(Enum):
    GPU = 0
    CPU = 1


class NetworkModel:
    def __init__(self, proto_file: str, caffe_file: str):

        self._number_of_points = 0
        self._proto_file = proto_file
        self._caffe_file = caffe_file
        self._network = None
        self._net_blob = None

    def init_network(self, backend: ModelBackend = ModelBackend.GPU):

        self._network = cv2.dnn.readNetFromCaffe(self._proto_file, self._caffe_file)
        if backend == ModelBackend.CPU:
            self._network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            self._network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self._network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

    def get_output(self,
                   image: np.ndarray,
                   aspect_height: int = 368,
                   scale: float = pow(255, -1),
                   mean: tuple = (0, 0, 0),
                   swapRB=False,
                   crop=False) -> np.ndarray:

        output = None
        if self._network is not None:
            aspect_width = int((aspect_height / image.shape[0]) * image.shape[1])
            self._net_blob = cv2.dnn.blobFromImage(image, scale, (aspect_width, aspect_height),
                                                   mean, swapRB, crop)
            self._network.setInput(self._net_blob)
            output = self._network.forward()

        return output
