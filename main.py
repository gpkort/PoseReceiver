import cv2
import numpy as np
import models.coco.pairs as coco
import models.body_25.pairs as body
import matplotlib.pyplot as plt
from Pose.PoseModel import NetworkModel
from Pose.Poser import PoseCalculator, keypoints_to_json

colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
          [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
          [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]

net = NetworkModel(coco.PROTO_PATH, coco.CAFFE_PATH)
net.init_network()
img = cv2.imread("dance.jpg")
frameClone = img.copy()
output = net.get_output(img)

calculator = PoseCalculator(coco.COCO_MODEL, output, img)
person_key_points = calculator.get_plottable_key_points()


# 612, 408

def draw_lines(kp_list: np.ndarray, pose_pairs: list, pk_points: np.ndarray, background: np.ndarray):
    for i in range(17):
        x = len(person_key_points)
        for n in range(len(pk_points)):
            index = pk_points[n][np.array(pose_pairs[i])]
            if -1 in index:
                continue
            B = np.int32(kp_list[index.astype(int), 0])
            A = np.int32(kp_list[index.astype(int), 1])
            cv2.line(background, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

    cv2.imshow("Detected Pose", frameClone)
    cv2.waitKey(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Whazzzup World")
    print(f"shape {output.shape}")
    print(f"plottable {person_key_points[:2]}")

    test = [1, 2, 3, 4, 5, 6, 7, 8]
    print(test[0:-1])
    js = keypoints_to_json(calculator.key_point_list, coco.COCO_MODEL.pose_pairs, person_key_points, frameClone)
    print(js)


    # draw_lines(calculator.key_point_list, coco.COCO_MODEL.pose_pairs, person_key_points, frameClone)
