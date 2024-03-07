import cv2
import numpy as np
import models.coco.pairs as coco
import models.body_25.pairs as body
import matplotlib.pyplot as plt
from Pose.PoseModel import NetworkModel
from Pose.Poser import PoseCalculator

colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
          [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
          [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]

net = NetworkModel(coco.PROTO_PATH, coco.CAFFE_PATH)
net.init_network()
img = cv2.imread("dance.jpg")
frameClone = img.copy()
output = net.get_output(img)

calculator = PoseCalculator(coco.COCO_MODEL, output, img)
personwiseKeypoints = calculator.get_plottable_key_points()
print(f"pkw: {personwiseKeypoints.shape}")
# 612, 408


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Whazzzup World")
    print(f"shape {output.shape}")
    print(f"plottable {personwiseKeypoints.shape}")
    print(f"plottable[1] {personwiseKeypoints}")
    for i in range(17):
        x = len(personwiseKeypoints)
        for n in range(len(personwiseKeypoints)):
            index = personwiseKeypoints[n][np.array(coco.COCO_MODEL.pose_pairs[i])]
            if -1 in index:
                continue
            B = np.int32(calculator.key_point_list[index.astype(int), 0])
            A = np.int32(calculator.key_point_list[index.astype(int), 1])
            cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

    cv2.imshow("Detected Pose", frameClone)
    cv2.waitKey(0)
    # cv2.imshow("Picture", img)
    # cv2.waitKey(0)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.imshow(mapMask, alpha=0.6)
    # plt.show()
