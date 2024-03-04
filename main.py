import cv2
import matplotlib
import numpy as np
import models.coco.pairs as coco
import models.body_25.pairs as body
import matplotlib.pyplot as plt

net = cv2.dnn.readNetFromCaffe(body.PROTO_PATH, body.CAFFE_PATH)

img = cv2.imread("dance.jpg")
# Fix the input Height and get the width according to the Aspect Ratio
inHeight = 368
inWidth = int((inHeight / img.shape[0]) * img.shape[1])

inpBlob = cv2.dnn.blobFromImage(img,
                                1.0 / 255,
                                (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)
net.setInput(inpBlob)
output = net.forward()

i = 0
probMap = output[0, i, :, :]
probMap = cv2.resize(probMap, (img.shape[1], img.shape[0]))

mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)
mapMask = np.uint8(mapSmooth > 0.1)

_, contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# for each blob find the maxima
# for cnt in contours:
#     blobMask = np.zeros(mapMask.shape)
#     blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
#     maskedProbMap = mapSmooth * blobMask
#     _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
#     keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))



# 612, 408


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Whazzzup World")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(mapMask, alpha=0.6)
    plt.show()